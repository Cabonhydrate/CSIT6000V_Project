#!/usr/bin/env bash
# Launch vLLM inside a TACC GPU compute allocation, never on a login node.
set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'ERROR: This launcher must run inside an allocated Slurm job.' >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo 'ERROR: nvidia-smi not found. Are you on a GPU compute node?' >&2
  exit 2
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo 'ERROR: vllm command unavailable. Activate the COURSE-provided serving environment first.' >&2
  exit 2
fi

# Optional configuration file is supplied by the user, never packaged with secrets.
# It must be a trusted local file containing shell VAR=value assignments.
if [[ -n "${TACC_CONFIG_FILE:-}" ]]; then
  if [[ ! -r "$TACC_CONFIG_FILE" ]]; then
    echo "ERROR: cannot read TACC_CONFIG_FILE=$TACC_CONFIG_FILE" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "$TACC_CONFIG_FILE"
  set +a
fi

: "${TACC_MODEL_PATH:?Set TACC_MODEL_PATH to your course model location (local path or allowed model ID)}"
if [[ "$TACC_MODEL_PATH" == /REPLACE/* ]]; then
  echo "ERROR: configure a real TACC_MODEL_PATH in your private file." >&2
  exit 2
fi
: "${TACC_SERVED_MODEL_NAME:=qwen3.8-27b}"
: "${TACC_PORT:=8000}"
: "${TACC_BIND_HOST:=127.0.0.1}"
: "${TACC_GPUS:=2}"
: "${TACC_MAX_MODEL_LEN:=32768}"
: "${TACC_MAX_SEQS:=4}"
: "${TACC_GPU_MEMORY_UTILIZATION:=0.90}"
: "${TACC_READY_TIMEOUT:=420}"
: "${TACC_METRICS_INTERVAL:=5}"
: "${TACC_RUNTIME_DIR:=${SLURM_SUBMIT_DIR:-$PWD}/tacc/runtime}"
: "${TACC_NO_THINKING:=1}"

for var in TACC_PORT TACC_GPUS TACC_MAX_MODEL_LEN TACC_MAX_SEQS TACC_READY_TIMEOUT TACC_METRICS_INTERVAL; do
  value="${!var}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -eq 0 ]]; then
    echo "ERROR: $var must be a positive integer" >&2; exit 2
  fi
done
if [[ "$TACC_BIND_HOST" != "127.0.0.1" && "$TACC_BIND_HOST" != "0.0.0.0" ]]; then
  echo 'ERROR: choose TACC_BIND_HOST=127.0.0.1 or 0.0.0.0 only.' >&2; exit 2
fi
if [[ "$TACC_GPU_MEMORY_UTILIZATION" != 0.* && "$TACC_GPU_MEMORY_UTILIZATION" != 1.0 ]]; then
  echo 'ERROR: TACC_GPU_MEMORY_UTILIZATION must be a fraction such as 0.90.' >&2; exit 2
fi

visible=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$visible" -lt "$TACC_GPUS" ]]; then
  echo "ERROR: requested tensor parallelism=$TACC_GPUS but only $visible visible GPU(s)" >&2
  exit 2
fi

umask 077
mkdir -p "$TACC_RUNTIME_DIR"
job_dir="$TACC_RUNTIME_DIR/$SLURM_JOB_ID"
mkdir -p "$job_dir"
hostname_full=$(hostname -f 2>/dev/null || hostname)
state_file="$job_dir/status.json"
export TACC_SERVER_HOST="$hostname_full"
export TACC_STATE_FILE="$state_file"
export TACC_SERVER_PID='' TACC_SERVER_STATE='starting'
export TACC_JOB_DIR="$job_dir"

write_state() {
  python3 - <<'PY'
import json,os,datetime,pathlib
p=pathlib.Path(os.environ['TACC_STATE_FILE'])
state={'job_id':os.environ['SLURM_JOB_ID'],
       'state':os.environ.get('TACC_SERVER_STATE','unknown'),
       'host':os.environ['TACC_SERVER_HOST'],
       'port':int(os.environ['TACC_PORT']),
       'model':os.environ['TACC_SERVED_MODEL_NAME'],
       'tensor_parallel':int(os.environ['TACC_GPUS']),
       'pid':os.environ.get('TACC_SERVER_PID',''),
       'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
tmp.replace(p)
PY
}
write_state

vllm_args=(serve "$TACC_MODEL_PATH"
  --served-model-name "$TACC_SERVED_MODEL_NAME"
  --host "$TACC_BIND_HOST" --port "$TACC_PORT"
  --tensor-parallel-size "$TACC_GPUS"
  --max-model-len "$TACC_MAX_MODEL_LEN"
  --max-num-seqs "$TACC_MAX_SEQS"
  --gpu-memory-utilization "$TACC_GPU_MEMORY_UTILIZATION")

# Disable Qwen's thinking mode on versions that support this vLLM option.
# A strict check avoids silently sending reasoning traces into short answer budgets.
if [[ "$TACC_NO_THINKING" == '1' ]]; then
  if vllm serve --help 2>&1 | grep -F -- '--default-chat-template-kwargs' > /dev/null; then
    vllm_args+=(--default-chat-template-kwargs '{"enable_thinking":false}')
  else
    echo 'ERROR: vLLM does not support --default-chat-template-kwargs. Set TACC_NO_THINKING=0 only if you accept reasoning output and adjust prompts.' >&2
    exit 2
  fi
fi

# Require a caller-provided secret if binding to non-loopback.
# Even on a private cluster, do not run an unauthenticated HTTP service.
if [[ "$TACC_BIND_HOST" == '0.0.0.0' ]]; then
  if [[ "${TACC_ALLOW_INTERNAL_BIND:-}" != 'I_HAVE_NETWORK_APPROVAL' ]]; then
    echo 'ERROR: non-loopback binding requires TACC_ALLOW_INTERNAL_BIND=I_HAVE_NETWORK_APPROVAL.' >&2
    exit 2
  fi
  if [[ -z "${TACC_API_KEY:-}" ]]; then
    echo 'ERROR: TACC_API_KEY required when binding to 0.0.0.0.' >&2
    exit 2
  fi
fi
# Avoid exposing the key in process arguments (`ps`). vLLM reads this env var.
# Authentication applies to /v1 endpoints, not to every vLLM route; still
# rely on TACC-approved network isolation / SSH tunneling.
if [[ -n "${TACC_API_KEY:-}" ]]; then
  export VLLM_API_KEY="$TACC_API_KEY"
fi

# Logs and local GPU telemetry remain in a per-job private folder.
(nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
   --format=csv -l "$TACC_METRICS_INTERVAL" > "$job_dir/gpu.csv" 2>"$job_dir/metrics.err") &
monitor_pid=$!

vllm "${vllm_args[@]}" > "$job_dir/vllm.log" 2>&1 &
server_pid=$!
export TACC_SERVER_PID="$server_pid"

cleanup() {
  set +e
  export TACC_SERVER_STATE='stopped'
  write_state
  kill "$monitor_pid" "$server_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "vLLM started on allocated compute node: $hostname_full port=$TACC_PORT job=$SLURM_JOB_ID"
echo "Status: $state_file  Log: $job_dir/vllm.log"
echo "Network scope: $TACC_BIND_HOST (compute node only). Avoid any public forwarding."

ready=0
for ((i=0; i<TACC_READY_TIMEOUT; i+=3)); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    export TACC_SERVER_STATE='failed'
    write_state
    echo "ERROR: vLLM exited. Last log lines:" >&2
    tail -n 40 "$job_dir/vllm.log" >&2
    exit 1
  fi
  if python3 - "$TACC_PORT" "${TACC_API_KEY:-}" <<'PY'
import sys,urllib.request
port,key=sys.argv[1:]
headers={'Authorization':'Bearer '+key} if key else {}
req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/models',headers=headers)
try:
    with urllib.request.urlopen(req,timeout=2) as r:
        sys.exit(0 if r.status==200 else 1)
except Exception:
    sys.exit(1)
PY
  then
    ready=1;break
  fi
  sleep 3
done
if ((ready==0)); then
  echo "ERROR: vLLM not ready after ${TACC_READY_TIMEOUT}s" >&2
  export TACC_SERVER_STATE='failed'
  write_state
  exit 1
fi
export TACC_SERVER_STATE='ready'
write_state
echo "READY: http://$hostname_full:$TACC_PORT/v1 (usable only through an approved route)"

# Keep the allocation occupied until vLLM exits, or Slurm time limit/cancellation.
wait "$server_pid"
