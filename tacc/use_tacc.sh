#!/usr/bin/env bash
# Source this in the SAME shell that starts backend/benchmark, inside TACC.
# Usage: source tacc/use_tacc.sh HOSTNAME 8000
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'ERROR: source this script rather than executing it.' >&2
  exit 2
fi
if [[ $# -lt 1 ]]; then
  echo 'Usage: source tacc/use_tacc.sh COMPUTE_HOST [PORT]' >&2
  return 2
fi
export MODEL_BASE_URL="http://$1:${2:-8000}/v1"
export MODEL_PROVIDER=openai
export MODEL_NAME="${TACC_SERVED_MODEL_NAME:-qwen3.8-27b}"
export MODEL_API_KEY="${TACC_API_KEY:-local}"
export MODEL_REASONING_EFFORT=''
export MODEL_TIMEOUT=240
printf 'Configured backend model: %s via %s\n' "$MODEL_NAME" "$MODEL_BASE_URL"
