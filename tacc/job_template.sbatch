#!/bin/bash -l
#SBATCH --job-name=paper-tutor-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=tacc-model-%j.out
#SBATCH --error=tacc-model-%j.err
# IMPORTANT: No partition, account or GPU directive is guessed here.
# Request 2 GPUs and the APPROVED partition/account at sbatch submission time,
# or replace this template with your course-supplied GPU job script.
set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:?}"
if [[ -z "${TACC_CONFIG_FILE:-}" ]]; then
  echo 'TACC_CONFIG_FILE must point to a private configuration file.' >&2
  exit 2
fi

# Activate the vLLM environment here, IF the course requires it, e.g.:
# module load ...          # use the module named by your course
# source .../bin/activate  # use the actual course vLLM environment
# No environment name or model path is invented.

bash tacc/serve_in_allocation.sh
