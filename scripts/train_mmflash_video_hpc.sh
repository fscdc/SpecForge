#!/bin/bash
# Continued MMFlash training on the LLaVA-Video rows, one 2-GPU PBS job per run.
# Submit from the repo root:
#
#   qsub scripts/train_mmflash_video_hpc.sh                                      # sparse run
#   qsub -v CONFIG=scripts/mmtraining_configs/qwen3.5-4b-mmflash-video-dense_hpc.yaml scripts/train_mmflash_video_hpc.sh
#
# - The job's first GPU serves capture, the second trains (passed by UUID, as
#   PBS hands them out).
# - Capture and Mooncake ports are derived from the job number, so two jobs (or
#   other users' jobs) on one node do not collide.
# - managed_local needs a fresh control_dir: an earlier attempt's control/ and
#   consumer-state/ are moved to <output_dir>/attempt-<jobid>/ (checkpoints stay).
# - SGLANG_MM_SKIP_COMPUTE_HASH=1 (default) skips the capture server's per-image
#   sha256 (~1 s per 48-frame row); outputs are identical. -v SGLANG_MM_SKIP_COMPUTE_HASH=0
#   turns it off.
# The log is <output_dir>/logs/train_<jobid>.log.
#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N mmflash-video
#PBS -q auto
#PBS -l select=1:ngpus=2
#PBS -l walltime=36:00:00

set -uo pipefail
export PYTHONUNBUFFERED=1
echo "[train] start $(date '+%F %T') host=$(hostname) job=${PBS_JOBID:-local} gpus=${CUDA_VISIBLE_DEVICES:-unset}"

# ------------------------------------------------------------- environment
CONDA_ENV="${CONDA_ENV:-specforge}"
if ! python3 -c "import torch" > /dev/null 2>&1; then
    # /etc/bashrc and conda's activate scripts read unset variables, which
    # set -u turns into an immediate exit
    set +u
    # shellcheck disable=SC1091
    [ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
    conda activate "${CONDA_ENV}" > /dev/null 2>&1 || true
    set -u
fi
if ! python3 -c "import torch" > /dev/null 2>&1; then
    echo "[train] ERROR: torch is not importable after activating '${CONDA_ENV}'" >&2
    exit 1
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2> /dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/specforge/cli.py" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
cd "${PROJECT_DIR}" || exit 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1
export SGLANG_NUMA_BIND_V2=0
export SGLANG_MM_SKIP_COMPUTE_HASH="${SGLANG_MM_SKIP_COMPUTE_HASH:-1}"
# Compile caches per JOB: a per-host inductor cache can hold entries whose
# Triton cubins lived in another job's TRITON_CACHE_DIR, and loading one fails
# with "CUDA driver error: file not found" at the first flex_attention step.
CACHE_ROOT="${SPECFORGE_CACHE_ROOT:-/scratch/${USER}/tmp}"
CACHE_TAG="${PBS_JOBID:-local-$$}"; CACHE_TAG="${CACHE_TAG%%.*}"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton-train-${CACHE_TAG}"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor-train-${CACHE_TAG}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# ------------------------------------------------------------- settings
CONFIG="${CONFIG:-scripts/mmtraining_configs/qwen3.5-4b-mmflash-video-sparse_hpc.yaml}"
[ -f "${CONFIG}" ] || { echo "[train] ERROR: config ${CONFIG} not found under ${PROJECT_DIR}" >&2; exit 1; }

IFS=',' read -ra VISIBLE <<< "${CUDA_VISIBLE_DEVICES:-}"
if [ "${#VISIBLE[@]}" -lt 2 ]; then
    echo "[train] ERROR: needs 2 GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
    exit 1
fi
CAPTURE_GPU="${VISIBLE[0]}"
TRAINER_GPU="${VISIBLE[1]}"

# output_dir, control_dir, consumer_state_dir and the capture server's memory
# fraction, read from the config itself
read -r OUT CONTROL STATE CAPTURE_MEM < <(python3 - "${CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
dis = cfg["deployment"]["disaggregated"]
server = dis["managed_local"]["capture_servers"][0]
print(cfg["output_dir"], dis["control_dir"], dis["consumer_state_dir"], server.get("mem_fraction_static", 0.7))
PY
)
JOBNUM="${PBS_JOBID:-0}"; JOBNUM="${JOBNUM%%.*}"
case "${JOBNUM}" in ''|*[!0-9]*) JOBNUM=0 ;; esac
OFFSET=$(( (10#${JOBNUM} % 400) * 10 ))
CAPTURE_PORT=$(( 31000 + OFFSET ))
MC_RPC=$(( 45000 + OFFSET ))
MC_METADATA=$(( 45001 + OFFSET ))
MC_METRICS=$(( 45002 + OFFSET ))

mkdir -p "${OUT}/logs"
if [ -e "${CONTROL}" ] || [ -e "${STATE}" ]; then
    ASIDE="${OUT}/attempt-${JOBNUM}-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "${ASIDE}"
    [ -e "${CONTROL}" ] && mv "${CONTROL}" "${ASIDE}/"
    [ -e "${STATE}" ] && mv "${STATE}" "${ASIDE}/"
    echo "[train] moved an earlier attempt's control/consumer state to ${ASIDE}"
fi
LOG="${OUT}/logs/train_${JOBNUM}.log"

echo "[train] config=${CONFIG}"
echo "[train] capture GPU ${CAPTURE_GPU} (port ${CAPTURE_PORT}, mem ${CAPTURE_MEM}), trainer GPU ${TRAINER_GPU}"
echo "[train] mooncake ports rpc ${MC_RPC} metadata ${MC_METADATA} metrics ${MC_METRICS}"
echo "[train] SGLANG_MM_SKIP_COMPUTE_HASH=${SGLANG_MM_SKIP_COMPUTE_HASH}; log ${LOG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python3 -m specforge.cli train --config "${CONFIG}" \
    "deployment.disaggregated.managed_local.trainer_cuda_visible_devices=[${TRAINER_GPU}]" \
    "deployment.disaggregated.managed_local.capture_servers=[{port: ${CAPTURE_PORT}, cuda_visible_devices: [${CAPTURE_GPU}], tp_size: 1, mem_fraction_static: ${CAPTURE_MEM}}]" \
    "deployment.disaggregated.managed_local.mooncake.rpc_port=${MC_RPC}" \
    "deployment.disaggregated.managed_local.mooncake.metadata_port=${MC_METADATA}" \
    "deployment.disaggregated.managed_local.mooncake.metrics_port=${MC_METRICS}" \
    > "${LOG}" 2>&1
status=$?
echo "[train] $(date '+%F %T') training exited ${status}"
tail -n 40 "${LOG}"
exit "${status}"
