#!/bin/bash
# Regenerate the video training rows (llava-video-2k) with the Qwen3.5 target,
# on ONE GPU: index 0 of the GPUs this job was given. A copy of
# scripts/mmflash_data.sh cut down to a single server and set up for video:
#
#   qsub scripts/mmflash_data-video.sh                         # submit directly
#   bash scripts/mmflash_data-video.sh                         # from pbs.sh, or on a GPU node
#   qsub -v REGEN_MODEL=Qwen/Qwen3.5-9B,REGEN_OUTPUT=... scripts/mmflash_data-video.sh
#
# qsub does NOT pass the submitting shell's environment on: override settings
# with `-v NAME=value[,NAME=value]`. With `bash` the usual `NAME=value bash ...`
# works.
#
# The input is what `scripts/prepare_data_mm.py --dataset llava-video-178k`
# writes: rows whose `image` is a list of 48 frame paths, whose user turn is
# 48 <image> placeholders and a question already worded exactly like the
# LongVideoBench / MVBench / MovieChat benchmarks. regenerate_train_data.py
# sends each row the way benchmarks/bench_mm.py sends a video question (one
# user message: 48 base64 frames, then the text, enable_thinking=False), and
# this script asks for the same sampling the benchmarks use (temperature 0,
# top-p 0.95, top-k 20, 4096 new tokens, reasoning off). --align-prompts stays
# OFF: the prompts must not be rewritten (the regen script also refuses to
# rewrite video rows).
#
# Sizing. A row is a ~42.4k-token prompt, ~42.2k of it image tokens; Qwen3.5-4B
# keeps ~32 KB of KV per token, so one unshared prompt is ~1.4 GB. The ten rows
# of a video are adjacent in the file and share their 48 frames, so the server
# prefills the frames once and the next nine rows hit the prefix cache (the
# hybrid model's cache keeps a linear-attention state every 256 tokens). 16
# in-flight requests is under two videos' worth -- well inside the KV pool --
# and enough to batch the decodes. Measured on one hopper GPU (30 rows, 3
# videos): 88% of prompt tokens served from the prefix cache, 1.53M-token KV
# pool, ~4.3 s a row, so the 2,000 rows take ~2.5 h; the walltime leaves room,
# and a job cut short is finished by re-submitting (--resume).
#
# What this does, end to end (unchanged from mmflash_data.sh):
#   1. one SGLang server on the chosen GPU, under a supervisor loop that
#      relaunches it if it dies;
#   2. regenerate_train_data.py --resume, repeated until its error file is
#      empty: --resume skips rows already in the output/skipped/rejected files
#      by id and re-queues everything in the error file;
#   3. a tally at the end that must add up to the input.
# Re-running this script on an interrupted or half-failed output is the
# intended way to finish it; nothing has to be moved or merged by hand.

#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N regen-video
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=06:00:00

export PYTHONUNBUFFERED=1

# --------------------------- environment -------------------------------------
# Under pbs.sh the env is already active; a job submitted straight with qsub
# starts from a bare shell, so activate it here when torch is not importable.
CONDA_ENV="${CONDA_ENV:-specforge}"
if ! python3 -c "import torch" > /dev/null 2>&1; then
    echo "[env] torch is not importable; activating conda env '${CONDA_ENV}'"
    if [ -f "${HOME}/.bashrc" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/.bashrc"
    fi
    conda activate "${CONDA_ENV}" > /dev/null 2>&1 || true
fi
if ! python3 -c "import torch" > /dev/null 2>&1; then
    echo "[error] torch is still not importable after activating '${CONDA_ENV}'" >&2
    exit 1
fi

# The checkout: this script's own directory when run with bash, the qsub
# directory when PBS spooled the script somewhere else.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2> /dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/scripts/regenerate_train_data.py" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
if [ ! -f "${PROJECT_DIR:-}/scripts/regenerate_train_data.py" ]; then
    echo "[error] cannot find the SpecForge checkout; submit from its root" >&2
    exit 1
fi
cd "${PROJECT_DIR}" || exit 1

# for deep100 (run after `conda activate specforge`, the path is taken from the env)
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
# export FLASHINFER_USE_CUDA_NORM=1
# export NVCC_PREPEND_FLAGS="-ccbin g++-11"

# for hopper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1
export SGLANG_NUMA_BIND_V2=0

# ----------------------------- settings -------------------------------------
# All overridable from the environment (see the top of this file).
# The MMFlash draft being trained serves Qwen3.5-4B, so its target answers.
MODEL="${REGEN_MODEL:-Qwen/Qwen3.5-4B}"
# deep100: /local_home2/fengsicheng/specforge ; hpc: /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge
DATA_ROOT="${REGEN_DATA_ROOT:-/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge}"
INPUT_FILE="${REGEN_INPUT:-${DATA_ROOT}/data/llava-video-2k_train.jsonl}"
OUTPUT_FILE="${REGEN_OUTPUT:-${DATA_ROOT}/regen_data/qwen35-4B_llava-video-2k_regen.jsonl}"
# The ONE GPU to use, as an index into the GPUs this job can see: PBS hands a
# job its cards in CUDA_VISIBLE_DEVICES, possibly by UUID, and "0" means the
# first of those -- not physical card 0, which may belong to another job.
GPU_INDEX="${REGEN_GPU:-0}"
CONCURRENCY="${REGEN_CONCURRENCY:-16}"   # in-flight requests; see "Sizing" above
MAX_TOKENS="${REGEN_MAX_TOKENS:-4096}"   # the benchmarks' --max-tokens
# --resume rounds: round 1 does the bulk, later rounds only re-queue the
# rows that failed (a server restart window, a timeout). 4 is plenty.
MAX_ROUNDS="${REGEN_MAX_ROUNDS:-4}"
SERVER_TIMEOUT="${REGEN_SERVER_TIMEOUT:-900}"  # seconds to wait for the server to come up
JOB_ID="${PBS_JOBID:-${SLURM_JOB_ID:-local}}"
# One port per job (40100-40499, off the image regen's 40000), so two jobs
# placed on the same node never share a server: the pre-launch check below
# cannot see a server that is still loading its weights.
_JOB_NUMBER="${JOB_ID%%.*}"
case "${_JOB_NUMBER}" in
    ''|*[!0-9]*) _JOB_NUMBER=0 ;;
esac
PORT="${REGEN_PORT:-$((40100 + 10#${_JOB_NUMBER} % 400))}"
LOG_DIR="logs/regen-video_${JOB_ID}"
mkdir -p "${LOG_DIR}"

# Compile caches, one directory per job and GPU (see mmflash_data.sh for the
# race a shared TRITON_CACHE_DIR caused).
# deep100: /local_home2/fengsicheng/tmp ; hpc: /scratch/${USER}/tmp
CACHE_ROOT="${SPECFORGE_CACHE_ROOT:-/scratch/${USER}/tmp}"

ERROR_FILE="${OUTPUT_FILE%.jsonl}_error.jsonl"
SKIPPED_FILE="${OUTPUT_FILE%.jsonl}_skipped.jsonl"
REJECTED_FILE="${OUTPUT_FILE%.jsonl}_rejected.jsonl"
STOP_FILE="${LOG_DIR}/stop-servers"
rm -f "${STOP_FILE}"

IFS=',' read -ra VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${VISIBLE_GPUS[${GPU_INDEX}]:-}"
if [ -z "${GPU_ID}" ]; then
    echo "[error] GPU index ${GPU_INDEX} is not among the ${#VISIBLE_GPUS[@]} GPU(s) visible to this job (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})" >&2
    exit 1
fi
if [ ! -f "${INPUT_FILE}" ]; then
    echo "[error] input ${INPUT_FILE} not found; build it with scripts/prepare_data_mm.py --dataset llava-video-178k" >&2
    exit 1
fi
mkdir -p "$(dirname "${OUTPUT_FILE}")"

SERVER_ADDRESSES=("localhost:${PORT}")
SUPERVISOR_PIDS=()

count_lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }

cleanup() {
    echo "[cleanup] stopping sglang servers..."
    touch "${STOP_FILE}"
    for pid_file in "${LOG_DIR}"/server_gpu*.pid; do
        [ -f "${pid_file}" ] || continue
        pid=$(cat "${pid_file}")
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${SUPERVISOR_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    echo "[cleanup] all sglang servers stopped"
}

trap cleanup EXIT INT TERM

# Launch one server and relaunch it whenever it exits before we asked it to.
# Rows that fail while the server is down are retried by the next --resume round.
supervise_server() {
    local gpu_id=$1 port=$2
    local attempt=0 pid status
    local cache_dir="${CACHE_ROOT}/triton-${JOB_ID}-gpu${gpu_id}"
    local log="${LOG_DIR}/server_gpu${gpu_id}_port${port}.log"
    local pid_file="${LOG_DIR}/server_gpu${gpu_id}.pid"
    mkdir -p "${cache_dir}" "${cache_dir}-inductor" || {
        echo "cannot create compile cache under ${CACHE_ROOT}; set SPECFORGE_CACHE_ROOT" >&2
        return 1
    }
    while [ ! -f "${STOP_FILE}" ]; do
        attempt=$((attempt + 1))
        echo "[server] GPU ${gpu_id} port ${port}: launch #${attempt} ($(date '+%F %T'))"
        TRITON_CACHE_DIR="${cache_dir}" \
        TORCHINDUCTOR_CACHE_DIR="${cache_dir}-inductor" \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        python3 -m sglang.launch_server \
            --model "${MODEL}" \
            --mem-fraction-static 0.7 \
            --tp 1 \
            --trust-remote-code \
            --cuda-graph-max-bs 128 \
            --attention-backend fa3 \
            --mm-attention-backend sdpa \
            --host 0.0.0.0 \
            --port "${port}" \
            --dtype bfloat16 \
            --reasoning-parser qwen3 \
            >> "${log}" 2>&1 &
        pid=$!
        echo "${pid}" > "${pid_file}"
        wait "${pid}"
        status=$?
        rm -f "${pid_file}"
        if [ -f "${STOP_FILE}" ]; then
            break
        fi
        echo "[server] GPU ${gpu_id} port ${port}: exited with status ${status} ($(date '+%F %T')); relaunching in 15s (see ${log})" >&2
        sleep 15
    done
}

server_healthy() {
    if command -v curl > /dev/null 2>&1; then
        curl -sf -m 5 "http://$1/health" > /dev/null 2>&1
    else
        ADDR="$1" python3 - <<'PY' > /dev/null 2>&1
import os, sys, urllib.request
try:
    with urllib.request.urlopen(f"http://{os.environ['ADDR']}/health", timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
    fi
}

# Block until every server answers /health (they may be restarting).
wait_for_servers() {
    local addr start_ts now
    for addr in "${SERVER_ADDRESSES[@]}"; do
        start_ts=$(date +%s)
        until server_healthy "${addr}"; do
            now=$(date +%s)
            if (( now - start_ts >= SERVER_TIMEOUT )); then
                echo "[error] timed out waiting for server ${addr}; check ${LOG_DIR}" >&2
                return 1
            fi
            sleep 10
        done
        echo "[ready] server ${addr} is up"
    done
}

# Same sampling as the video benchmarks; --align-prompts deliberately absent.
run_regen_round() {
    python3 scripts/regenerate_train_data.py \
        --model "${MODEL}" \
        --concurrency "${CONCURRENCY}" \
        --max-tokens "${MAX_TOKENS}" \
        --server-address "${SERVER_ADDRESSES[@]}" \
        --temperature 0.0 \
        --top-p 0.95 \
        --top-k 20 \
        --input-file-path "${INPUT_FILE}" \
        --output-file-path "${OUTPUT_FILE}" \
        --resume \
        --reasoning disable
}

echo "[info] job ${JOB_ID}; log directory: ${LOG_DIR}"
echo "[info] model ${MODEL}; GPU index ${GPU_INDEX} -> CUDA device ${GPU_ID}; port ${PORT}; ${CONCURRENCY} in-flight requests"
echo "[info] input  ${INPUT_FILE}"
echo "[info] output ${OUTPUT_FILE}"
echo "[info] already on disk: $(count_lines "${OUTPUT_FILE}") regenerated, $(count_lines "${ERROR_FILE}") failed (will be retried), $(count_lines "${SKIPPED_FILE}") skipped, $(count_lines "${REJECTED_FILE}") rejected"

# A server already answering on the port belongs to someone else (another job
# on this node); the regen would silently send every row to it.
if server_healthy "${SERVER_ADDRESSES[0]}"; then
    echo "[error] something already answers on ${SERVER_ADDRESSES[0]}; pick another REGEN_PORT" >&2
    exit 1
fi

echo "[info] starting the SGLang server..."
supervise_server "${GPU_ID}" "${PORT}" &
SUPERVISOR_PIDS+=("$!")

echo "[wait] waiting for the server to become ready..."
wait_for_servers || exit 1

TOTAL_INPUT=$(count_lines "${INPUT_FILE}")
prev_failed=-1
for round in $(seq 1 "${MAX_ROUNDS}"); do
    echo "[round ${round}/${MAX_ROUNDS}] $(date '+%F %T') regenerating; log: ${LOG_DIR}/regen.log"
    # the server may be mid-restart; the client drops any server that is down
    # when it starts, so make sure it is back first
    wait_for_servers || exit 1
    echo "==================== round ${round} $(date '+%F %T') ====================" >> "${LOG_DIR}/regen.log"
    if ! run_regen_round >> "${LOG_DIR}/regen.log" 2>&1; then
        status=$?
        echo "[error] regeneration exited with ${status} in round ${round}; check ${LOG_DIR}/regen.log" >&2
        exit "${status}"
    fi
    failed=$(count_lines "${ERROR_FILE}")
    echo "[round ${round}] done: $(count_lines "${OUTPUT_FILE}") regenerated, ${failed} failed"
    if [ "${failed}" -eq 0 ]; then
        break
    fi
    if [ "${prev_failed}" -ge 0 ] && [ "${failed}" -ge "${prev_failed}" ]; then
        echo "[error] round ${round} made no progress (${failed} failures, previously ${prev_failed}); the server is probably unhealthy, check ${LOG_DIR}" >&2
        exit 1
    fi
    prev_failed=${failed}
    echo "[round ${round}] ${failed} rows failed (Connection error / timeout); retrying them"
done

success=$(count_lines "${OUTPUT_FILE}")
failed=$(count_lines "${ERROR_FILE}")
skipped=$(count_lines "${SKIPPED_FILE}")
rejected=$(count_lines "${REJECTED_FILE}")
echo "[done] $(date '+%F %T') input ${TOTAL_INPUT} = regenerated ${success} + failed ${failed} + skipped ${skipped} + rejected ${rejected} (sum $((success + failed + skipped + rejected)))"
if [ "${failed}" -ne 0 ]; then
    echo "[done] ${failed} rows still failed after ${MAX_ROUNDS} rounds; re-run this script to retry them" >&2
    exit 1
fi
