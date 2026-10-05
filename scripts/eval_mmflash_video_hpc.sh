#!/bin/bash
# Accept length of an exported MMFlash draft on the paper's video benchmarks
# (LongVideoBench, MovieChat, MVBench; 48 frames), one GPU, served with the
# draft's TRAINING-TIME context: the sparse pattern and window come from the
# export's dflash_config.draft_sparse via scripts/draft_sparse_env.py, so a
# sparse-trained draft can never be served dense or with a different pattern by
# mistake (a dense export is served with full context). Submit from the repo root:
#
#   qsub scripts/eval_mmflash_video_hpc.sh                                    # the sparse run's final export
#   qsub -v DRAFT=/path/to/export scripts/eval_mmflash_video_hpc.sh
#
# Server flags, frame settings and result naming match the earlier video runs,
# so the results file sits next to its baselines in results/:
#   video_mmflash_qwen35-4B_concurrency1_temp0_4096_f16-48-48[_win<W>_sparse-<pattern>]_<export name>
# (the same pattern on the warm-start draft is the file without _<export name>).
# bench_mm skips a benchmark its results file already holds, so a rerun finishes
# a partial one. ~5 min server start + ~25 min for 60 questions on an H200.
#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N mmflash-eval
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=02:00:00

set -uo pipefail
export PYTHONUNBUFFERED=1
echo "[eval] start $(date '+%F %T') host=$(hostname) job=${PBS_JOBID:-local} gpus=${CUDA_VISIBLE_DEVICES:-unset}"

# ------------------------------------------------------------- environment
CONDA_ENV="${CONDA_ENV:-specforge}"
if ! python3 -c "import torch" > /dev/null 2>&1; then
    # /etc/bashrc and conda's activate scripts read unset variables
    set +u
    # shellcheck disable=SC1091
    [ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
    conda activate "${CONDA_ENV}" > /dev/null 2>&1 || true
    set -u
fi
if ! python3 -c "import torch" > /dev/null 2>&1; then
    echo "[eval] ERROR: torch is not importable after activating '${CONDA_ENV}'" >&2
    exit 1
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2> /dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/benchmarks/bench_mm.py" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
cd "${PROJECT_DIR}" || exit 1

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1
export SGLANG_NUMA_BIND_V2=0
export SGLANG_FORCE_STREAM_INTERVAL=1
CACHE_TAG="${PBS_JOBID:-local-$$}"; CACHE_TAG="${CACHE_TAG%%.*}"
export TRITON_CACHE_DIR="${SPECFORGE_CACHE_ROOT:-/scratch/${USER}/tmp}/triton-eval-${CACHE_TAG}"
mkdir -p "${TRITON_CACHE_DIR}"

# ------------------------------------------------------------- settings
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
DRAFT="${DRAFT:-/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-mmflash-video-sparse-final}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
VIDEO_BENCHES="${VIDEO_BENCHES:-longvideobench:20 moviechat:20 mvbench:20}"
RUN_NAME="${RUN_NAME:-video_mmflash_qwen35-4B_concurrency1_temp0_4096}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"

# the frame settings behind the existing video results (_f16-48-48)
export VDC_NUM_FRAMES=16 LVB_NUM_FRAMES=48 MOVIECHAT_NUM_FRAMES=48
export VIDEOMME_NUM_FRAMES=48 MVBENCH_NUM_FRAMES=48 MVBENCH_FRAME_PIXELS=1280x720
FRAMES_SUFFIX="_f16-48-48"

# serving flags from the export itself
unset SGLANG_DFLASH_DRAFT_SPARSE
flags=$(python3 scripts/draft_sparse_env.py "${DRAFT}") || { echo "[eval] ERROR: cannot derive serving flags for ${DRAFT}" >&2; exit 1; }
eval "${flags}"
window_args=()
SPARSE_SUFFIX=""
if [ -n "${DRAFT_SPARSE}" ]; then
    export SGLANG_DFLASH_DRAFT_SPARSE="${DRAFT_SPARSE}"
    window_args=(--speculative-draft-window-size "${DRAFT_WINDOW}")
    SPARSE_SUFFIX="_sparse-$(printf '%s' "${DRAFT_SPARSE}" | tr -d ' ' | sed -e 's/=//g' -e 's/,/-/g')"
fi
NAME="${RUN_NAME}${FRAMES_SUFFIX}${DRAFT_WINDOW:+_win${DRAFT_WINDOW}}${SPARSE_SUFFIX}${NAME_SUFFIX}"

IFS=',' read -ra VISIBLE <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${VISIBLE[0]}"
JOBNUM="${PBS_JOBID:-0}"; JOBNUM="${JOBNUM%%.*}"
case "${JOBNUM}" in ''|*[!0-9]*) JOBNUM=0 ;; esac
PORT="${PORT:-$((33500 + 10#${JOBNUM} % 400))}"
LOG_DIR="logs/mmflash-eval_${PBS_JOBID:-local}"
mkdir -p "${LOG_DIR}" results
SERVER_LOG="${LOG_DIR}/server.log"

bash scripts/benchmark_helper.sh || exit 1
echo "[eval] draft=${DRAFT_MODEL}"
echo "[eval] SGLANG_DFLASH_DRAFT_SPARSE=${DRAFT_SPARSE:-<dense>} window=${DRAFT_WINDOW:-<none>}"
echo "[eval] benches=${VIDEO_BENCHES} -> results/${NAME}_results.jsonl"
echo "[eval] gpu=${GPU_ID} port=${PORT} server log ${SERVER_LOG}"

SERVER_PID=""
stop_server() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2> /dev/null; then
        kill "${SERVER_PID}" 2> /dev/null
        for _ in $(seq 1 30); do kill -0 "${SERVER_PID}" 2> /dev/null || break; sleep 1; done
        kill -9 "${SERVER_PID}" 2> /dev/null
    fi
    pkill -f "sglang.launch_server.*--port ${PORT}" 2> /dev/null
    SERVER_PID=""
}
trap stop_server EXIT

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -m sglang.launch_server \
    --model "${MODEL}" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "${DRAFT_MODEL}" \
    --speculative-dflash-block-size "${BLOCK_SIZE}" \
    "${window_args[@]}" \
    --mem-fraction-static 0.7 \
    --tp 1 \
    --trust-remote-code \
    --cuda-graph-max-bs 128 \
    --attention-backend fa3 \
    --mm-attention-backend sdpa \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --reasoning-parser qwen3 > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

if ! python3 - "${PORT}" "${SERVER_TIMEOUT}" "${SERVER_PID}" <<'PY'
import os, sys, time, urllib.request
port, timeout, pid = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5).read()
        print(f"[eval] server ready on port {port}", flush=True)
        sys.exit(0)
    except Exception:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        print("[eval] server exited before becoming ready", flush=True)
        sys.exit(1)
    time.sleep(5)
print(f"[eval] server not ready within {timeout:.0f}s", flush=True)
sys.exit(1)
PY
then
    tail -n 40 "${SERVER_LOG}" >&2
    exit 1
fi

# bench_mm is only an HTTP client, but importing it runs sglang/test/test_utils.py,
# which does int(CUDA_VISIBLE_DEVICES[0]) and dies on the GPU UUIDs PBS hands out
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0 python3 benchmarks/bench_mm.py \
    --model "${MODEL}" \
    --base-url "http://localhost:${PORT}" \
    --concurrency 1 \
    --block-size "${BLOCK_SIZE}" \
    --benchmark-list ${VIDEO_BENCHES} \
    --reasoning off \
    --temperature 0.0 \
    --top-p 0.95 \
    --top-k 20 \
    --max-tokens 4096 \
    --name "${NAME}"
status=$?

# the worker logs its pattern at the first prefill; without the line the
# server ran some other context and the numbers are not the trained setting
if [ -n "${DRAFT_SPARSE}" ]; then
    line=$(grep -m1 "DFLASH draft sparse context (${DRAFT_SPARSE})" "${SERVER_LOG}")
    if [ -z "${line}" ]; then
        echo "[eval] ERROR: the server log never confirms sparse context ${DRAFT_SPARSE}" >&2
        status=1
    else
        echo "[eval] confirmed: ${line##*] }"
    fi
fi
echo "[eval] $(date '+%F %T') bench_mm exited ${status}"
exit "${status}"
