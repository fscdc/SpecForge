#!/bin/bash
# Sliding-window runs for the 4B MMFlash draft on image (and video) benchmarks.
#
# Question: does restricting the draft's attention to the most recent W target
# tokens (SGLang --speculative-draft-window-size, training-free) change the
# acceptance length and decode throughput? Video prompts are ~43k tokens, almost
# all frames; image prompts are a few hundred (ChartQA p50 664, DynaMath p50
# 333), so a window only bites on images when W is below prompt + answer
# (~1-1.3k): W=2048 and up leave every image request untouched, W=512 cuts it.
#
# Each window is its own server start (the flag is a launch argument), then
# bench_mm.py over IMAGE_BENCHES and VIDEO_BENCHES, then the server is killed.
# Results land in results/ as
#   <IMAGE_RUN_NAME>_win<W>_results.jsonl
#   <VIDEO_RUN_NAME>_f16-48-48_win<W>_results.jsonl
# ("full" in WINDOWS re-measures full context as ..._winfull). bench_mm skips a
# benchmark its results file already holds, so re-running finishes a partial run.
#
# Defaults = what the paper figure still needs: image benchmarks at W=512.
# Full context needs no run: plot_window_ablation.py reads it from the main
# runs (image: results/mmflash_qwen35-4B_concurrency1_temp0_4096, 2026-09-23,
# same draft and settings; video: ..._f16-48-48_winfull / ..._f16-48-48), and
# the video windows 512/2048/8192 were measured on 2026-09-30, so
# VIDEO_BENCHES is empty by default.
#
# This script is self-contained on purpose: it no longer goes through
# scripts/benchmark_mmflash*.sh, whose target/draft/block size get edited for
# other runs.
#
#   bash scripts/sweep_draft_window.sh                               # image @ W=512, interactive 1 GPU
#   WINDOWS="512 2048" bash scripts/sweep_draft_window.sh
#   VIDEO_BENCHES="longvideobench:20 moviechat:20 mvbench:20" WINDOWS="1024" bash scripts/sweep_draft_window.sh
#   qsub -l select=1:ngpus=1 -l walltime=01:00:00 scripts/sweep_draft_window.sh
#   (qsub: options go through -v NAME=value, which splits on commas)
#
# Cost on one H100/H200: ~3 min server start per window, chartqa:200 +
# dynamath:200 at concurrency 1 ~10 min; a video window adds ~12 min
# (60 questions x ~25 s prefill + decode).
#
# Then:  python scripts/plot_window_ablation.py
#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N draft-window
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:00:00

export PYTHONUNBUFFERED=1

# --------------------------- environment -------------------------------------
CONDA_ENV="${CONDA_ENV:-specforge}"
if ! python3 -c "import torch" > /dev/null 2>&1; then
    echo "[env] torch is not importable; activating conda env '${CONDA_ENV}'"
    # shellcheck disable=SC1091
    [ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
    conda activate "${CONDA_ENV}" > /dev/null 2>&1 || true
fi
if ! python3 -c "import torch" > /dev/null 2>&1; then
    echo "[error] torch is still not importable after activating '${CONDA_ENV}'" >&2
    exit 1
fi

# The checkout: this script's own directory when run with bash, the qsub
# directory when PBS spooled the script somewhere else.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2> /dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/benchmarks/bench_mm.py" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
if [ ! -f "${PROJECT_DIR:-}/benchmarks/bench_mm.py" ]; then
    echo "[error] cannot find the SpecForge checkout; run from its root" >&2
    exit 1
fi
cd "${PROJECT_DIR}" || exit 1

# for hopper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1
export SGLANG_NUMA_BIND_V2=0
export SGLANG_FORCE_STREAM_INTERVAL=1
# a window run must not inherit a sparse pattern from the calling shell
unset SGLANG_DFLASH_DRAFT_SPARSE

# ----------------------------- settings -------------------------------------
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-mmflash-llava-ov15-1M-prompted-final}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
WINDOWS="${WINDOWS:-512}"
IMAGE_BENCHES="${IMAGE_BENCHES-chartqa:200 dynamath:200}"
VIDEO_BENCHES="${VIDEO_BENCHES-}"
IMAGE_RUN_NAME="${IMAGE_RUN_NAME:-mmflash_qwen35-4B_concurrency1_temp0_4096}"
VIDEO_RUN_NAME="${VIDEO_RUN_NAME:-video_mmflash_qwen35-4B_concurrency1_temp0_4096}"
COOLDOWN="${COOLDOWN:-20}"            # seconds for the old server to free GPU and port
# SAVE_GENERATIONS=1 also writes results/<name>_<benchmark>_generations.jsonl
# (used by scripts/analyze_draft_attention_video.sh); SUMMARY=0 skips the table/plot.
SAVE_GENERATIONS="${SAVE_GENERATIONS:-0}"
SUMMARY="${SUMMARY:-1}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"

# The frame settings behind the existing video results (_f16-48-48): the
# benchmarks read these, and the name has to match for those runs to be reused.
export VDC_NUM_FRAMES=16 LVB_NUM_FRAMES=48 MOVIECHAT_NUM_FRAMES=48
export VIDEOMME_NUM_FRAMES=48 MVBENCH_NUM_FRAMES=48 MVBENCH_FRAME_PIXELS=1280x720
FRAMES_SUFFIX="_f16-48-48"

# The ONE GPU, as an index into this job's CUDA_VISIBLE_DEVICES (PBS may give
# UUIDs; "0" is the job's first card, not physical card 0).
GPU_INDEX="${GPU_INDEX:-0}"
IFS=',' read -ra VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${VISIBLE_GPUS[${GPU_INDEX}]:-}"
if [ -z "${GPU_ID}" ]; then
    echo "[error] GPU index ${GPU_INDEX} is not among the job's GPUs (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})" >&2
    exit 1
fi
# One port per job, so two sweeps placed on one node never share a server.
JOB_ID="${PBS_JOBID:-local}"
_JOB_NUMBER="${JOB_ID%%.*}"
case "${_JOB_NUMBER}" in ''|*[!0-9]*) _JOB_NUMBER=0 ;; esac
PORT="${PORT:-$((32500 + 10#${_JOB_NUMBER} % 400))}"
LOG_DIR="logs/draft-window_${JOB_ID}"
mkdir -p "${LOG_DIR}" results

bash scripts/benchmark_helper.sh || exit 1

echo "[plan] model=${MODEL} block=${BLOCK_SIZE}"
echo "[plan] draft=${DRAFT_MODEL}"
echo "[plan] windows=${WINDOWS}"
echo "[plan] image=${IMAGE_BENCHES:-<none>} -> results/${IMAGE_RUN_NAME}_win<W>"
echo "[plan] video=${VIDEO_BENCHES:-<none>} -> results/${VIDEO_RUN_NAME}${FRAMES_SUFFIX}_win<W>"
echo "[plan] gpu=${GPU_ID} port=${PORT} server logs in ${LOG_DIR}/"

SERVER_PID=""
stop_server() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2> /dev/null; then
        kill "${SERVER_PID}" 2> /dev/null
        for _ in $(seq 1 30); do kill -0 "${SERVER_PID}" 2> /dev/null || break; sleep 1; done
        kill -9 "${SERVER_PID}" 2> /dev/null
    fi
    # the scheduler/detokenizer children outlive a killed launcher otherwise
    pkill -f "sglang.launch_server.*--port ${PORT}" 2> /dev/null
    SERVER_PID=""
}
trap stop_server EXIT

run_bench() {  # run_bench <benchmark list> <results name>
    local list="$1" name="$2"
    [ -z "${list}" ] && return 0
    echo "[bench] ${name}: ${list}"
    local extra=()
    [ "${SAVE_GENERATIONS}" = "1" ] && extra+=(--save-generations)
    # shellcheck disable=SC2086
    # bench_mm is only an HTTP client, but importing it runs sglang/test/test_utils.py,
    # which does int(CUDA_VISIBLE_DEVICES[0]) and dies on the GPU UUIDs PBS hands out
    CUDA_VISIBLE_DEVICES=0 python3 benchmarks/bench_mm.py \
        --model "${MODEL}" \
        --base-url "http://localhost:${PORT}" \
        --concurrency 1 \
        --block-size "${BLOCK_SIZE}" \
        --benchmark-list ${list} \
        --reasoning off \
        --temperature 0.0 \
        --top-p 0.95 \
        --top-k 20 \
        --max-tokens 4096 \
        "${extra[@]}" \
        --name "${name}"
}

for w in ${WINDOWS}; do
    echo "==================== draft window: ${w}  $(date '+%F %T') ===================="
    if [ "${w}" = "full" ]; then
        window_args=()
        suffix="_winfull"
    else
        case "${w}" in ''|*[!0-9]*) echo "[error] window '${w}' is neither a number nor 'full'" >&2; exit 1 ;; esac
        window_args=(--speculative-draft-window-size "${w}")
        suffix="_win${w}"
    fi
    if python3 - "${PORT}" <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen(f"http://localhost:{sys.argv[1]}/health", timeout=3)
except Exception:
    sys.exit(1)
PY
    then
        echo "[error] something already answers on port ${PORT}; set PORT=..." >&2
        exit 1
    fi

    server_log="${LOG_DIR}/server${suffix}.log"
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
        --reasoning-parser qwen3 > "${server_log}" 2>&1 &
    SERVER_PID=$!

    if ! python3 - "${PORT}" "${SERVER_TIMEOUT}" "${SERVER_PID}" <<'PY'
import os, sys, time, urllib.request
port, timeout, pid = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5).read()
        print(f"[server] ready on port {port}", flush=True)
        sys.exit(0)
    except Exception:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        print("[server] exited before becoming ready", flush=True)
        sys.exit(1)
    time.sleep(5)
print(f"[server] not ready within {timeout:.0f}s", flush=True)
sys.exit(1)
PY
    then
        echo "[error] server for window ${w} did not come up; last lines of ${server_log}:" >&2
        tail -n 30 "${server_log}" >&2
        exit 1
    fi
    if [ -n "${window_args[*]}" ] && ! grep -q "draft_window_size=${w}" "${server_log}"; then
        echo "[warn] the server log does not confirm draft_window_size=${w}" >&2
    fi

    run_bench "${IMAGE_BENCHES}" "${IMAGE_RUN_NAME}${suffix}" || { echo "[error] image pass failed for window ${w}" >&2; exit 1; }
    run_bench "${VIDEO_BENCHES}" "${VIDEO_RUN_NAME}${FRAMES_SUFFIX}${suffix}" || { echo "[error] video pass failed for window ${w}" >&2; exit 1; }

    stop_server
    sleep "${COOLDOWN}"
done

[ "${SUMMARY}" = "1" ] || exit 0
echo "==================== summary $(date '+%F %T') ===================="
python3 scripts/plot_window_ablation.py --summary-only --video-benches longvideobench mvbench moviechat
python3 scripts/plot_window_ablation.py --windows 512
