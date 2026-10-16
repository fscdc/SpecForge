#!/bin/bash
# Where the time of a DFlash decode step goes -- draft vs target verify -- for
# several drafts on the video benchmarks, one GPU. Each configuration gets its
# own server with scripts/profiling/dflash_step_timing/sitecustomize.py on its
# PYTHONPATH (CUDA events around the draft forward, the draft's token choice,
# the sparse gather, the target verify and the draft-KV write; nothing in the
# installed SGLang is edited), then answers N questions of each benchmark, one
# bench_mm call per benchmark so the timings are labelled per benchmark.
# Submit from the repo root:
#
#   qsub scripts/profile_dflash_steps_hpc.sh
#   qsub -v N=10,BENCHES="longvideobench mvbench" scripts/profile_dflash_steps_hpc.sh
#
# CONFIGS: "name|draft dir|sparse override" entries separated by ';' (an empty
# override serves the draft as exported: its own trained pattern, or dense).
# The questions are the first N of the 200-question runs, read from the caches
# scripts/prefetch_video_benches.py filled (downloads off, every LongVideoBench
# video as 48 frames). Then:
#
#   python scripts/summarize_dflash_step_timing.py <OUT_DIR>
#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N dflash-timing
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=04:00:00

set -uo pipefail
export PYTHONUNBUFFERED=1
echo "[timing] start $(date '+%F %T') host=$(hostname) job=${PBS_JOBID:-local} gpus=${CUDA_VISIBLE_DEVICES:-unset}"

CONDA_ENV="${CONDA_ENV:-specforge}"
if ! python3 -c "import torch" > /dev/null 2>&1; then
    set +u
    # shellcheck disable=SC1091
    [ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
    conda activate "${CONDA_ENV}" > /dev/null 2>&1 || true
    set -u
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2> /dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/benchmarks/bench_mm.py" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
cd "${PROJECT_DIR}" || exit 1

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1 SGLANG_NUMA_BIND_V2=0 SGLANG_FORCE_STREAM_INTERVAL=1
JOBNUM="${PBS_JOBID:-0}"; JOBNUM="${JOBNUM%%.*}"
case "${JOBNUM}" in ''|*[!0-9]*) JOBNUM=0 ;; esac
export TRITON_CACHE_DIR="${SPECFORGE_CACHE_ROOT:-/scratch/${USER}/tmp}/triton-timing-${JOBNUM}"
mkdir -p "${TRITON_CACHE_DIR}"
# the 200-question runs' frame settings; read the prefetched caches only
export VDC_NUM_FRAMES=16 LVB_NUM_FRAMES=48 MOVIECHAT_NUM_FRAMES=48 VIDEOMME_NUM_FRAMES=48 MVBENCH_NUM_FRAMES=48
export MVBENCH_FRAME_PIXELS=1280x720 LVB_FPS_CAP=0
export LVB_AUTO_DOWNLOAD=0 VIDEOMME_AUTO_DOWNLOAD=0 MVBENCH_AUTO_DOWNLOAD=0

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
MODEL_TAG="$(basename "${MODEL}" | sed 's/^Qwen3\.5-/qwen35-/')"
DM=/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models
ZLAB4B=/scratch/fengsicheng/.cache/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots/9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf
CONFIGS="${CONFIGS:-mmflash|${DM}/qwen3.5-4b-mmflash-llava-ov15-1M-prompted-final|;mmflash-sparse-free|${DM}/qwen3.5-4b-mmflash-llava-ov15-1M-prompted-final|sink=4,text=1,stride=32,window=2048;mmflash-sparse-trained|${DM}/qwen3.5-4b-mmflash-video-sparse-final|;zlab|${ZLAB4B}|}"
BENCHES="${BENCHES:-longvideobench mvbench videomme}"
N="${N:-20}"
BLOCK_SIZE=16
OUT_DIR="${OUT_DIR:-results/dflash_step_timing/${MODEL_TAG}_${JOBNUM}}"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
IFS=',' read -ra VISIBLE <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${VISIBLE[0]}"
PORT="${PORT:-$((34500 + 10#${JOBNUM} % 400))}"
bash scripts/benchmark_helper.sh || exit 1
echo "[timing] model=${MODEL} benches=${BENCHES} n=${N} out=${OUT_DIR} port=${PORT}"

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

status=0
IFS=';' read -ra ENTRIES <<< "${CONFIGS}"
for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r name draft override <<< "${entry}"
    [ -z "${name}" ] && continue
    echo "==================== ${name}  $(date '+%F %T') ===================="
    unset SGLANG_DFLASH_DRAFT_SPARSE
    flags=$(python3 scripts/draft_sparse_env.py "${draft}") || { echo "[timing] ERROR: cannot read ${draft}" >&2; status=1; continue; }
    eval "${flags}"
    if [ -n "${override}" ]; then
        [ -n "${DRAFT_SPARSE}" ] && { echo "[timing] ERROR: ${draft} has its own pattern; no override" >&2; status=1; continue; }
        DRAFT_SPARSE="${override}"
        DRAFT_WINDOW=$(printf '%s' "${override}" | tr ',' '\n' | sed -n 's/^window=//p')
    fi
    window_args=()
    if [ -n "${DRAFT_SPARSE}" ]; then
        export SGLANG_DFLASH_DRAFT_SPARSE="${DRAFT_SPARSE}"
        window_args=(--speculative-draft-window-size "${DRAFT_WINDOW}")
    fi
    cfg_dir="${OUT_DIR}/${name}"
    mkdir -p "${cfg_dir}"
    echo "warmup" > "${cfg_dir}/label"
    results_name="video_timing_${name}_${MODEL_TAG}_concurrency1_temp0_4096_f16-48-48_n${N}"
    printf 'draft=%s\nsparse=%s\nwindow=%s\nresults=%s\n' "${DRAFT_MODEL}" "${DRAFT_SPARSE:-<dense>}" "${DRAFT_WINDOW:-<none>}" "${results_name}" > "${cfg_dir}/config.txt"
    PYTHONPATH="${PROJECT_DIR}/scripts/profiling/dflash_step_timing${PYTHONPATH:+:${PYTHONPATH}}" \
    DFLASH_TIMING_DIR="${cfg_dir}/timing" DFLASH_TIMING_LABEL_FILE="${cfg_dir}/label" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -m sglang.launch_server \
        --model "${MODEL}" \
        --speculative-algorithm DFLASH \
        --speculative-draft-model-path "${DRAFT_MODEL}" \
        --speculative-dflash-block-size "${BLOCK_SIZE}" \
        "${window_args[@]}" \
        --mem-fraction-static 0.7 --tp 1 --trust-remote-code --cuda-graph-max-bs 128 \
        --attention-backend fa3 --mm-attention-backend sdpa \
        --host 0.0.0.0 --port "${PORT}" --dtype bfloat16 --reasoning-parser qwen3 \
        > "${cfg_dir}/server.log" 2>&1 &
    SERVER_PID=$!
    if ! python3 - "${PORT}" 1800 "${SERVER_PID}" <<'PY'
import os, sys, time, urllib.request
port, timeout, pid = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5).read(); sys.exit(0)
    except Exception:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        sys.exit(1)
    time.sleep(5)
sys.exit(1)
PY
    then
        echo "[timing] ERROR: server for ${name} did not come up" >&2; tail -n 30 "${cfg_dir}/server.log" >&2
        stop_server; status=1; continue
    fi
    for bench in ${BENCHES}; do
        echo "${bench}" > "${cfg_dir}/label"
        echo "[timing] ${name} / ${bench} $(date '+%T')"
        CUDA_VISIBLE_DEVICES=0 python3 benchmarks/bench_mm.py \
            --model "${MODEL}" --base-url "http://localhost:${PORT}" --concurrency 1 \
            --block-size "${BLOCK_SIZE}" --benchmark-list "${bench}:${N}" --reasoning off \
            --temperature 0.0 --top-p 0.95 --top-k 20 --max-tokens 4096 \
            --name "${results_name}" \
            > "${cfg_dir}/bench_${bench}.log" 2>&1 || { echo "[timing] bench_mm failed for ${name}/${bench}" >&2; status=1; }
    done
    echo "done" > "${cfg_dir}/label"
    sleep 15   # let the flusher thread write the last totals
    stop_server
    ls "${cfg_dir}/timing" 2> /dev/null | sed "s/^/[timing] ${name} timing file: /"
    sleep 15
done

python3 scripts/summarize_dflash_step_timing.py "${OUT_DIR}" | tee "${OUT_DIR}/summary.txt"
echo "[timing] $(date '+%F %T') exit ${status}"
exit "${status}"
