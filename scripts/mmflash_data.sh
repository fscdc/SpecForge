#!/bin/bash


# dataset: sharegpt4v / sharegpt4v-pt / llava-onevision-1.5
# hpc: /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/data/
# deep100: /local_home1/fengsicheng/specforge/data/

# export HF_HOME=/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/.cache/huggingface

export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_NUM_CONCURRENT_RANGE_GETS=32
export HF_HUB_DOWNLOAD_TIMEOUT=120

# python scripts/prepare_data.py \
#     --dataset perfectblend \
#     --output-name perfectblend \
#     --output-path /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/data/ \
#     --overwrite

# python scripts/prepare_data_mm.py \
#     --dataset sharegpt4v \
#     --image-root /local_home1/fengsicheng/specforge/data \
#     --output-path /local_home1/fengsicheng/specforge/data/ \

# python scripts/prepare_data_mm.py \
#     --dataset llava-onevision-1.5 \
#     --sample-size 1000000 \
#     --fetch shards \
#     --image-root /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/data \
#     --output-path /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/data \
#     --output-name llava-ov15-1M



# TODO@song: use the xx_manifest.json to construct same dataset on other server
# deep100: /local_home2/fengsicheng/specforge/data
# 这里先把数据造出来之后，再去跑后面的regen
# python scripts/prepare_data_mm.py \
#     --dataset llava-onevision-1.5 \
#     --sample-size 1000000 \
#     --fetch shards \
#     --manifest ./scripts/data_reproduce/llava-ov15-1M_manifest.json \
#     --image-root /local_home2/fengsicheng/specforge/data \
#     --output-path /local_home2/fengsicheng/specforge/data \
#     --output-name llava-ov15-1M


####################################   up build data up   ##############################
#################################### down regen data down ##########################

# What this does, end to end:
#   1. one SGLang server per GPU, each under a supervisor loop that relaunches
#      it if it dies (a dead server used to silently fail every other row);
#   2. regenerate_train_data.py --resume, repeated until its error file is
#      empty: --resume skips rows already in the output/skipped/rejected files
#      by id and re-queues everything in the error file, so a round only costs
#      the rows that actually failed;
#   3. a tally at the end that must add up to the input.
# Re-running this script on an interrupted or half-failed output is the
# intended way to finish it; nothing has to be moved or merged by hand.

# for deep100 (run after `conda activate specforge`, the path is taken from the env)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1
export NVCC_PREPEND_FLAGS="-ccbin g++-11"

# for hopper
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
# export FLASHINFER_USE_CUDA_NORM=1

export PYTHONUNBUFFERED=1

# ----------------------------- settings -------------------------------------
# All overridable from the environment, e.g.
#   REGEN_GPUS="0 1 2 3" bash scripts/mmflash_data.sh
MODEL="${REGEN_MODEL:-Qwen/Qwen3.5-9B}"
# deep100: /local_home2/fengsicheng/specforge ; hpc: /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge
DATA_ROOT="${REGEN_DATA_ROOT:-/local_home2/fengsicheng/specforge}"
INPUT_FILE="${REGEN_INPUT:-${DATA_ROOT}/data/llava-ov15-1M_train.jsonl}"
# OUTPUT_FILE="${REGEN_OUTPUT:-${DATA_ROOT}/regen_data/qwen35-9B_llava-ov15-1M-prompted_regen_first_turn.jsonl}"
OUTPUT_FILE="${REGEN_OUTPUT:-${DATA_ROOT}/regen_data/qwen35-9B_llava-ov15-1M_regen_first_turn.jsonl}"
# one server per entry; must match the job's GPU allocation (pbs.sh ngpus=)
GPU_IDS=(${REGEN_GPUS:-0 1})
CONCURRENCY="${REGEN_CONCURRENCY:-64}"   # in-flight requests per server
MAX_TOKENS="${REGEN_MAX_TOKENS:-4096}"
# --resume rounds: round 1 does the bulk, later rounds only re-queue the
# rows that failed (a server restart window, a timeout). 4 is plenty.
MAX_ROUNDS="${REGEN_MAX_ROUNDS:-4}"
SERVER_TIMEOUT="${REGEN_SERVER_TIMEOUT:-900}"  # seconds to wait for a server to come up
PORT_BASE=40000

JOB_ID="${PBS_JOBID:-${SLURM_JOB_ID:-local}}"
LOG_DIR="logs/regen_${JOB_ID}"
mkdir -p "${LOG_DIR}"

# Compile caches. ~/.bashrc points TRITON_CACHE_DIR at ONE shared directory on
# /scratch. Two servers that start together JIT-compile the same kernels at the
# same moment and race on it: one of them reads a .ptx/.json the other is in
# the middle of replacing, gets FileNotFoundError, and its scheduler exits.
# That is exactly what killed one server in each of the two 9B runs. One
# directory per job and per GPU, like scripts/training.sh does per host.
# deep100: /local_home2/fengsicheng/tmp ; hpc: /scratch/${USER}/tmp
CACHE_ROOT="${SPECFORGE_CACHE_ROOT:-/local_home2/fengsicheng/tmp}"

ERROR_FILE="${OUTPUT_FILE%.jsonl}_error.jsonl"
SKIPPED_FILE="${OUTPUT_FILE%.jsonl}_skipped.jsonl"
REJECTED_FILE="${OUTPUT_FILE%.jsonl}_rejected.jsonl"
STOP_FILE="${LOG_DIR}/stop-servers"
rm -f "${STOP_FILE}"

SERVER_ADDRESSES=()
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
# Rows that fail while a server is down are retried by the next --resume round.
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
        # --align-prompts

echo "[info] job ${JOB_ID}; log directory: ${LOG_DIR}"
echo "[info] model ${MODEL}; GPUs ${GPU_IDS[*]}; ${CONCURRENCY} in-flight requests per server"
echo "[info] input  ${INPUT_FILE}"
echo "[info] output ${OUTPUT_FILE}"
echo "[info] already on disk: $(count_lines "${OUTPUT_FILE}") regenerated, $(count_lines "${ERROR_FILE}") failed (will be retried), $(count_lines "${SKIPPED_FILE}") skipped, $(count_lines "${REJECTED_FILE}") rejected"

echo "[info] starting SGLang servers..."
for idx in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$idx]}"
    port=$((PORT_BASE + idx * 10))
    SERVER_ADDRESSES+=("localhost:${port}")
    supervise_server "${gpu_id}" "${port}" &
    SUPERVISOR_PIDS+=("$!")
done

echo "[wait] waiting for all servers to become ready..."
wait_for_servers || exit 1

TOTAL_INPUT=$(count_lines "${INPUT_FILE}")
prev_failed=-1
for round in $(seq 1 "${MAX_ROUNDS}"); do
    echo "[round ${round}/${MAX_ROUNDS}] $(date '+%F %T') regenerating; log: ${LOG_DIR}/regen.log"
    # a server may be mid-restart; the client drops any server that is down
    # when it starts, so make sure they are all back first
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
        echo "[error] round ${round} made no progress (${failed} failures, previously ${prev_failed}); the servers are probably unhealthy, check ${LOG_DIR}" >&2
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
