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


# for deep100
export LD_LIBRARY_PATH="/home/svu/fengsicheng/miniconda3/envs/specforge/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
export FLASHINFER_USE_CUDA_NORM=1
export NVCC_PREPEND_FLAGS="-ccbin g++-11"

# for hopper
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
# export FLASHINFER_USE_CUDA_NORM=1

export PYTHONUNBUFFERED=1
GPU_IDS=(0 1 2 3)

JOB_ID="${PBS_JOBID:-${SLURM_JOB_ID:-local}}"
LOG_DIR="logs/regen_${JOB_ID}"

mkdir -p "${LOG_DIR}"

SERVER_ADDRESSES=()
SERVER_PIDS=()

cleanup() {
    echo "[cleanup] stopping sglang servers..."

    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done

    for pid in "${SERVER_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done

    echo "[cleanup] all sglang servers stopped"
}

trap cleanup EXIT INT TERM

echo "[info] log directory: ${LOG_DIR}"
echo "[info] starting SGLang servers..."

for idx in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$idx]}"
    port=$((40000 + idx * 10))
    addr="localhost:${port}"

    SERVER_ADDRESSES+=("${addr}")

    echo "[start] GPU ${gpu_id}, address ${addr}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -m sglang.launch_server \
        --model Qwen/Qwen3.5-4B \
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
        > "${LOG_DIR}/server_gpu${gpu_id}_port${port}.log" 2>&1 &

    SERVER_PIDS+=("$!")
done

WAIT_TIMEOUT=600

echo "[wait] waiting for all servers to become ready..."

for idx in "${!SERVER_ADDRESSES[@]}"; do
    addr="${SERVER_ADDRESSES[$idx]}"
    pid="${SERVER_PIDS[$idx]}"
    start_ts=$(date +%s)

    until ADDR="${addr}" python3 - <<'PY'
import os
import sys
import urllib.request

url = f"http://{os.environ['ADDR']}/health"

try:
    with urllib.request.urlopen(url, timeout=5) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
    do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[error] server ${addr} (pid ${pid}) exited during startup" >&2
            echo "[error] check logs in: ${LOG_DIR}" >&2
            exit 1
        fi

        current_ts=$(date +%s)

        if (( current_ts - start_ts >= WAIT_TIMEOUT )); then
            echo "[error] timed out waiting for server ${addr}" >&2
            echo "[error] check logs in: ${LOG_DIR}" >&2
            exit 1
        fi

        sleep 10
    done

    echo "[ready] server ${addr} is up"
done

echo "[run] all servers ready, starting regeneration..."
echo "[run] regen log: ${LOG_DIR}/regen.log"


# TODO@song: 这地方换成自己的路径即可
# hpc
# /scratch/Projects/CFP-04/CFP04-CF-054
if python3 scripts/regenerate_train_data.py \
    --model Qwen/Qwen3.5-4B \
    --concurrency 64 \
    --max-tokens 4096 \
    --server-address "${SERVER_ADDRESSES[@]}" \
    --temperature 0.0 \
    --top-p 0.95 \
    --top-k 20 \
    --input-file-path /local_home2/fengsicheng/specforge/data/llava-ov15-1M_train.jsonl \
    --output-file-path /local_home2/fengsicheng/specforge/regen_data/qwen35-4B_llava-ov15-1M_regen_first_turn.jsonl \
    --resume \
    --reasoning disable \
    > "${LOG_DIR}/regen.log" 2>&1
then
    echo "[done] regeneration finished successfully"
else
    status=$?
    echo "[error] regeneration failed with exit code ${status}" >&2
    echo "[error] check log: ${LOG_DIR}/regen.log" >&2
    exit "${status}"
fi

