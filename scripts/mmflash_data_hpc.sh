#!/bin/bash
# 启动 sglang server -> 等全部就绪 -> 跑 regen -> 自动清理 server
# 用法: bash scripts/mmflash_regen.sh

set -uo pipefail


export LD_LIBRARY_PATH="/home/svu/fengsicheng/miniconda3/envs/specforge/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1
export NVCC_PREPEND_FLAGS="-ccbin g++-11"

# Slurm 分配的 GPU 在作业内部从 0 开始重新编号,和申请的 GPU 数保持一致即可
GPU_IDS=(0 1)

LOG_DIR="logs/regen_${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_DIR}"

SERVER_ADDRESSES=()
SERVER_PIDS=()

# 脚本退出时(正常结束/报错/被 scancel)把 server 杀掉,避免作业挂着不退出
cleanup() {
    echo "[cleanup] stopping sglang servers..."
    for pid in "${SERVER_PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null
}
trap cleanup EXIT

for idx in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$idx]}"
    port=$((30000 + idx * 10))
    SERVER_ADDRESSES+=("localhost:${port}")
    CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m sglang.launch_server \
        --model Qwen/Qwen3.5-4B \
        --mem-fraction-static 0.7 \
        --tp 1 \
        --trust-remote-code \
        --cuda-graph-max-bs 128 \
        --attention-backend triton \
        --mm-attention-backend sdpa \
        --host 0.0.0.0 \
        --port ${port} \
        --dtype bfloat16 \
        --reasoning-parser qwen3 \
        > "${LOG_DIR}/server_gpu${gpu_id}_port${port}.log" 2>&1 &
    SERVER_PIDS+=($!)
done

# 等所有 server 就绪: 模型加载 + cuda graph capture 完成后 /health 才会返回 200
WAIT_TIMEOUT=1800   # 最多等 30 分钟
for idx in "${!SERVER_ADDRESSES[@]}"; do
    addr="${SERVER_ADDRESSES[$idx]}"
    pid="${SERVER_PIDS[$idx]}"
    start_ts=$(date +%s)
    until curl -sf "http://${addr}/health" > /dev/null; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[error] server ${addr} (pid ${pid}) exited during startup, see ${LOG_DIR}" >&2
            exit 1
        fi
        if (( $(date +%s) - start_ts >= WAIT_TIMEOUT )); then
            echo "[error] timed out waiting for server ${addr}" >&2
            exit 1
        fi
        sleep 10
    done
    echo "[ready] server ${addr} is up"
done

echo "[run] all servers ready, starting regeneration..."

python scripts/regenerate_train_data_mm.py \
    --model Qwen/Qwen3.5-4B \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address "${SERVER_ADDRESSES[@]}" \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --input-file-path /local_home1/fengsicheng/specforge/data/sharegpt4v-pt_train.jsonl \
    --output-file-path /local_home1/fengsicheng/specforge/regen_data/qwen35-4B_sharegpt4v-pt_regen_first_turn.jsonl \
    --resume \
    --reasoning disable

echo "[done] regeneration finished"
# server 由 trap cleanup 自动回收
