#!/bin/bash

# dataset: sharegpt4v
python scripts/prepare_data_mm.py \
    --dataset sharegpt4v \
    --image-root /local_home3/fengsicheng/specforge/data \
    --sample-size 100 \


# regen dataset: sharegpt4v
# Edit GPU_IDS to pick which 4 physical GPUs to use (one server per GPU).
GPU_IDS=(0 1 2 3)
SERVER_ADDRESSES=()
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
        --host 0.0.0.0 \
        --port ${port} \
        --dtype bfloat16 \
        --reasoning-parser qwen3 &
done

# wait for all 4 servers to finish loading before sending traffic
for addr in "${SERVER_ADDRESSES[@]}"; do
    until curl -s -o /dev/null "http://${addr}/health"; do
        sleep 5
    done
done

python scripts/regenerate_train_data_mm.py \
    --model Qwen/Qwen3.5-4B \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address "${SERVER_ADDRESSES[@]}" \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --input-file-path /local_home3/fengsicheng/specforge/data/sharegpt4v_train.jsonl \
    --output-file-path /local_home3/fengsicheng/specforge/regen_data/sharegpt4v_regen_first_turn.jsonl \
    --resume \
    --reasoning disable