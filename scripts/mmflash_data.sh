#!/bin/bash


# ## env
# # pip install -v --pre .
# # pip install --force-reinstall torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# # pip uninstall -y sgl-deep-gemm







# dataset: sharegpt4v / sharegpt4v-pt
# hpc: /Project_Storage/CFP-04/CFP04-CF-054/fengsicheng/specforge/data
# python scripts/prepare_data_mm.py \
#     --dataset sharegpt4v \
#     --image-root /local_home1/fengsicheng/specforge/data \
#     --output-path /local_home1/fengsicheng/specforge/data/ \
    # --dump-missing-prefix web-celebrity/images
    # --sample-size 10000 \

GPU_IDS=(0 1)

# regen dataset: sharegpt4v

# for deep100
# export LD_LIBRARY_PATH="/home/svu/fengsicheng/miniconda3/envs/specforge/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
# export FLASHINFER_USE_CUDA_NORM=1
# export NVCC_PREPEND_FLAGS="-ccbin g++-11"

# for hopper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1

# SERVER_ADDRESSES=()
# for idx in "${!GPU_IDS[@]}"; do
#     gpu_id="${GPU_IDS[$idx]}"
#     port=$((30000 + idx * 10))
#     SERVER_ADDRESSES+=("localhost:${port}")
#     CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m sglang.launch_server \
#         --model Qwen/Qwen3.5-4B \
#         --mem-fraction-static 0.7 \
#         --tp 1 \
#         --trust-remote-code \
#         --cuda-graph-max-bs 128 \
#         --attention-backend triton \
#         --mm-attention-backend sdpa \
#         --host 0.0.0.0 \
#         --port ${port} \
#         --dtype bfloat16 \
#         --reasoning-parser qwen3 &
# done


# deep100: /local_home1/fengsicheng/specforge
# hopper: /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge

SERVER_ADDRESSES=()
for idx in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$idx]}"
    port=$((30000 + idx * 10))
    SERVER_ADDRESSES+=("localhost:${port}")
done

python scripts/regenerate_train_data_mm.py \
    --model Qwen/Qwen3.5-4B \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address "${SERVER_ADDRESSES[@]}" \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --input-file-path /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/data/sharegpt4v-pt_train.jsonl \
    --output-file-path /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/regen_data/qwen35-4B_sharegpt4v-pt_regen_first_turn.jsonl \
    --resume \
    --reasoning disable



# pkill -f "sglang.launch_server"