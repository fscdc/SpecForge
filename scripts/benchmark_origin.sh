export CUDA_VISIBLE_DEVICES=0

GPU_IDS=(0)


# for deep100
export LD_LIBRARY_PATH="/home/svu/fengsicheng/miniconda3/envs/specforge/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
export FLASHINFER_USE_CUDA_NORM=1
export NVCC_PREPEND_FLAGS="-ccbin g++-11"

# for hopper
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
# export FLASHINFER_USE_CUDA_NORM=1

export HF_HUB_DISABLE_XET=1


# Patch the installed SGLang so every response carries its prefill/decode split
# (first_token_latency / decode_latency). Must run before launch_server imports
# tokenizer_manager.py. `bash scripts/benchmark_helper.sh --unpatch` undoes it.
bash scripts/benchmark_helper.sh || exit 1


# The scheduler decides which GPUs this job gets and may name them by UUID
# (GPU-xxxxxxxx-...), so GPU_IDS indexes into that list rather than assuming the
# device numbers are 0,1,2,...
IFS=',' read -ra VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

SERVER_ADDRESSES=()
PORTS=()
BASE_URLS=()
for idx in "${!GPU_IDS[@]}"; do
    gpu_id="${VISIBLE_GPUS[${GPU_IDS[$idx]}]:-}"
    if [ -z "${gpu_id}" ]; then
        echo "GPU ${GPU_IDS[$idx]} is not among the ${#VISIBLE_GPUS[@]} GPU(s) of this job" >&2
        exit 1
    fi
    port=$((30000 + idx * 10))
    SERVER_ADDRESSES+=("localhost:${port}")
    PORTS+=("${port}")
    BASE_URLS+=("http://localhost:${port}")
    CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m sglang.launch_server \
        --model Qwen/Qwen3.5-4B \
        --mem-fraction-static 0.7 \
        --tp 1 \
        --trust-remote-code \
        --cuda-graph-max-bs 128 \
        --attention-backend fa3 \
        --mm-attention-backend sdpa \
        --host 0.0.0.0 \
        --port ${port} \
        --dtype bfloat16 \
        --reasoning-parser qwen3 &
done


# The servers load in the background, so wait for all of them before benchmarking.
# Polling is done with python rather than curl, which is not available everywhere,
# and it costs nothing but the standard library. Replace the whole block with
# `sleep 300` if you would rather wait blind.
python3 - "${SERVER_ADDRESSES[@]}" <<'PY'
import sys
import time
import urllib.error
import urllib.request

TIMEOUT = 1800  # a cold start pulling weights from disk can take a while
deadline = time.time() + TIMEOUT

for address in sys.argv[1:]:
    while True:
        try:
            urllib.request.urlopen(f"http://{address}/health", timeout=5).read()
            print(f"server {address} is ready", flush=True)
            break
        except (urllib.error.URLError, OSError) as error:
            if time.time() > deadline:
                sys.exit(f"server {address} was not ready within {TIMEOUT}s: {error}")
            time.sleep(5)
PY
# stop here if a server never came up, benchmarking would only fail later anyway
if [ $? -ne 0 ]; then
    exit 1
fi


# # temperature = 1.0
# python benchmarks/bench_mm.py \
#     --model Qwen/Qwen3.5-4B \
#     --base-url "${BASE_URLS[@]}" \
#     --concurrency 1 \
#     --block-size 0 \
#     --benchmark-list chartqa:200 mmstar:200 \
#     --reasoning off \
#     --temperature 1.0 \
#     --top-p 0.95 \
#     --top-k 20 \
#     --max-tokens 8192 \
#     --name origin_qwen35-4B_concurrency1

# temperature = 0.0
#     --save-generations \
# chartqa:200 mmstar:200 realworldqa:200 mmmu:200 textvqa:200 seedbench-image:200 mathvision:200 vdc:20
python benchmarks/bench_mm.py \
    --model Qwen/Qwen3.5-4B \
    --base-url "${BASE_URLS[@]}" \
    --concurrency 1 \
    --block-size 0 \
    --benchmark-list dynamath:200 seedbench-image:200 \
    --reasoning off \
    --temperature 0.0 \
    --top-p 0.95 \
    --top-k 20 \
    --max-tokens 8192 \
    --name origin_qwen35-4B_concurrency1_temp0


# # for text benchmark
# python benchmarks/bench_text.py \
#     --model Qwen/Qwen3.5-4B \
#     --base-url "${BASE_URLS[@]}" \
#     --concurrency 1 \
#     --block-size 0 \
#     --benchmark-list gsm8k:200 \
#     --reasoning off \
#     --temperature 0.0 \
#     --top-p 0.95 \
#     --top-k 20 \
#     --max-tokens 8192 \
#     --name origin_qwen35-4B_concurrency1_temp0


# pkill -f "sglang.launch_server"