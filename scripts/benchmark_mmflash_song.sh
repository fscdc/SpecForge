# 对于draft model，我们需要先转换为一个sglang可以load的形式
# TODO: 改成真实路径
specforge export --to hf \
  --checkpoint /TODO/outputs/qwen3.5-4b-mmflash/qwen3.5-4b-mmflash_latest \
  --draft-config configs/qwen3.5-4b-dflash.json \
  --output-dir /TODO/specforge/draft_models/qwen3.5-4b-mmflash-hf


# TODO: 这个地方把要多少卡一些测试的GPU_IDs写在这里就行了，后面会自动根据这个列表去启动server
export CUDA_VISIBLE_DEVICES=0,1

GPU_IDS=(0 1)

# 对于松哥的环境，下面这两个export的东西应该都不需要
# for deep100
# export LD_LIBRARY_PATH="/home/svu/fengsicheng/miniconda3/envs/specforge/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
# export FLASHINFER_USE_CUDA_NORM=1
# export NVCC_PREPEND_FLAGS="-ccbin g++-11"

# for hopper
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
# export FLASHINFER_USE_CUDA_NORM=1


BLOCK_SIZE=16

IFS=',' read -ra VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# TODO: 这里把draft model的路径写在这里就行了
SERVER_ADDRESSES=()
PORTS=()
for idx in "${!GPU_IDS[@]}"; do
    gpu_id="${VISIBLE_GPUS[${GPU_IDS[$idx]}]:-}"
    if [ -z "${gpu_id}" ]; then
        echo "GPU ${GPU_IDS[$idx]} is not among the ${#VISIBLE_GPUS[@]} GPU(s) of this job" >&2
        exit 1
    fi
    port=$((30000 + idx * 10))
    SERVER_ADDRESSES+=("localhost:${port}")
    PORTS+=("${port}")
    CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m sglang.launch_server \
        --model Qwen/Qwen3.5-4B \
        --speculative-algorithm DFLASH \
        --speculative-draft-model-path /TODO/exports/qwen3.5-4b-mmflash-hf \
        --speculative-dflash-block-size ${BLOCK_SIZE} \
        --mem-fraction-static 0.7 \
        --tp 1 \
        --trust-remote-code \
        --cuda-graph-max-bs 128 \
        --attention-backend triton \
        --mm-attention-backend sdpa \
        --host 0.0.0.0 \
        --port ${port} \
        --dtype bfloat16 \
        --reasoning-parser qwen3 &
done



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


CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_mmflash.py \
    --model-path Qwen/Qwen3.5-4B \
    --ports "${PORTS[@]}" \
    --config-list 64,${BLOCK_SIZE} \
    --benchmark-list ocrbench chartqa realworldqa mmstar \
    --dtype bfloat16 \
    --chat-template-name qwen2-vl \
    --disable-thinking \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --max-tokens 4096 \
    --skip-launch-server \
    --save-generations \
    --name mmflash_qwen35-4B_concurrency128