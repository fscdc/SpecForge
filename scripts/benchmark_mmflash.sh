# 1) export the training checkpoint to an HF directory
# specforge export --to hf \
#   --checkpoint /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/outputs/qwen3.5-4b-mmflash-llava-ov15-1M/qwen3.5-4b-mmflash-step100000 \
#   --draft-config configs/qwen3.5-4b-mmflash.json \
#   --output-dir /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-mmflash-llava-ov15-1M-100000
#
# 2) rewrite architectures -> DFlashDraftModel so SGLang's DFLASH loader accepts it
# python scripts/gates/normalize_dflash_export.py \
#   --config /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-mmflash-llava-ov15-1M-100000/config.json \
#   --block-size 16

export CUDA_VISIBLE_DEVICES=0
export SGLANG_FORCE_STREAM_INTERVAL=1

GPU_IDS=(0)

# for deep100
# export LD_LIBRARY_PATH="/home/fengsicheng/miniconda3/envs/specforge/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
# export FLASHINFER_USE_CUDA_NORM=1
# export NVCC_PREPEND_FLAGS="-ccbin g++-11"

# for hopper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1
export SGLANG_NUMA_BIND_V2=0


# z-lab/Qwen3.5-4B-DFlash
# /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-mmflash-sharegpt4v
# /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-mmflash-hf 这个是一个只有1000step的test版本
# /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-dflash-baseline-llava-ov15-1M-50000 用llava那个数据集训的baseline版本，数据没有改prompt
# /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-dflash-baseline-llava-ov15-1M-prompted-final


BLOCK_SIZE=16

# Draft context window, in target tokens. Empty = the draft attends to the whole
# context (default). Set it to add --speculative-draft-window-size: SGLang then
# keeps a compact per-request draft cache and the DFlash draft attends only to
# the most recent DRAFT_WINDOW target tokens (no sink, no top-k). Training-free,
# so it is the cheapest "sparse attention for the draft" experiment; the result
# name gets a _win<N> suffix so runs stay apart. scripts/sweep_draft_window.sh
# drives a sweep. NAME_SUFFIX is a free extra tag appended after it.
DRAFT_WINDOW="${DRAFT_WINDOW:-}"
# Sparse draft context on top of the window (needs DRAFT_WINDOW set): the
# patched worker reads SGLANG_DFLASH_DRAFT_SPARSE, e.g.
#   DRAFT_SPARSE="sink=4,text=1,stride=32,window=2048"
# meaning sink positions + every text token + one of every 32 visual tokens
# per frame + the most recent 2048 tokens. patches/sglang/v0.5.14/
# dflash-draft-sparse-context.patch documents the semantics; the result name
# gets a _sparse-<spec> suffix. See scripts/sweep_draft_sparse.sh.
DRAFT_SPARSE="${DRAFT_SPARSE:-}"
if [ -n "${DRAFT_SPARSE}" ]; then
    if [ -z "${DRAFT_WINDOW}" ]; then
        echo "DRAFT_SPARSE needs DRAFT_WINDOW (the compact draft cache)" >&2
        exit 1
    fi
    export SGLANG_DFLASH_DRAFT_SPARSE="${DRAFT_SPARSE}"
    SPARSE_SUFFIX="_sparse-$(printf '%s' "${DRAFT_SPARSE}" | tr -d ' ' | sed -e 's/=//g' -e 's/,/-/g')"
else
    unset SGLANG_DFLASH_DRAFT_SPARSE
    SPARSE_SUFFIX=""
fi
NAME_SUFFIX="${NAME_SUFFIX:-}"
RUN_SUFFIX="${DRAFT_WINDOW:+_win${DRAFT_WINDOW}}${SPARSE_SUFFIX}${NAME_SUFFIX}"

# Patch the installed SGLang so every response carries its prefill/decode split
# (first_token_latency / decode_latency). Must run before launch_server imports
# tokenizer_manager.py. `bash scripts/benchmark_helper.sh --unpatch` undoes it.
bash scripts/benchmark_helper.sh || exit 1


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
    port=$((32000 + idx * 10))
    SERVER_ADDRESSES+=("localhost:${port}")
    PORTS+=("${port}")
    BASE_URLS+=("http://localhost:${port}")
    CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m sglang.launch_server \
        --model Qwen/Qwen3.5-4B \
        --speculative-algorithm DFLASH \
        --speculative-draft-model-path /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-mmflash-llava-ov15-1M-prompted-final \
        --speculative-dflash-block-size ${BLOCK_SIZE} \
        ${DRAFT_WINDOW:+--speculative-draft-window-size ${DRAFT_WINDOW}} \
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




# CONCURRENCIES="${CONCURRENCIES:-2 4 8 16}"

# for CONC in ${CONCURRENCIES}; do
#     echo "===== concurrency ${CONC} ====="
#     python benchmarks/bench_mm.py \
#         --model Qwen/Qwen3.5-4B \
#         --base-url "${BASE_URLS[@]}" \
#         --concurrency ${CONC} \
#         --block-size ${BLOCK_SIZE} \
#         --benchmark-list chartqa:200 charxiv:200 mmstar:200 mmbench-origin:200 dynamath:200 mathvista:200 mathverse:200  \
#         --reasoning off \
#         --temperature 0.0 \
#         --top-p 0.95 \
#         --top-k 20 \
#         --max-tokens 4096 \
#         --name "mmflash_qwen35-4B_concurrency${CONC}_temp0_4096"
# done

# python benchmarks/bench_mm.py \
#     --model Qwen/Qwen3.5-4B \
#     --base-url "${BASE_URLS[@]}" \
#     --concurrency 1 \
#     --block-size ${BLOCK_SIZE} \
#     --benchmark-list chartqa:200 charxiv:200 mmstar:200 mmbench-origin:200 dynamath:200 mathvista:200 mathverse:200  \
#     --reasoning off \
#     --temperature 0.0 \
#     --top-p 0.95 \
#     --top-k 20 \
#     --max-tokens 4096 \
#     --name mmflash_qwen35-4B_concurrency1_temp0_4096


# python benchmarks/bench_mm.py \
#     --model Qwen/Qwen3.5-4B \
#     --base-url "${BASE_URLS[@]}" \
#     --concurrency 1 \
#     --block-size ${BLOCK_SIZE} \
#     --benchmark-list chartqa:200 charxiv:200 mmstar:200 mmbench-origin:200 dynamath:200 mathvista:200 mathverse:200  \
#     --reasoning off \
#     --temperature 1.0 \
#     --top-p 0.95 \
#     --top-k 20 \
#     --max-tokens 4096 \
#     --name mmflash_qwen35-4B_concurrency1_temp1_4096


python benchmarks/bench_mm.py \
    --model Qwen/Qwen3.5-4B \
    --base-url "${BASE_URLS[@]}" \
    --concurrency 1 \
    --block-size ${BLOCK_SIZE} \
    --benchmark-list vdc:20  \
    --reasoning off \
    --temperature 0.0 \
    --top-p 0.95 \
    --top-k 20 \
    --max-tokens 4096 \
    --name "video_mmflash_qwen35-4B_concurrency1_temp0_4096${RUN_SUFFIX}"


# # for text benchmark
# python benchmarks/bench_text.py \
#     --model Qwen/Qwen3.5-4B \
#     --base-url "${BASE_URLS[@]}" \
#     --concurrency 1 \
#     --block-size ${BLOCK_SIZE} \
#     --benchmark-list gsm8k:200 \
#     --reasoning off \
#     --temperature 0.0 \
#     --top-p 0.95 \
#     --top-k 20 \
#     --max-tokens 8192 \
#     --name dflash_qwen35-4B_concurrency1


pkill -f "sglang.launch_server"


