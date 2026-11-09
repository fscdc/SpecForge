#!/bin/bash
# The concurrency tables (decode throughput at 1 to 32 requests in flight on the seven
# image benchmarks) served by vLLM instead of SGLang: the target alone and/or the
# MMFlash draft, one GPU, one server after the other in the same job so that each
# speedup is measured on one card. Submit from the repo root:
#
#   qsub scripts/benchmark_vllm_concurrency.sh                          # 4B: origin, then MMFlash
#   qsub -v CONCURRENCIES="8 16 32" scripts/benchmark_vllm_concurrency.sh
#   qsub -v MODEL=Qwen/Qwen3.5-9B,DRAFT=<9B export> scripts/benchmark_vllm_concurrency.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/benchmark_vllm_concurrency.sh  # without PBS
#
# Results land next to the SGLang ones, prefixed vllm_:
#   results/vllm_<origin|mmflash>_<qwen35-4B>_concurrency<N>_temp0_4096<RUN_TAG>_results.jsonl
# (bench_mm's format, so a rerun skips what a file already holds); the table comes from
#   python benchmarks/vllm/summarize_concurrency.py
#
# The server runs from its own vLLM environment (VLLM_ENV), so the SGLang environments
# and scripts are untouched. One-time setup of that environment:
#   python -m venv $VLLM_ENV && $VLLM_ENV/bin/pip install vllm==0.32.0
#   $VLLM_ENV/bin/pip install -e benchmarks/vllm/phase_plugin     # the phase accounting
# vLLM serves DFlash drafts natively (--speculative-config method=dflash); nothing in
# vLLM is patched, the plugin only measures. Its V2 model runner (the default in 0.32)
# gives the draft plain sequence positions, the target's M-RoPE ones going to the
# target only: the positions SpecForge trains the draft with and SGLang serves it with. The client is bench_mm.py through
# benchmarks/vllm/bench_mm_vllm.py, from the SpecForge environment.
#
# Kept as in the SGLang runs: questions and prompts, greedy decoding (top-p 0.95 and
# top-k 20 sent as bench_mm sends them), reasoning off, 4096-token cap, a warmup of N
# requests and a cache flush before each timed run, block 16 (15 draft tokens), bf16,
# 70% of GPU memory, and the server-side decode-step throughput the tables report.
#
# Optional (-v NAME=value):
#   METHODS        "origin mmflash" (default), or one of them
#   CONCURRENCIES  "1 2 4 8 16 32" (default)
#   BENCHES        the seven image benchmarks x200 (MMBench: task prompt for the 4B,
#                  boxed for the 9B, as in the SGLang tables)
#   RUN_TAG        appended to the results names, e.g. _smoke
#   TEMPERATURE    0 (default) or 1 (then top-p 0.95, top-k 20 sample)
#   VLLM_ENV       /scratch/$USER/envs/vllm0320
#   CUDA_COMPAT    NVIDIA's forward-compat libraries for a cu130 build on an older
#                  driver ("" to leave LD_LIBRARY_PATH alone)
#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N vllm-conc
#PBS -q auto
#PBS -l select=1:ncpus=12:mem=100gb:ngpus=1:gpu_model=H200
#PBS -l walltime=06:00:00

set -uo pipefail
export PYTHONUNBUFFERED=1
echo "[vllm] start $(date '+%F %T') host=$(hostname) job=${PBS_JOBID:-local} gpus=${CUDA_VISIBLE_DEVICES:-unset}"

# ------------------------------------------------------------- environment
# the client (bench_mm) runs from the SpecForge env, which has sglang for its helpers
CONDA_ENV="${CONDA_ENV:-specforge}"
if ! python3 -c "import sglang" > /dev/null 2>&1; then
    # /etc/bashrc and conda's activate scripts read unset variables
    set +u
    # shellcheck disable=SC1091
    [ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
    conda activate "${CONDA_ENV}" > /dev/null 2>&1 || true
    set -u
fi
if ! python3 -c "import sglang" > /dev/null 2>&1; then
    echo "[vllm] ERROR: the client needs the '${CONDA_ENV}' env (bench_mm imports sglang)" >&2
    exit 1
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2> /dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/benchmarks/bench_mm.py" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
cd "${PROJECT_DIR}" || exit 1

VLLM_ENV="${VLLM_ENV:-/scratch/${USER}/envs/vllm0320}"
VLLM_PY="${VLLM_ENV}/bin/python"
CUDA_COMPAT="${CUDA_COMPAT-/scratch/${USER}/envs/cuda-compat-13-0/usr/local/cuda-13.0/compat}"
SERVER_LD="${CUDA_COMPAT}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
[ -z "${CUDA_COMPAT}" ] && SERVER_LD="${LD_LIBRARY_PATH:-}"
if [ ! -x "${VLLM_PY}" ]; then
    echo "[vllm] ERROR: no vLLM environment at ${VLLM_ENV} (see the header)" >&2
    exit 1
fi
if ! "${VLLM_PY}" -c "import specforge_vllm_phase" > /dev/null 2>&1; then
    echo "[vllm] ERROR: the phase plugin is not installed in ${VLLM_ENV}:" >&2
    echo "         ${VLLM_PY} -m pip install -e benchmarks/vllm/phase_plugin" >&2
    exit 1
fi

# ------------------------------------------------------------- settings
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL}" | sed 's/^Qwen3\.5-/qwen35-/')}"
DRAFT="${DRAFT:-/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-mmflash-llava-ov15-1M-prompted-final}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
METHODS="${METHODS:-origin mmflash}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32}"
case "${MODEL_TAG}" in
    *9B) MMBENCH=mmbench ;;          # the 9B tables use the boxed prompt
    *) MMBENCH=mmbench-origin ;;
esac
BENCHES="${BENCHES:-chartqa:200 charxiv:200 mmstar:200 ${MMBENCH}:200 dynamath:200 mathvista:200 mathverse:200}"
TEMPERATURE="${TEMPERATURE:-0}"
case "${TEMPERATURE}" in 0|0.0) TEMP_TAG=0 ;; 1|1.0) TEMP_TAG=1 ;; *) TEMP_TAG="${TEMPERATURE}" ;; esac
RUN_TAG="${RUN_TAG:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"

IFS=',' read -ra VISIBLE <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${VISIBLE[0]}"
JOBNUM="${PBS_JOBID:-0}"; JOBNUM="${JOBNUM%%.*}"
case "${JOBNUM}" in ''|*[!0-9]*) JOBNUM=$$ ;; esac
PORT="${PORT:-$((36500 + (10#${JOBNUM} % 400) * 2))}"
PHASE_PORT=$((PORT + 1))
# every cache the server writes (torch.compile, Triton, FlashInfer JIT) stays on
# scratch: the server runs with HOME there, never the NFS home
WORK="${WORK:-/scratch/${USER}/tmp/vllm-bench/${PBS_JOBID:-local-$$}}"
# the client runs from here: the benchmark loaders write their images to
# .cache/<benchmark>_specforge under the working directory, and jobs sharing the
# repository's would rewrite the images another job is reading (the server then
# rejects a half-written image)
CLIENT_DIR="${WORK}/client"
mkdir -p "${WORK}/home" "${WORK}/tmp" "${CLIENT_DIR}" results
echo "[vllm] model=${MODEL} methods=${METHODS} concurrencies=${CONCURRENCIES} temperature=${TEMPERATURE}"
echo "[vllm] benches=${BENCHES}"
echo "[vllm] gpu=${GPU_ID} port=${PORT} phase=${PHASE_PORT} work=${WORK}"

server_env=(
    env
    "CUDA_VISIBLE_DEVICES=${GPU_ID}"
    "LD_LIBRARY_PATH=${SERVER_LD}"
    "HOME=${WORK}/home"
    "TMPDIR=${WORK}/tmp"
    "TRITON_CACHE_DIR=${WORK}/triton"
    "VLLM_CACHE_ROOT=/scratch/${USER}/.cache/vllm-$("${VLLM_PY}" -c 'import vllm; print(vllm.__version__)' 2> /dev/null)"
    "HF_HOME=${HF_HOME:-/scratch/${USER}/.cache/huggingface}"
    "HF_HUB_OFFLINE=1"
    # /reset_prefix_cache, /reset_mm_cache and /reset_encoder_cache, used after the warmup
    "VLLM_SERVER_DEV_MODE=1"
    "SPECFORGE_PHASE_PORT=${PHASE_PORT}"
)

# the cu130 torch must see the card before anything else is started
if ! "${server_env[@]}" "${VLLM_PY}" -c "
import torch, vllm
a = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
(a @ a).float().sum().item()
print('[vllm] torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__,
      torch.cuda.get_device_name(0), flush=True)
"; then
    echo "[vllm] ERROR: the vLLM environment cannot use the GPU" >&2
    exit 1
fi

SERVER_PID=""
stop_server() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2> /dev/null; then
        # the server and its EngineCore child share the process group setsid gave them
        kill -TERM -- "-${SERVER_PID}" 2> /dev/null
        for _ in $(seq 1 60); do kill -0 "${SERVER_PID}" 2> /dev/null || break; sleep 1; done
        kill -KILL -- "-${SERVER_PID}" 2> /dev/null
    fi
    SERVER_PID=""
}
trap stop_server EXIT

start_server() {  # $1 = origin | mmflash; sets SERVER_PID and SERVER_ARGS
    local method="$1" log="${WORK}/server_$1.log"
    SERVER_ARGS=(
        serve "${MODEL}"
        --host 127.0.0.1 --port "${PORT}"
        --dtype bfloat16
        --gpu-memory-utilization "${GPU_MEM_UTIL}"
        --max-model-len "${MAX_MODEL_LEN}"
        --attention-backend FLASH_ATTN
        --reasoning-parser qwen3
        --generation-config vllm
        --enable-per-request-metrics
        --enable-prompt-tokens-details
    )
    if [ "${method}" != origin ]; then
        SERVER_ARGS+=(
            --speculative-config "{\"method\": \"dflash\", \"model\": \"${DRAFT}\", \"num_speculative_tokens\": $((BLOCK_SIZE - 1))}"
            --per-request-spec-decode-metrics summary
        )
    fi
    echo "[vllm] $(date '+%T') starting ${method}: vllm ${SERVER_ARGS[*]}"
    echo "[vllm] server log ${log}"
    setsid "${server_env[@]}" "${VLLM_ENV}/bin/vllm" "${SERVER_ARGS[@]}" > "${log}" 2>&1 &
    SERVER_PID=$!
    if ! python3 - "${PORT}" "${SERVER_TIMEOUT}" "${SERVER_PID}" <<'PY'
import os, sys, time, urllib.request
port, timeout, pid = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read()
        print(f"[vllm] server ready on port {port}", flush=True)
        sys.exit(0)
    except Exception:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        print("[vllm] server exited before becoming ready", flush=True)
        sys.exit(1)
    time.sleep(5)
print(f"[vllm] server not ready within {timeout:.0f}s", flush=True)
sys.exit(1)
PY
    then
        tail -n 60 "${log}" >&2
        return 1
    fi
    # one tiny request: the per-request metrics and the phase counters must both be
    # there, or every number of the sweep would be missing
    python3 - "${PORT}" "${PHASE_PORT}" "${MODEL}" "${method}" <<'PY'
import json, sys, urllib.request
port, phase_port, model, method = sys.argv[1:5]
body = {"model": model, "max_tokens": 32, "temperature": 0.0,
        "messages": [{"role": "user", "content": "Count from one to ten."}],
        "chat_template_kwargs": {"enable_thinking": False}}
request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
payload = json.load(urllib.request.urlopen(request, timeout=300))
metrics = payload.get("metrics") or {}
spec = metrics.get("speculative_decoding")
phase = json.load(urllib.request.urlopen(f"http://127.0.0.1:{phase_port}/phase_accounting", timeout=30))
print(f"[vllm] probe: {payload['usage']['completion_tokens']} tokens, generation "
      f"{metrics.get('generation_time_ms')} ms, spec {json.dumps(spec)}, phase steps {phase['steps']}", flush=True)
problems = []
if metrics.get("generation_time_ms") is None:
    problems.append("no per-request timing metrics")
if method != "origin" and not spec:
    problems.append("no per-request speculative-decoding metrics")
if method == "origin" and spec:
    problems.append("the origin server is speculating")
if sum(phase["steps"].values()) == 0:
    problems.append("the phase plugin counted no step")
if problems:
    sys.exit("[vllm] ERROR: " + "; ".join(problems))
PY
}

overall=0
for METHOD in ${METHODS}; do
    case "${METHOD}" in
        origin) bench_block_args=(--block-size 0) ;;
        mmflash) bench_block_args=(--block-size "${BLOCK_SIZE}") ;;
        *) echo "[vllm] ERROR: unknown method ${METHOD}" >&2; exit 1 ;;
    esac
    if ! start_server "${METHOD}"; then
        echo "[vllm] ERROR: the ${METHOD} server did not come up" >&2
        stop_server
        overall=1
        continue
    fi
    for C in ${CONCURRENCIES}; do
        NAME="vllm_${METHOD}_${MODEL_TAG}_concurrency${C}_temp${TEMP_TAG}_4096${RUN_TAG}"
        echo "[vllm] $(date '+%T') ===== ${METHOD} concurrency ${C} -> results/${NAME}_results.jsonl"
        # bench_mm is only an HTTP client, but importing it runs sglang/test/test_utils.py,
        # which does int(CUDA_VISIBLE_DEVICES[0]) and dies on the GPU UUIDs PBS hands out
        # shellcheck disable=SC2086
        (cd "${CLIENT_DIR}" && CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_ARGS="${SERVER_ARGS[*]}" \
            python3 "${PROJECT_DIR}/benchmarks/vllm/bench_mm_vllm.py" \
            --phase-url "http://127.0.0.1:${PHASE_PORT}" \
            --model "${MODEL}" \
            --base-url "http://127.0.0.1:${PORT}" \
            --concurrency "${C}" \
            "${bench_block_args[@]}" \
            --benchmark-list ${BENCHES} \
            --reasoning off \
            --temperature "${TEMPERATURE}" \
            --top-p 0.95 \
            --top-k 20 \
            --max-tokens 4096 \
            --output-dir "${PROJECT_DIR}/results" \
            --name "${NAME}")
        status=$?
        echo "[vllm] $(date '+%T') ${METHOD} concurrency ${C} exit ${status}"
        [ "${status}" -ne 0 ] && overall=1
        if ! kill -0 "${SERVER_PID}" 2> /dev/null; then
            echo "[vllm] ERROR: the ${METHOD} server died" >&2
            tail -n 60 "${WORK}/server_${METHOD}.log" >&2
            overall=1
            break
        fi
    done
    stop_server
done
# the images the loaders wrote; the server logs stay
rm -rf "${CLIENT_DIR}/.cache"
echo "[vllm] $(date '+%F %T') done, exit ${overall}"
exit "${overall}"
