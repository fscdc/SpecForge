#!/bin/bash
# Continued MMFlash training on the LLaVA-Video rows, one 2-GPU PBS job per run.
# Any other managed_local run config launches the same way through CONFIG, e.g.
# the Gemma 4 E4B recipes scripts/mmtraining_configs/gemma4-e4b-*_hpc.yaml.
# Submit from the repo root:
#
#   qsub scripts/train_mmflash_video_hpc.sh                                      # sparse run
#   qsub -v CONFIG=scripts/mmtraining_configs/qwen3.5-4b-mmflash-video-dense_hpc.yaml scripts/train_mmflash_video_hpc.sh
#   qsub -l select=1:ngpus=4 -v CONFIG=scripts/mmtraining_configs/gemma4-e4b-eagle3_hpc.yaml scripts/train_mmflash_video_hpc.sh
#
# - The job's first GPU serves capture, the others train (passed by UUID, as
#   PBS hands them out), and deployment.trainer.nproc_per_node follows their
#   count whatever the config says: 2 GPUs train on 1, 4 GPUs on 3.
# - With ONE GPU (qsub -l select=1:ngpus=1 ...) the capture server and the
#   trainer share it: the capture server's static memory drops to
#   SHARED_CAPTURE_MEM (default 0.30 of the card) and the rest is the trainer's.
#   GPU memory is logged every minute to <output_dir>/logs/gpu_<jobid>.log.
# - STACK=1 stacks both on EVERY GPU of the job: one capture server
#   tensor-parallel over all of them (tp_size = GPU count) and one trainer rank
#   per GPU, the capture server again held to SHARED_CAPTURE_MEM per card. For
#   a target too big to share one card with its trainer, e.g. the 27B:
#     qsub -l select=1:ncpus=24:mem=450gb:ngpus=2:gpu_model=H200 -v STACK=1,CONFIG=scripts/mmtraining_configs/qwen3.8-27b-mmflash_hpc.yaml scripts/train_mmflash_video_hpc.sh
#   CAPTURE_TP=k splits the cards into capture servers of k GPUs each instead;
#   CAPTURE_TP=1 gives every card its own capture replica, which suits a target
#   that fits on one card next to its trainer rank (Gemma 4 E4B, Qwen3.5-4B/9B):
#     qsub -l select=1:ncpus=24:mem=450gb:ngpus=2:gpu_model=H200 -v STACK=1,CAPTURE_TP=1,CONFIG=scripts/mmtraining_configs/gemma4-e4b-mmflash_hpc.yaml scripts/train_mmflash_video_hpc.sh
# - SEGMENT_EPOCHS=E splits training.num_epochs into segments of E epochs, one
#   job each, for a run longer than one walltime (managed_local cannot resume
#   inside a stream). SEGMENT=k (from 0) streams epochs [k*E, (k+1)*E) of the
#   one plan (training.online_epoch_range) on the whole run's LR horizon, and
#   from k=1 on continues from segment k-1's final checkpoint
#   (training.continue_from: weights, optimizer, scheduler, RNG, global_step).
#   A segment that exits cleanly hardlinks its final checkpoint to
#   <output_dir>/segments/segment<k>-final (safe from max_checkpoints rotation)
#   and records it in <output_dir>/segments/segment<k>.json; the next segment
#   refuses to start without it. Chain the jobs with afterok:
#     s0=$(qsub -v SEGMENT_EPOCHS=3,SEGMENT=0,CONFIG=... scripts/train_mmflash_video_hpc.sh)
#     qsub -W depend=afterok:${s0} -v SEGMENT_EPOCHS=3,SEGMENT=1,CONFIG=... scripts/train_mmflash_video_hpc.sh
# - The job runs from a copy of specforge/, configs/ and scripts/ taken when it
#   starts (<output_dir>/code/<jobid>), so edits to the checkout during a long
#   run cannot reach it; CODE_SNAPSHOT=0 runs the checkout itself.
# - EXTRA_OVERRIDES: more dotted overrides, whitespace-separated on one line and
#   applied last. output_dir, control_dir and consumer_state_dir are read after
#   them, so a smoke run on a data subset logs to and clears its own directories:
#     export EXTRA_OVERRIDES="data.train_data_path=<subset.jsonl> training.num_epochs=1 training.log_interval=10 run_id=<id> output_dir=<dir> deployment.disaggregated.control_dir=<dir>/control deployment.disaggregated.consumer_state_dir=<dir>/consumer-state"
#     qsub -l select=1:ngpus=1 -v CONFIG=<yaml>,EXTRA_OVERRIDES scripts/train_mmflash_video_hpc.sh
#   (-v NAME without a value passes the submitting shell's exported value; no
#   spaces or commas inside a value).
# - DRY_RUN=1 prints the overrides and the resolved launch plan (specforge train
#   --plan) and exits before anything is moved or started.
# - Capture and Mooncake ports are derived from the job number, so two jobs (or
#   other users' jobs) on one node do not collide.
# - managed_local needs a fresh control_dir: an earlier attempt's control/ and
#   consumer-state/ are moved to <output_dir>/attempt-<jobid>/ (checkpoints stay).
# - SGLANG_MM_SKIP_COMPUTE_HASH=1 (default) skips the capture server's per-image
#   sha256 (~1 s per 48-frame row); outputs are identical. -v SGLANG_MM_SKIP_COMPUTE_HASH=0
#   turns it off.
# The log is <output_dir>/logs/train_<jobid>.log.
#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N mmflash-video
#PBS -q auto
#PBS -l select=1:ngpus=2
#PBS -l walltime=36:00:00

set -uo pipefail
export PYTHONUNBUFFERED=1
echo "[train] start $(date '+%F %T') host=$(hostname) job=${PBS_JOBID:-local} gpus=${CUDA_VISIBLE_DEVICES:-unset}"

# ------------------------------------------------------------- environment
CONDA_ENV="${CONDA_ENV:-specforge}"
if ! python3 -c "import torch" > /dev/null 2>&1; then
    # /etc/bashrc and conda's activate scripts read unset variables, which
    # set -u turns into an immediate exit
    set +u
    # shellcheck disable=SC1091
    [ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
    conda activate "${CONDA_ENV}" > /dev/null 2>&1 || true
    set -u
fi
if ! python3 -c "import torch" > /dev/null 2>&1; then
    echo "[train] ERROR: torch is not importable after activating '${CONDA_ENV}'" >&2
    exit 1
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2> /dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/specforge/cli.py" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
cd "${PROJECT_DIR}" || exit 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1
export SGLANG_NUMA_BIND_V2=0
export SGLANG_MM_SKIP_COMPUTE_HASH="${SGLANG_MM_SKIP_COMPUTE_HASH:-1}"
# Compile caches per JOB: a per-host inductor cache can hold entries whose
# Triton cubins lived in another job's TRITON_CACHE_DIR, and loading one fails
# with "CUDA driver error: file not found" at the first flex_attention step.
CACHE_ROOT="${SPECFORGE_CACHE_ROOT:-/scratch/${USER}/tmp}"
CACHE_TAG="${PBS_JOBID:-local-$$}"; CACHE_TAG="${CACHE_TAG%%.*}"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton-train-${CACHE_TAG}"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor-train-${CACHE_TAG}"
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# ------------------------------------------------------------- settings
CONFIG="${CONFIG:-scripts/mmtraining_configs/qwen3.5-4b-mmflash-video-sparse_hpc.yaml}"
[ -f "${CONFIG}" ] || { echo "[train] ERROR: config ${CONFIG} not found under ${PROJECT_DIR}" >&2; exit 1; }
read -r -a EXTRA <<< "${EXTRA_OVERRIDES:-}"

IFS=',' read -ra VISIBLE <<< "${CUDA_VISIBLE_DEVICES:-}"
if [ "${#VISIBLE[@]}" -eq 0 ] || [ -z "${VISIBLE[0]}" ]; then
    echo "[train] ERROR: needs at least 1 GPU, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
    exit 1
fi
if [ "${STACK:-0}" = 1 ] || [ "${#VISIBLE[@]}" -eq 1 ]; then
    CAPTURE_GPUS=("${VISIBLE[@]}")
    TRAINER_GPUS=("${VISIBLE[@]}")
    SHARED=1
else
    CAPTURE_GPUS=("${VISIBLE[0]}")
    TRAINER_GPUS=("${VISIBLE[@]:1}")
    SHARED=0
fi
CAPTURE_TP="${CAPTURE_TP:-${#CAPTURE_GPUS[@]}}"
case "${CAPTURE_TP}" in ''|*[!0-9]*) CAPTURE_TP=0 ;; esac
if (( CAPTURE_TP < 1 || ${#CAPTURE_GPUS[@]} % CAPTURE_TP )); then
    echo "[train] ERROR: CAPTURE_TP=${CAPTURE_TP} must divide the ${#CAPTURE_GPUS[@]} capture GPU(s)" >&2
    exit 1
fi
CAPTURE_SERVERS=$(( ${#CAPTURE_GPUS[@]} / CAPTURE_TP ))
CAPTURE_CSV="$(IFS=,; echo "${CAPTURE_GPUS[*]}")"
TRAINER_CSV="$(IFS=,; echo "${TRAINER_GPUS[*]}")"
# YAML flow lists of quoted tokens: a bare numeric index would parse as an int
yaml_list() { local l; l="$(printf '"%s", ' "$@")"; echo "[${l%, }]"; }
TRAINER_LIST="$(yaml_list "${TRAINER_GPUS[@]}")"

# output_dir, control_dir, consumer_state_dir and the capture server's memory
# fraction, read from the config itself with EXTRA_OVERRIDES applied (a typo in
# them fails here, before anything is moved)
mapfile -t RESOLVED < <(python3 - "${CONFIG}" "${EXTRA[@]}" <<'PY'
import sys
from specforge.config import load_config
cfg = load_config(sys.argv[1], sys.argv[2:])
dis = cfg.deployment.disaggregated
mem = dis.managed_local.capture_servers[0].mem_fraction_static
print(cfg.output_dir, dis.control_dir, dis.consumer_state_dir or "", 0.7 if mem is None else mem,
      cfg.run_id, cfg.training.num_epochs, sep="\n")
PY
)
OUT="${RESOLVED[0]:-}" CONTROL="${RESOLVED[1]:-}" STATE="${RESOLVED[2]:-}" CAPTURE_MEM="${RESOLVED[3]:-}"
RUN_ID="${RESOLVED[4]:-}" NUM_EPOCHS="${RESOLVED[5]:-}"
if [ -z "${OUT}" ] || [ -z "${CONTROL}" ] || [ -z "${CAPTURE_MEM}" ] || [ -z "${RUN_ID}" ] || [ -z "${NUM_EPOCHS}" ]; then
    echo "[train] ERROR: cannot resolve ${CONFIG}${EXTRA_OVERRIDES:+ with EXTRA_OVERRIDES}" >&2
    exit 1
fi
SHARE_OVERRIDE=()
if [ "${SHARED}" = 1 ]; then
    CAPTURE_MEM="${SHARED_CAPTURE_MEM:-0.30}"
    SHARE_OVERRIDE=("deployment.disaggregated.managed_local.share_trainer_gpu=true")
fi
JOBNUM="${PBS_JOBID:-0}"; JOBNUM="${JOBNUM%%.*}"
case "${JOBNUM}" in ''|*[!0-9]*) JOBNUM=0 ;; esac
OFFSET=$(( (10#${JOBNUM} % 400) * 10 ))
CAPTURE_PORT=$(( 31000 + OFFSET ))
MC_RPC=$(( 45000 + OFFSET ))
# one capture server per CAPTURE_TP cards, on consecutive ports
SERVERS=""
for (( i = 0; i < CAPTURE_SERVERS; i++ )); do
    SERVERS+="{port: $(( CAPTURE_PORT + i )), cuda_visible_devices: $(yaml_list "${CAPTURE_GPUS[@]:i * CAPTURE_TP:CAPTURE_TP}"), tp_size: ${CAPTURE_TP}, mem_fraction_static: ${CAPTURE_MEM}}, "
done
SERVERS="[${SERVERS%, }]"
MC_METADATA=$(( 45001 + OFFSET ))
MC_METRICS=$(( 45002 + OFFSET ))

# segments: this job's epochs, and the previous segment's final checkpoint
SEGMENT_OVERRIDES=()
SEG_NOTE=""
CONTINUE_FROM=""
if [ -n "${SEGMENT_EPOCHS:-}" ]; then
    SEGMENT="${SEGMENT:-0}"
    case "${SEGMENT_EPOCHS}:${SEGMENT}" in *[!0-9:]*|:*|*:)
        echo "[train] ERROR: SEGMENT_EPOCHS and SEGMENT must be integers, got ${SEGMENT_EPOCHS}:${SEGMENT}" >&2; exit 1 ;;
    esac
    SEG_START=$(( SEGMENT * SEGMENT_EPOCHS ))
    SEG_END=$(( SEG_START + SEGMENT_EPOCHS ))
    (( SEG_END > NUM_EPOCHS )) && SEG_END=${NUM_EPOCHS}
    if (( SEGMENT_EPOCHS < 1 || SEG_START >= NUM_EPOCHS )); then
        echo "[train] ERROR: segment ${SEGMENT} of ${SEGMENT_EPOCHS} epoch(s) starts past num_epochs=${NUM_EPOCHS}" >&2
        exit 1
    fi
    SEG_DIR="${OUT}/segments"
    if [ -e "${SEG_DIR}/segment${SEGMENT}.json" ]; then
        echo "[train] ERROR: segment ${SEGMENT} already finished (${SEG_DIR}/segment${SEGMENT}.json); refusing to train its epochs twice" >&2
        exit 1
    fi
    SEGMENT_OVERRIDES=("training.online_epoch_range=[${SEG_START},${SEG_END}]")
    if (( SEGMENT > 0 )); then
        PREV="${SEG_DIR}/segment$(( SEGMENT - 1 )).json"
        if ! CONTINUE_FROM="$(python3 - "${PREV}" "${SEG_START}" <<'PY'
import json, os, sys
path, start = sys.argv[1], int(sys.argv[2])
if not os.path.isfile(path):
    sys.exit(f"no record of the previous segment: {path}")
with open(path) as handle:
    rec = json.load(handle)
if rec["epochs"][1] != start:
    sys.exit(f"{path} ends at epoch {rec['epochs'][1]}, this segment starts at {start}")
if not os.path.isfile(os.path.join(rec["checkpoint"], "training_state.pt")):
    sys.exit(f"the previous segment's final checkpoint is gone: {rec['checkpoint']}")
print(rec["checkpoint"])
PY
)"; then
            echo "[train] ERROR: cannot continue segment ${SEGMENT} (see above)" >&2
            exit 1
        fi
        SEGMENT_OVERRIDES+=("training.continue_from=${CONTINUE_FROM}")
    fi
    SEG_NOTE="segment ${SEGMENT}: epochs [${SEG_START}, ${SEG_END}) of ${NUM_EPOCHS}${CONTINUE_FROM:+, continuing from ${CONTINUE_FROM}}"
fi

# the job's GPUs and ports replace the config's; EXTRA_OVERRIDES come last
OVERRIDES=(
    "deployment.trainer.nproc_per_node=${#TRAINER_GPUS[@]}"
    "deployment.disaggregated.managed_local.trainer_cuda_visible_devices=${TRAINER_LIST}"
    "deployment.disaggregated.managed_local.capture_servers=${SERVERS}"
    "deployment.disaggregated.managed_local.mooncake.rpc_port=${MC_RPC}"
    "deployment.disaggregated.managed_local.mooncake.metadata_port=${MC_METADATA}"
    "deployment.disaggregated.managed_local.mooncake.metrics_port=${MC_METRICS}"
    "${SHARE_OVERRIDE[@]}"
    "${SEGMENT_OVERRIDES[@]}"
    "${EXTRA[@]}"
)
if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "[train] dry run: config=${CONFIG}, capture GPUs ${CAPTURE_CSV} (${CAPTURE_SERVERS} server(s), tp ${CAPTURE_TP}), trainer GPUs ${TRAINER_CSV}$([ "${SHARED}" = 1 ] && echo " (shared)")"
    [ -n "${SEG_NOTE}" ] && echo "[train] dry run: ${SEG_NOTE}"
    printf '[train]   %s\n' "${OVERRIDES[@]}"
    if [ -e "${CONTROL}" ] || [ -e "${STATE}" ]; then
        echo "[train] dry run: an earlier attempt's control/consumer state exists; a real run moves it aside first, so the plan may refuse it"
    fi
    unset TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR   # the per-job caches were never created
    exec python3 -m specforge.cli train --config "${CONFIG}" --plan "${OVERRIDES[@]}"
fi

mkdir -p "${OUT}/logs"
if [ -e "${CONTROL}" ] || [ -e "${STATE}" ]; then
    ASIDE="${OUT}/attempt-${JOBNUM}-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "${ASIDE}"
    [ -e "${CONTROL}" ] && mv "${CONTROL}" "${ASIDE}/"
    [ -e "${STATE}" ] && mv "${STATE}" "${ASIDE}/"
    echo "[train] moved an earlier attempt's control/consumer state to ${ASIDE}"
fi
LOG="${OUT}/logs/train_${JOBNUM}.log"
if [ "${CODE_SNAPSHOT:-1}" = 1 ]; then
    SNAP="${OUT}/code/${JOBNUM}"
    rm -rf "${SNAP}"
    mkdir -p "${SNAP}"
    cp -a specforge configs scripts "${SNAP}/" || { echo "[train] ERROR: cannot copy the code to ${SNAP}" >&2; exit 1; }
    cd "${SNAP}" || exit 1
    export PYTHONPATH="${SNAP}${PYTHONPATH:+:${PYTHONPATH}}"
fi

echo "[train] config=${CONFIG}; code $(pwd)"
[ -n "${SEG_NOTE}" ] && echo "[train] ${SEG_NOTE}"
[ "${#EXTRA[@]}" -gt 0 ] && echo "[train] extra overrides: ${EXTRA[*]}"
echo "[train] capture GPUs ${CAPTURE_CSV} (${CAPTURE_SERVERS} server(s), tp ${CAPTURE_TP}, ports from ${CAPTURE_PORT}, mem ${CAPTURE_MEM}), trainer GPUs ${TRAINER_CSV}$([ "${SHARED}" = 1 ] && echo " (shared)")"
echo "[train] mooncake ports rpc ${MC_RPC} metadata ${MC_METADATA} metrics ${MC_METRICS}"
echo "[train] SGLANG_MM_SKIP_COMPUTE_HASH=${SGLANG_MM_SKIP_COMPUTE_HASH}; log ${LOG}"
nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader
GPU_LOG="${OUT}/logs/gpu_${JOBNUM}.log"
(
    while true; do
        echo "$(date '+%F %T') $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';') | $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
        sleep 60
    done
) > "${GPU_LOG}" 2>&1 &
GPU_LOG_PID=$!

python3 -c "import specforge; print('[train] specforge from', specforge.__file__)"
python3 -m specforge.cli train --config "${CONFIG}" "${OVERRIDES[@]}" > "${LOG}" 2>&1
status=$?
kill "${GPU_LOG_PID}" 2> /dev/null
echo "[train] $(date '+%F %T') training exited ${status}"
if [ "${status}" = 0 ] && [ -n "${SEGMENT_EPOCHS:-}" ]; then
    # keep this segment's final checkpoint out of reach of max_checkpoints
    # rotation (hardlinks: no extra space) and record it for the next segment
    python3 - "${OUT}" "${RUN_ID}" "${SEG_DIR}" "${SEGMENT}" "${SEG_START}" "${SEG_END}" "${PBS_JOBID:-local}" <<'PY' || status=1
import json, os, re, subprocess, sys, time
import torch
out, run_id, seg_dir, seg, start, end, job = sys.argv[1:]
latest = os.path.realpath(os.path.join(out, f"{run_id}-latest"))
match = re.search(r"-step(\d+)$", latest)
if not match or not os.path.isfile(os.path.join(latest, "training_state.pt")):
    sys.exit(f"[train] ERROR: no final checkpoint behind {run_id}-latest ({latest})")
state = torch.load(os.path.join(latest, "training_state.pt"), map_location="cpu", weights_only=False, mmap=True)
if state.get("epoch") != 1 or state.get("epoch_samples"):
    sys.exit(f"[train] ERROR: {latest} records epoch={state.get('epoch')} "
             f"epoch_samples={state.get('epoch_samples')}, not a closed stream")
final = os.path.join(seg_dir, f"segment{seg}-final")
os.makedirs(seg_dir, exist_ok=True)
if os.path.exists(final):
    subprocess.run(["rm", "-rf", final], check=True)
subprocess.run(["cp", "-al", latest, final], check=True)
record = {"checkpoint": final, "source": latest, "global_step": int(match.group(1)),
          "epochs": [int(start), int(end)], "job": job,
          "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
tmp = os.path.join(seg_dir, f"segment{seg}.json.tmp")
with open(tmp, "w") as handle:
    json.dump(record, handle, indent=2)
os.replace(tmp, os.path.join(seg_dir, f"segment{seg}.json"))
print(f"[train] segment {seg} done at global_step {record['global_step']}: {final}")
PY
fi
tail -n 40 "${LOG}"
exit "${status}"
