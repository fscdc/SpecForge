#!/bin/bash
# Score the MMFlash training corpus for per-token visual dependency, one
# shard per GPU of this job (see scripts/score_visual_kl.py).
#
#   qsub -l select=1:ngpus=4 scripts/score_visual_kl_hpc.sh
#   qsub -l select=1:ngpus=1 -v MAX_ROWS=20 scripts/score_visual_kl_hpc.sh   # smoke test
#   bash scripts/score_visual_kl_hpc.sh [config.yaml]                        # interactive
#
# PASSING OPTIONS. qsub does NOT inherit the submitting shell's environment, so
# `NUM_SHARDS=8 qsub ...` sets nothing inside the job -- use `-v NAME=value`
# (comma-separated for several). Interactively with `bash`, the ordinary
# `NAME=value bash ...` form works, and the first positional argument is taken
# as the config.
#
# GPUS. By default every card the config claims under
# deployment.managed_local -- the capture servers' plus the trainer's -- runs
# one shard, because scoring happens before training and both roles are idle.
# GPUS=0,1,2,3 overrides; a config that claims none falls back to
# CUDA_VISIBLE_DEVICES and then to nvidia-smi.
#
# Each GPU runs shard SHARD_BASE + i of NUM_SHARDS. With two 4-GPU jobs use
# NUM_SHARDS=8 and SHARD_BASE=0 / 4:
#
#   qsub -l select=1:ngpus=4 -v NUM_SHARDS=8,SHARD_BASE=0 scripts/score_visual_kl_hpc.sh
#   qsub -l select=1:ngpus=4 -v NUM_SHARDS=8,SHARD_BASE=4 scripts/score_visual_kl_hpc.sh
#
# Re-submitting the same command resumes: each shard skips the ids already in
# its own file. Two jobs cannot be given the same shard by accident -- each
# takes an flock on it and the second one says so instead of interleaving
# appends into one file.
#
# PATHS. Nothing machine-specific is baked in here. The checkout is found from
# this script's own location (PBS_O_WORKDIR as a fallback, because qsub spools
# the script elsewhere); the DATA to score and the DIRECTORY to score into are
# both read out of --config, from data.train_data_path and
# data.visual_score_path -- the same two fields training reads. So on another
# server the only thing to change is the yaml, which has to change anyway
# because it holds the data path. OUTPUT_DIR=... still overrides.
#
#   CONFIG=scripts/mmtraining_configs/qwen3.5-4b-mmflash.yaml bash scripts/score_visual_kl_hpc.sh
#
# runs the same thing against the non-HPC yaml's /local_home2 paths.

#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N visual-kl
#PBS -q auto
#PBS -l select=1:ngpus=4
#PBS -l walltime=48:00:00

set -uo pipefail
export PYTHONUNBUFFERED=1

# --------------------------- environment -------------------------------------
CONDA_ENV=${CONDA_ENV:-specforge}
if ! python3 -c "import torch" >/dev/null 2>&1; then
    echo "[env] torch is not importable; activating conda env '${CONDA_ENV}'"
    conda_hook=""
    for candidate in \
        "${CONDA_EXE:+$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh}" \
        "${CONDA_PREFIX:+${CONDA_PREFIX}/etc/profile.d/conda.sh}" \
        "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh"
    do
        if [ -n "${candidate}" ] && [ -f "${candidate}" ]; then
            conda_hook="${candidate}"
            break
        fi
    done
    if [ -n "${conda_hook}" ]; then
        # shellcheck disable=SC1090
        . "${conda_hook}"
    elif [ -f "${HOME}/.bashrc" ]; then
        # shellcheck disable=SC1090
        . "${HOME}/.bashrc"
    fi
    conda activate "${CONDA_ENV}" >/dev/null 2>&1 || true
fi
if ! python3 -c "import torch" >/dev/null 2>&1; then
    echo "ERROR: torch is still not importable after trying to activate '${CONDA_ENV}'." >&2
    exit 1
fi
echo "[env] python3 $(python3 -c 'import sys,torch;print(sys.version.split()[0], "torch", torch.__version__)')"
echo "[env] CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"
# asked of torch rather than assembled from a hard-coded python version, which
# would silently produce a non-existent path on an env built on another minor
_TORCH_LIB="$(python3 -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null)"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+${CONDA_PREFIX}/lib:}${_TORCH_LIB}:${LD_LIBRARY_PATH:-}"

# ----------------------------- CONFIG ---------------------------------------
SCORER=scripts/score_visual_kl.py
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/${SCORER}" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
if [ ! -f "${PROJECT_DIR:-}/${SCORER}" ]; then
    echo "ERROR: cannot locate the SpecForge checkout (looked for ${SCORER})." >&2
    exit 1
fi
cd "${PROJECT_DIR}"

# first positional argument wins, then $CONFIG, then the default
CONFIG=${1:-${CONFIG:-scripts/mmtraining_configs/qwen3.5-4b-mmflash_hpc.yaml}}
MAX_ROWS=${MAX_ROWS:-}
DTYPE=${DTYPE:-bfloat16}

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: config ${CONFIG} not found under ${PROJECT_DIR}" >&2
    exit 1
fi

# Where to write, and what to read, both come from the CONFIG rather than from
# constants baked in here: data.visual_score_path is the same field training
# reads, so scoring cannot land somewhere the trainer will not look, and moving
# this to another machine means editing the yaml -- which has to change anyway,
# because it holds the data path -- and nothing else.
_CFG=$(python3 - "${CONFIG}" <<'READCFG'
import sys, yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
data = cfg.get("data") or {}
print(data.get("visual_score_path") or "")
print(data.get("train_data_path") or "")

# Every GPU this config claims on the box: the capture servers' and the
# trainer's, in that order, deduplicated. Scoring runs before training, so both
# roles are idle and the whole allocation is free to use.
managed = (
    ((cfg.get("deployment") or {}).get("disaggregated") or {}).get("managed_local")
) or {}
gpus, seen = [], set()
for server in managed.get("capture_servers") or []:
    for gpu in (server or {}).get("cuda_visible_devices") or []:
        if str(gpu) not in seen:
            seen.add(str(gpu)); gpus.append(str(gpu))
for gpu in managed.get("trainer_cuda_visible_devices") or []:
    if str(gpu) not in seen:
        seen.add(str(gpu)); gpus.append(str(gpu))
print(",".join(gpus))
READCFG
) || { echo "ERROR: could not read ${CONFIG}" >&2; exit 1; }
CFG_OUTPUT_DIR=$(printf '%s\n' "${_CFG}" | sed -n '1p')
TRAIN_DATA=$(printf '%s\n' "${_CFG}" | sed -n '2p')
CFG_GPUS=$(printf '%s\n' "${_CFG}" | sed -n '3p')

OUTPUT_DIR=${OUTPUT_DIR:-${CFG_OUTPUT_DIR}}
if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: ${CONFIG} sets no data.visual_score_path, so there is no" >&2
    echo "       default place to score into. Set it in the yaml (recommended," >&2
    echo "       training reads the same field) or pass OUTPUT_DIR=..." >&2
    exit 1
fi
if [ -z "${TRAIN_DATA}" ] || [ ! -f "${TRAIN_DATA}" ]; then
    echo "ERROR: data.train_data_path in ${CONFIG} is empty or does not exist on" >&2
    echo "       this machine: '${TRAIN_DATA}'" >&2
    echo "       Point the yaml at this server's copy of the regen jsonl." >&2
    exit 1
fi

# only feeds the ETA print; counting a 3 GB jsonl costs a few seconds
if [ -z "${EXPECTED_ROWS:-}" ]; then
    EXPECTED_ROWS=$(wc -l < "${TRAIN_DATA}" 2>/dev/null || echo 0)
fi

# Which GPUs to use, in order of precedence:
#   1. GPUS=...            explicit override
#   2. the config          every card deployment.managed_local claims -- the
#                          capture servers' plus the trainer's. Scoring runs
#                          before training, so both roles are idle and the whole
#                          allocation is free; this also keeps the pre-pass off
#                          cards that belong to someone else on a shared box.
#   3. CUDA_VISIBLE_DEVICES / nvidia-smi   when the config claims none
#                          (colocated mode, or an externally managed deployment)
if [ -n "${GPUS_OVERRIDE:-${GPUS:-}}" ]; then
    IFS=',' read -r -a GPUS <<< "${GPUS_OVERRIDE:-${GPUS}}"
    GPU_SOURCE="GPUS override"
elif [ -n "${CFG_GPUS}" ]; then
    IFS=',' read -r -a GPUS <<< "${CFG_GPUS}"
    GPU_SOURCE="${CONFIG} deployment.managed_local"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    GPU_SOURCE="CUDA_VISIBLE_DEVICES (config claims no GPUs)"
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    GPU_SOURCE="nvidia-smi (config claims no GPUs)"
fi

# The config names cards by index. If the job was handed a different set --
# PBS on this cluster passes UUIDs -- those indices mean something else, so say
# so rather than quietly scoring on cards this job may not even own.
if [ -n "${CFG_GPUS}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ] \
   && [ "${CFG_GPUS}" != "${CUDA_VISIBLE_DEVICES}" ]; then
    _job_count=$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
    _cfg_count=$(printf '%s' "${CFG_GPUS}" | awk -F, '{print NF}')
    echo "[warn] config asks for GPUs [${CFG_GPUS}] but this job was given" >&2
    echo "[warn]   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    echo "[warn] using the config's (${_cfg_count} card(s), job has ${_job_count})." >&2
    echo "[warn] Pass GPUS=\"${CUDA_VISIBLE_DEVICES}\" to use the job's instead." >&2
fi
LOCAL=${#GPUS[@]}
if [ "${LOCAL}" -eq 0 ]; then
    echo "ERROR: no GPU to score on." >&2
    echo "       ${CONFIG} names none under deployment.managed_local, and neither" >&2
    echo "       CUDA_VISIBLE_DEVICES nor nvidia-smi found any. Submit with" >&2
    echo "       -l select=1:ngpus=N, or pass GPUS=0,1,2,3." >&2
    exit 1
fi
NUM_SHARDS=${NUM_SHARDS:-${LOCAL}}
if [ -n "${PBS_ARRAY_INDEX:-}" ] && [ -z "${SHARD_BASE:-}" ]; then
    SHARD_BASE=$(( PBS_ARRAY_INDEX * LOCAL ))
fi
SHARD_BASE=${SHARD_BASE:-0}
if [ $(( SHARD_BASE + LOCAL )) -gt "${NUM_SHARDS}" ]; then
    echo "ERROR: SHARD_BASE(${SHARD_BASE}) + local GPUs(${LOCAL}) exceeds NUM_SHARDS(${NUM_SHARDS})" >&2
    exit 1
fi

if ! mkdir -p "${OUTPUT_DIR}/logs" 2>/dev/null; then
    echo "ERROR: cannot create ${OUTPUT_DIR}/logs -- check the path and permissions" >&2
    exit 1
fi
echo "[plan] project=${PROJECT_DIR}"
echo "[plan] data=${TRAIN_DATA} (${EXPECTED_ROWS} rows)"
echo "[plan] config=${CONFIG}"
echo "[plan] output=${OUTPUT_DIR}"
echo "[plan] gpus=${GPUS[*]}  (from ${GPU_SOURCE})"
echo "[plan] local_shards=${LOCAL} shard_base=${SHARD_BASE} num_shards=${NUM_SHARDS} max_rows=${MAX_ROWS:-all}"

# One shard must never be written by two processes: the appends would interleave
# into torn lines. An flock is enough and needs no cleanup -- the kernel drops it
# when the holder dies, so a killed job still resumes.
LOCKED=1
command -v flock >/dev/null 2>&1 || {
    LOCKED=0
    echo "[warn] flock not found; concurrent jobs on the same shard are NOT prevented" >&2
}

PIDS=()
for i in "${!GPUS[@]}"; do
    shard=$(( SHARD_BASE + i ))
    tag="shard$(printf '%03d' "${shard}")"
    log="${OUTPUT_DIR}/logs/${tag}.log"
    extra=()
    if [ -n "${MAX_ROWS}" ]; then extra+=(--max-rows "${MAX_ROWS}"); fi
    echo "[launch] shard ${shard}/${NUM_SHARDS} on GPU ${GPUS[$i]} -> ${log}"
    if [ "${LOCKED}" = "1" ]; then
        flock -n -E 99 "${OUTPUT_DIR}/logs/${tag}.lock" \
            env CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python3 "${SCORER}" \
            --config "${CONFIG}" \
            --output-dir "${OUTPUT_DIR}" \
            --shard-index "${shard}" \
            --num-shards "${NUM_SHARDS}" \
            --dtype "${DTYPE}" \
            --expected-rows "${EXPECTED_ROWS}" \
            "${extra[@]}" \
            > "${log}" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python3 "${SCORER}" \
            --config "${CONFIG}" \
            --output-dir "${OUTPUT_DIR}" \
            --shard-index "${shard}" \
            --num-shards "${NUM_SHARDS}" \
            --dtype "${DTYPE}" \
            --expected-rows "${EXPECTED_ROWS}" \
            "${extra[@]}" \
            > "${log}" 2>&1 &
    fi
    PIDS+=($!)
done

status=0
for i in "${!PIDS[@]}"; do
    shard=$(( SHARD_BASE + i ))
    wait "${PIDS[$i]}"
    rc=$?
    if [ "${rc}" -eq 0 ]; then
        echo "[done] shard ${shard} finished"
    elif [ "${rc}" -eq 99 ]; then
        echo "[done] shard ${shard} SKIPPED: another job already holds it." >&2
        echo "       Two submissions are covering the same shards -- check that" >&2
        echo "       NUM_SHARDS/SHARD_BASE were passed with 'qsub -v', not as" >&2
        echo "       shell variables (qsub does not inherit them)." >&2
        status=1
    else
        echo "[done] shard ${shard} FAILED rc=${rc} (see ${log%/*}/shard$(printf '%03d' "${shard}").log)" >&2
        status=1
    fi
done
echo "[done] tail of each log:"
for f in "${OUTPUT_DIR}"/logs/shard*.log; do
    echo "--- ${f}"; tail -n 2 "${f}"
done
exit "${status}"
