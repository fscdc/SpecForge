#!/bin/bash
# Download and extract every image source ShareGPT4V's annotations reference,
# laid out the way docs/Data.md expects:
#
#   <ROOT_DIR>/llava/llava_pretrain/images
#   <ROOT_DIR>/coco/train2017
#   <ROOT_DIR>/sam/images
#   <ROOT_DIR>/gqa/images
#   <ROOT_DIR>/ocr_vqa/images
#   <ROOT_DIR>/textvqa/train_images
#   <ROOT_DIR>/vg/VG_100K
#   <ROOT_DIR>/vg/VG_100K_2
#   <ROOT_DIR>/sharegpt4v/*.json
#   <ROOT_DIR>/share_textvqa/images
#   <ROOT_DIR>/web-celebrity/images
#   <ROOT_DIR>/web-landmark/images
#   <ROOT_DIR>/wikiart/images
#
# Sources with a plain, non-interactive URL are downloaded and extracted
# automatically, and are safe to re-run: any target directory that already
# has files is skipped. ShareGPT4V's own share_textvqa/web-celebrity/
# web-landmark/wikiart extras are single-file Google Drive links and are
# fetched with gdown. SAM downloads every shard listed in
# scripts/sam_download.txt (file_name<TAB>cdn_link rows, generated from
# https://ai.meta.com/datasets/segment-anything-downloads/) and tracks
# completed shards so a re-run only fetches what's missing. WebData and
# OCR-VQA are Google Drive *folders* (not single files, and OCR-VQA's is a
# download script rather than images) and cannot be fetched unattended;
# this script creates their target directories and prints what still needs
# to be fetched by hand.
#
# Usage:
#   bash scripts/download_sharegpt4v_images.sh <ROOT_DIR>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <ROOT_DIR>"
    exit 1
fi

ROOT_DIR=$(realpath -m "$1")
mkdir -p "${ROOT_DIR}"
echo "Root directory: ${ROOT_DIR}"
df -h "${ROOT_DIR}" | tail -1 | awk '{print "Free space on that filesystem: " $4}'
echo "COCO + GQA + TextVQA + VG + LAION-CC-SBU-558K together are roughly 60-70GB; make sure there is enough headroom before this runs unattended."
echo

MANUAL_STEPS=()

dir_is_populated() {
    [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]
}

# normalize_into_target <stage_dir> <target_dir>
#
# Merges <stage_dir>'s contents into <target_dir> (creating it if needed),
# then repeatedly flattens <target_dir> whenever its entire content is just
# one wrapping subfolder -- some archives wrap their files in more than one
# redundant layer (e.g. wikiart.zip turned out to contain wikiart/*.jpg
# instead of a flat file list, and blindly renaming that single folder onto
# an existing target_dir nested it instead of replacing it). Every step
# here is a content-merge move (mv -t into a directory guaranteed to
# already exist), never a bare `mv source target` rename, so it can't
# accidentally nest inside a target_dir that turns out to exist already.
# This is why every caller must pass the *leaf* directory it wants the
# images in (e.g. .../gqa/images), not the parent -- the function decides
# on its own whether a layer needs to be stripped.
normalize_into_target() {
    local stage_dir="$1" target_dir="$2"
    mkdir -p "${target_dir}"
    find "${stage_dir}" -mindepth 1 -maxdepth 1 ! -name "__MACOSX" -exec mv -t "${target_dir}" {} + 2>/dev/null
    rm -rf "${stage_dir}"

    while true; do
        local entries=()
        for e in "${target_dir}"/*; do
            [ -e "${e}" ] || continue
            entries+=("${e}")
        done
        if [ "${#entries[@]}" -ne 1 ] || [ ! -d "${entries[0]}" ]; then
            break
        fi
        local inner="${entries[0]}"
        find "${inner}" -mindepth 1 -maxdepth 1 -exec mv -t "${target_dir}" {} + 2>/dev/null
        rmdir "${inner}" 2>/dev/null
    done
}

# download_zip <name> <url> <target_leaf_dir>
download_zip() {
    local name="$1" url="$2" target_dir="$3"
    if dir_is_populated "${target_dir}"; then
        echo "[skip] ${name}: ${target_dir} already has files"
        return
    fi

    echo "[download] ${name} -> ${target_dir}"
    local parent_dir
    parent_dir=$(dirname "${target_dir}")
    mkdir -p "${parent_dir}"
    local stage_dir
    stage_dir=$(mktemp -d "${parent_dir}/.stage.XXXXXX")
    local zip_path="${stage_dir}/download.zip"

    if ! wget -c "${url}" -O "${zip_path}"; then
        echo "[error] ${name}: download failed from ${url}"
        rm -rf "${stage_dir}"
        MANUAL_STEPS+=("${name} -> ${target_dir} : download failed, retry ${url}")
        return
    fi
    if ! unzip -q "${zip_path}" -d "${stage_dir}"; then
        echo "[error] ${name}: failed to unzip (incomplete/corrupt download?)"
        rm -rf "${stage_dir}"
        MANUAL_STEPS+=("${name} -> ${target_dir} : unzip failed, retry ${url}")
        return
    fi
    rm -f "${zip_path}"

    normalize_into_target "${stage_dir}" "${target_dir}"

    if ! dir_is_populated "${target_dir}"; then
        echo "[warn] ${name}: ${target_dir} is empty after extraction -- inspect manually."
    fi
}

# download_gdown_zip <name> <google_drive_url> <target_leaf_dir>
#
# Same staging/normalization as download_zip, but fetches a single-file
# Google Drive share link with gdown instead of wget -- plain wget/curl
# can't get past Drive's HTML viewer page or its large-file virus-scan
# confirmation interstitial. The file ID is parsed out of the .../file/d/
# <ID>/... share URL and passed to gdown bare, rather than relying on
# gdown's --fuzzy flag, which older gdown releases don't have.
download_gdown_zip() {
    local name="$1" url="$2" target_dir="$3"
    if dir_is_populated "${target_dir}"; then
        echo "[skip] ${name}: ${target_dir} already has files"
        return
    fi
    if ! command -v gdown >/dev/null 2>&1; then
        echo "[manual] gdown not installed (pip install gdown) -- cannot fetch ${name} automatically."
        MANUAL_STEPS+=("${name} -> ${target_dir} : ${url}")
        return
    fi

    local file_id="${url#*/d/}"
    file_id="${file_id%%/*}"
    if [ "${file_id}" = "${url}" ] || [ -z "${file_id}" ]; then
        echo "[error] ${name}: could not parse a file ID out of ${url}"
        MANUAL_STEPS+=("${name} -> ${target_dir} : ${url} (unrecognized Drive URL format)")
        return
    fi

    echo "[download] ${name} -> ${target_dir}"
    local parent_dir
    parent_dir=$(dirname "${target_dir}")
    mkdir -p "${parent_dir}"
    local stage_dir
    stage_dir=$(mktemp -d "${parent_dir}/.stage.XXXXXX")
    local zip_path="${stage_dir}/download.zip"

    if ! gdown "${file_id}" -O "${zip_path}"; then
        echo "[error] ${name}: gdown download failed"
        rm -rf "${stage_dir}"
        MANUAL_STEPS+=("${name} -> ${target_dir} : ${url} (gdown failed, retry manually)")
        return
    fi
    if ! unzip -q "${zip_path}" -d "${stage_dir}"; then
        echo "[error] ${name}: failed to unzip (incomplete/corrupt download?)"
        rm -rf "${stage_dir}"
        MANUAL_STEPS+=("${name} -> ${target_dir} : ${url} (unzip failed, retry)")
        return
    fi
    rm -f "${zip_path}"

    normalize_into_target "${stage_dir}" "${target_dir}"

    if ! dir_is_populated "${target_dir}"; then
        echo "[warn] ${name}: ${target_dir} is empty after extraction -- inspect manually."
    fi
}

echo "=== COCO train2017 ==="
download_zip "COCO train2017" \
    "http://images.cocodataset.org/zips/train2017.zip" \
    "${ROOT_DIR}/coco/train2017"

echo "=== LAION-CC-SBU-558K (LLaVA-Pretrain) ==="
download_zip "LAION-CC-SBU-558K" \
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip" \
    "${ROOT_DIR}/llava/llava_pretrain/images"

echo "=== GQA ==="
download_zip "GQA" \
    "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" \
    "${ROOT_DIR}/gqa/images"

echo "=== TextVQA ==="
download_zip "TextVQA" \
    "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" \
    "${ROOT_DIR}/textvqa/train_images"

echo "=== VisualGenome (2 parts) ==="
download_zip "VisualGenome part1 (VG_100K)" \
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip" \
    "${ROOT_DIR}/vg/VG_100K"
download_zip "VisualGenome part2 (VG_100K_2)" \
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" \
    "${ROOT_DIR}/vg/VG_100K_2"

echo "=== ShareGPT4V extra image sets (Google Drive) ==="
download_gdown_zip "share_textvqa" \
    "https://drive.google.com/file/d/1f4v_3e1OJtyYqam1CEp6RenCNTU5_mG2/view?usp=drive_link" \
    "${ROOT_DIR}/share_textvqa/images"
download_gdown_zip "web-celebrity" \
    "https://drive.google.com/file/d/1-SB71C3j1mVg0kDDXwj2IWGEoBoRUD-J/view?usp=drive_link" \
    "${ROOT_DIR}/web-celebrity/images"
download_gdown_zip "web-landmark" \
    "https://drive.google.com/file/d/1JpJkN7ZMA50xAhMx9O-rVb5yLhfGm3_o/view?usp=drive_link" \
    "${ROOT_DIR}/web-landmark/images"
download_gdown_zip "wikiart" \
    "https://drive.google.com/file/d/1FxB2Nw-vWUcTUSI_dBpPIykb-uGYoEqV/view?usp=drive_link" \
    "${ROOT_DIR}/wikiart/images"

echo "=== SAM (sa_000000~sa_000050 only, from scripts/sam_download.txt) ==="
# ShareGPT4V's docs only use shards 000000~000050 (this is the subset that
# actually gets referenced by the "image" paths in its JSON annotations);
# the rest of the SAM release is not needed here and is skipped entirely.
SAM_MAX_SHARD=50
SAM_DIR="${ROOT_DIR}/sam/images"
SAM_LIST="$(dirname "$(realpath -m "$0")")/sam_download.txt"
mkdir -p "${SAM_DIR}"

# df_free_gb <path> -> free space on that path's filesystem, in whole GB
df_free_gb() {
    df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}

if [ ! -f "${SAM_LIST}" ]; then
    echo "[manual] ${SAM_LIST} not found -- cannot download SAM shards."
    MANUAL_STEPS+=("SAM sa_000000~sa_000050 -> ${SAM_DIR} : provide scripts/sam_download.txt (file_name<TAB>cdn_link rows from https://ai.meta.com/datasets/segment-anything-downloads/)")
else
    # Tracks which shards finished successfully so a re-run only fetches
    # what's still missing, instead of re-downloading everything.
    SAM_DONE_LOG="${ROOT_DIR}/sam/.downloaded_shards.txt"
    touch "${SAM_DONE_LOG}"
    total_shards=$(awk -F'\t' -v max="${SAM_MAX_SHARD}" \
        'NR>1 && $1 ~ /^sa_[0-9]+\.tar$/ { n=$1; gsub(/[^0-9]/,"",n); if ((n+0)<=max) c++ } END{print c+0}' \
        "${SAM_LIST}")
    echo "Restricted to sa_000000.tar~sa_00$(printf '%04d' "${SAM_MAX_SHARD}").tar (${total_shards} shards found in the list, ~11GB each). Downloads stop automatically if free space drops below 20GB; re-run later to resume."
    shard_num=0
    sam_stopped_early=0

    while IFS=$'\t' read -r fname url; do
        [ "${fname}" = "file_name" ] && continue
        # sam_download.txt also lists non-shard auxiliary files (e.g.
        # sa_images_ids.txt, sa_co_gold.tar); only sa_<digits>.tar rows are
        # actual numbered shards.
        if [[ ! "${fname}" =~ ^sa_([0-9]+)\.tar$ ]]; then
            continue
        fi

        shard_id="${BASH_REMATCH[1]}"
        if [ "$((10#${shard_id}))" -gt "${SAM_MAX_SHARD}" ]; then
            continue
        fi
        shard_num=$((shard_num + 1))

        if grep -qxF "${fname}" "${SAM_DONE_LOG}" 2>/dev/null; then
            echo "[skip] (${shard_num}/${total_shards}) ${fname} already extracted"
            continue
        fi

        free_gb=$(df_free_gb "${ROOT_DIR}")
        if [ -n "${free_gb}" ] && [ "${free_gb}" -lt 20 ]; then
            echo "[stop] Only ${free_gb}GB free -- stopping SAM downloads before filling the disk."
            MANUAL_STEPS+=("SAM full release -> ${SAM_DIR} : stopped at shard ${shard_num}/${total_shards} (${fname}) due to low disk space; free up space and re-run to resume")
            sam_stopped_early=1
            break
        fi

        echo "[download] (${shard_num}/${total_shards}) ${fname}"
        tar_path="${ROOT_DIR}/sam/${fname}"
        if ! wget -c "${url}" -O "${tar_path}"; then
            echo "[error] ${fname}: download failed"
            MANUAL_STEPS+=("SAM shard ${fname} -> ${SAM_DIR} : download failed -- the signed CDN link may have expired, regenerate scripts/sam_download.txt")
            rm -f "${tar_path}"
            continue
        fi
        if ! tar -xf "${tar_path}" -C "${SAM_DIR}"; then
            echo "[error] ${fname}: failed to extract (incomplete/corrupt download?)"
            MANUAL_STEPS+=("SAM shard ${fname} -> ${SAM_DIR} : extraction failed, retry")
            rm -f "${tar_path}"
            continue
        fi
        rm -f "${tar_path}"
        echo "${fname}" >> "${SAM_DONE_LOG}"
    done < "${SAM_LIST}"

    if [ "${sam_stopped_early}" -eq 0 ]; then
        echo "SAM: all ${total_shards} shards processed."
    fi
fi

echo "=== ShareGPT4V annotation JSONs ==="
SHAREGPT4V_DIR="${ROOT_DIR}/sharegpt4v"
mkdir -p "${SHAREGPT4V_DIR}"
for fname in \
    "share-captioner_coco_lcs_sam_1246k_1107.json" \
    "sharegpt4v_instruct_gpt4-vision_cap100k.json" \
    "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"; do
    if [ -s "${SHAREGPT4V_DIR}/${fname}" ]; then
        echo "[skip] ${fname} already downloaded"
        continue
    fi
    echo "[download] ${fname}"
    wget -c "https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/${fname}" \
        -O "${SHAREGPT4V_DIR}/${fname}"
done

echo "=== Sources with no direct/unattended download ==="
declare -A MANUAL_ONLY=(
    ["OCR-VQA"]="${ROOT_DIR}/ocr_vqa/images::https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_ (Drive folder with a download script; save every image as .jpg)"
    ["WebData (academic use only)"]="${ROOT_DIR}/webdata::https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax"
)
for name in "${!MANUAL_ONLY[@]}"; do
    entry="${MANUAL_ONLY[${name}]}"
    dir="${entry%%::*}"
    url="${entry#*::}"
    mkdir -p "${dir}"
    if dir_is_populated "${dir}"; then
        echo "[skip] ${name}: ${dir} already has files"
    else
        MANUAL_STEPS+=("${name} -> ${dir} : ${url}")
    fi
done

echo
echo "================ Summary ================"
if [ "${#MANUAL_STEPS[@]}" -eq 0 ]; then
    echo "Everything that could be automated is done. No manual steps pending."
else
    echo "Manual steps still needed:"
    for step in "${MANUAL_STEPS[@]}"; do
        echo "  - ${step}"
    done
fi
