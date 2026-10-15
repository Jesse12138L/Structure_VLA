#!/usr/bin/env bash
# Pack everything a new machine needs, on the machine where the project already works.
#
#     bash deploy/pack.sh              # archives go to your home folder
#     bash deploy/pack.sh /some/dir    # or to a folder of your choice
#
# It writes three archives. Upload them into the folder that holds Structure_VLA/ on the new
# machine, then run deploy/setup.sh there:
#
#     lerobot-env.tar.gz    the conda environment
#     lerobot-data.tar.gz   SmolVLM2 weights, LIBERO assets, the LeRobot dataset
#     lerobot-src.tar.gz    the lerobot source, which the environment points into
#
# Nothing is installed or changed on this machine; the archives are only read from it.

set -euo pipefail

ENV_NAME="${ENV_NAME:-lerobot}"
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DEPLOY_DIR")"
OUT_DIR="${1:-$HOME}"

MODEL_DIR="$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
ASSETS_DIR="$HOME/.cache/libero/assets"
DATASET_DIR="$REPO/data/lerobot_libero"

echo "[pack] environment: $ENV_NAME"
echo "[pack] output:      $OUT_DIR"

# Fail now, not after twenty minutes of compressing.
for needed in "$MODEL_DIR" "$ASSETS_DIR" "$DATASET_DIR/meta/info.json"; do
    if [ ! -e "$needed" ]; then
        echo "[pack] not found: $needed" >&2
        exit 1
    fi
done
mkdir -p "$OUT_DIR"


echo "[pack] 1/3 the environment (the slow one, a few minutes)"
# --ignore-editable-packages: lerobot is installed editable, which conda-pack refuses by
#   default. It travels in its own archive below instead.
# --ignore-missing-files: setuptools and packaging were upgraded with pip over the conda
#   versions, and conda-pack stops on files conda no longer finds.
conda pack -n "$ENV_NAME" -f -j -1 \
    --ignore-editable-packages --ignore-missing-files \
    -o "$OUT_DIR/lerobot-env.tar.gz"


echo "[pack] 2/3 the lerobot source"
# Ask the environment itself where lerobot is, rather than assuming a folder.
# `conda run` adds a blank line after the output, and importing can print warnings before
# it, so pick the line that is the file path instead of taking the first or the last.
LEROBOT_FILE="$(conda run -n "$ENV_NAME" python -c "import lerobot; print(lerobot.__file__)" \
    | grep '__init__.py$' | tail -n 1)"
LEROBOT_ROOT="$(dirname "$(dirname "$(dirname "$LEROBOT_FILE")")")"
if [ ! -d "$LEROBOT_ROOT/src/lerobot" ]; then
    echo "[pack] expected the lerobot source at $LEROBOT_ROOT/src/lerobot, but lerobot was" >&2
    echo "       imported from '$LEROBOT_FILE'. Is it an editable install?" >&2
    exit 1
fi
echo "[pack]     $LEROBOT_ROOT"
tar -czf "$OUT_DIR/lerobot-src.tar.gz" -C "$LEROBOT_ROOT" src pyproject.toml


echo "[pack] 3/3 the weights, the LIBERO assets and the dataset"
# --dereference: inside the Hugging Face cache, snapshots/ holds only links into blobs/.
#   Archived as links they would arrive pointing at nothing, so the real files are stored.
# --exclude blobs: after that, blobs/ would hold the same 2 GB a second time.
# `-cf` with `-I 'gzip -1'` rather than `-czf`, which tar rejects as two compressors.
# gzip -1: the dataset is already compressed video, so a heavier setting only costs time.
tar -cf "$OUT_DIR/lerobot-data.tar.gz" -I 'gzip -1' --dereference \
    --exclude='huggingface/hub/models--*/blobs' \
    -C "$HOME/.cache" \
        huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct \
        huggingface/hub/version.txt \
        huggingface/hub/CACHEDIR.TAG \
        libero \
    -C "$REPO/data" lerobot_libero


echo "[pack] checking the archives are not truncated"
for archive in lerobot-env lerobot-src lerobot-data; do
    gzip -t "$OUT_DIR/$archive.tar.gz"
done
ls -lh "$OUT_DIR"/lerobot-env.tar.gz "$OUT_DIR"/lerobot-src.tar.gz "$OUT_DIR"/lerobot-data.tar.gz

echo
echo "[pack] done. On the new machine:"
echo "         1. git clone the repository, then upload the three archives into the folder"
echo "            that contains Structure_VLA/"
echo "         2. bash Structure_VLA/deploy/setup.sh"
