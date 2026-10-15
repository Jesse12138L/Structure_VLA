#!/usr/bin/env bash
# Set the project up on a new machine from the three archives deploy/pack.sh made.
#
# Put these next to the repository folder, i.e. in the folder that contains Structure_VLA/:
#
#     lerobot-env.tar.gz    the conda environment
#     lerobot-data.tar.gz   SmolVLM2 weights, LIBERO assets, the LeRobot dataset
#     lerobot-src.tar.gz    the lerobot source, which the environment points into
#
# then run, from anywhere:
#
#     bash Structure_VLA/deploy/setup.sh
#
# It can be run again: each step is skipped when it has already been done.

set -eu

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DEPLOY_DIR")"
ROOT="$(dirname "$REPO")"
ENV_DIR="$ROOT/lerobot-env"
SRC_DIR="$ROOT/lerobot-src"

echo "[setup] repository: $REPO"
echo "[setup] root:       $ROOT  (the archives are expected here)"


require_archive() {
    if [ ! -f "$ROOT/$1" ]; then
        echo "[setup] missing $ROOT/$1 - upload it there first" >&2
        exit 1
    fi
}


echo "[setup] 1/6 unpacking"
if [ ! -x "$ENV_DIR/bin/python" ]; then
    require_archive lerobot-env.tar.gz
    mkdir -p "$ENV_DIR"
    tar -xzf "$ROOT/lerobot-env.tar.gz" -C "$ENV_DIR"
fi
if [ ! -d "$SRC_DIR/src/lerobot" ]; then
    require_archive lerobot-src.tar.gz
    mkdir -p "$SRC_DIR"
    tar -xzf "$ROOT/lerobot-src.tar.gz" -C "$SRC_DIR"
fi
if [ ! -d "$ROOT/huggingface" ] || [ ! -d "$ROOT/libero" ]; then
    require_archive lerobot-data.tar.gz
    tar -xzf "$ROOT/lerobot-data.tar.gz" -C "$ROOT"
fi

echo "[setup] 2/6 putting the dataset inside the project"
# the configs say `dataset_root: data/lerobot_libero`, relative to the project root
if [ -d "$ROOT/lerobot_libero" ] && [ ! -d "$REPO/data/lerobot_libero" ]; then
    mv "$ROOT/lerobot_libero" "$REPO/data/lerobot_libero"
fi

echo "[setup] 3/6 pointing the environment at the lerobot source"
# The environment was packed on another machine, so this file holds that machine's path.
# It contains nothing but one directory to add to sys.path, so it is simply rewritten.
for python_dir in "$ENV_DIR"/lib/python3.*; do
    # the environment has a python3.1 that is only a link to python3.12; skip the link
    if [ -L "$python_dir" ]; then
        continue
    fi
    for pth in "$python_dir"/site-packages/__editable__.lerobot-*.pth; do
        if [ -f "$pth" ]; then
            echo "$SRC_DIR/src" > "$pth"
            echo "[setup]     $pth -> $SRC_DIR/src"
        fi
    done
done

echo "[setup] 4/6 linking the LIBERO assets"
# libero looks for its assets in ~/.cache/libero/assets and offers no way to change that
mkdir -p "$HOME/.cache"
if [ -e "$HOME/.cache/libero" ] && [ ! -L "$HOME/.cache/libero" ]; then
    echo "[setup]     $HOME/.cache/libero already exists as a real folder, leaving it alone"
else
    ln -sfn "$ROOT/libero" "$HOME/.cache/libero"
fi

echo "[setup] 5/6 answering libero's first-run question"
# The first `import libero` asks, on stdin, whether to store datasets in a custom place.
# A background run (nohup, setsid) cannot answer and would hang. LIBERO's own datasets are
# not used here, so the answer is no.
if [ ! -f "$HOME/.libero/config.yaml" ]; then
    echo N | "$ENV_DIR/bin/python" -c "import libero.libero" > /dev/null
fi

echo "[setup] 6/6 making every new shell use the environment"
LINE="source $DEPLOY_DIR/activate.sh"
if ! grep -qF "$LINE" "$HOME/.bashrc" 2>/dev/null; then
    echo "$LINE" >> "$HOME/.bashrc"
fi
source "$DEPLOY_DIR/activate.sh"


echo
echo "[check] each line below should say ok"
failures=0

check() {
    description="$1"
    shift
    if output="$("$@" 2>&1)"; then
        echo "  ok      $description"
    else
        echo "  FAILED  $description"
        echo "$output" | tail -n 3 | sed 's/^/            /'
        failures=$((failures + 1))
    fi
}

check "python is the packed one ($(which python))" test "$(which python)" = "$ENV_DIR/bin/python"
check "lerobot imports from $SRC_DIR" python -c \
    "import lerobot, sys; sys.exit(0 if lerobot.__file__.startswith('$SRC_DIR') else 1)"
check "torch sees the GPU and a matching cuDNN" python -c \
    "import torch; assert torch.cuda.is_available(); assert torch.backends.cudnn.version() >= 91900"
check "SmolVLM2 loads with no network" python -c \
    "from transformers import AutoProcessor; AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct')"
check "LIBERO assets are found" python -c \
    "from libero.libero import get_assets_path; import os; assert os.path.isdir(get_assets_path() + '/scenes')"
check "dataset is in the project" test -f "$REPO/data/lerobot_libero/meta/info.json"

# Training does not need the simulator, so a failure here is a warning and not a failure.
if python -c "import libero.libero.envs" > /dev/null 2>&1; then
    echo "  ok      the simulator can render (scripts.test will work here)"
else
    echo "  warning the simulator cannot render here; training is unaffected, so run"
    echo "          scripts.test on another machine"
fi

echo
if [ "$failures" -eq 0 ]; then
    echo "[setup] done. In a new shell, or after: source $DEPLOY_DIR/activate.sh"
    echo "        cd $REPO && python -m scripts.train --config configs/smolvla_goal.yaml"
else
    echo "[setup] $failures check(s) failed, see above" >&2
    exit 1
fi
