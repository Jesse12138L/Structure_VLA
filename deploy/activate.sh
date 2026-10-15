# Make the current shell use the packed environment. Source it, do not run it:
#
#     source deploy/activate.sh
#
# Every path is worked out from where this file lives, so nothing has to be edited when the
# project is put somewhere else. The layout it expects is the one deploy/setup.sh creates:
#
#     <root>/Structure_VLA/     this repository
#     <root>/lerobot-env/       the unpacked conda environment
#     <root>/huggingface/       the Hugging Face cache (holds the SmolVLM2 weights)

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$DEPLOY_DIR")")"

# `conda activate` is not used: in an environment unpacked by conda-pack it can return early
# and leave PATH untouched. Pointing PATH at the environment directly always works.
export CONDA_PREFIX="$ROOT/lerobot-env"
export PATH="$CONDA_PREFIX/bin:$PATH"

# All weights are local, so any attempt to reach the Hub is a mistake. Failing at once is
# better than hanging, or being rate limited by a mirror.
export HF_HOME="$ROOT/huggingface"
export HF_HUB_OFFLINE=1

export MUJOCO_GL=egl

# CUDA 12.4 images put their own cuDNN 9.1 first in LD_LIBRARY_PATH, but this PyTorch was
# built against 9.19 and refuses to start with the older one. Its bundled copy goes first.
CUDNN_DIR="$(ls -d "$CONDA_PREFIX"/lib/python3.*/site-packages/nvidia/cudnn/lib 2>/dev/null | head -n 1)"
if [ -n "$CUDNN_DIR" ]; then
    export LD_LIBRARY_PATH="$CUDNN_DIR:${LD_LIBRARY_PATH:-}"
fi

# A container reports every core of its host in nproc but may be limited by a CPU quota to a
# few of them. PyTorch sizes its thread pool from nproc, so on a 14-core quota over 128
# cores it ran about 20 times slower than with one thread (measured: 16 against 344
# samples/s in the data loader). One thread per process avoids that.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
