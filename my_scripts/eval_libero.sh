#!/usr/bin/env bash

set -e

# ==============================
# 参数
# ==============================

POLICY_CHECKPOINT=$1
HOST=${2:-127.0.0.1}
PORT=${3:-8000}

if [ -z "$POLICY_CHECKPOINT" ]; then
    echo "Usage:"
    echo "bash run_eval_tmux.sh <checkpoint_path> [host] [port]"
    exit 1
fi

# ==============================
# 生成 session 名字
# ==============================

TIME_TAG=$(date +%m%d%H%M)
SESSION_NAME="eval_${TIME_TAG}"

echo "Creating tmux session: $SESSION_NAME"

# ==============================
# 创建 tmux session
# ==============================

tmux new-session -d -s $SESSION_NAME -n policy

# ==============================
# window 1: policy server
# ==============================

tmux send-keys -t $SESSION_NAME:policy "
cd $(pwd)

XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
uv run scripts/serve_policy.py policy:$POLICY_CHECKPOINT \
    --policy.config=pi05_droid \
    --policy.dir=$POLICY_CHECKPOINT \
    --env LIBERO \
    --host $HOST \
    --port $PORT
" C-m

# ==============================
# window 2: eval
# ==============================

tmux new-window -t $SESSION_NAME -n eval

tmux send-keys -t $SESSION_NAME:eval "
cd $(pwd)

source examples/libero/.venv/bin/activate

MUJOCO_GL=osmesa \
PYTHONPATH=\$PWD/third_party/libero \
python examples/libero/eval_with_log.py \
    --host $HOST \
    --port $PORT
" C-m

# ==============================
# attach
# ==============================

echo "Session created: $SESSION_NAME"
echo "Attach with:"
echo "tmux attach -t $SESSION_NAME"