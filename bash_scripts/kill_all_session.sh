#!/bin/bash
# =========================
# 批量关闭所有 LIBERO tmux 会话
# =========================

echo "Listing all tmux sessions..."
tmux ls | grep '^libero_' >/tmp/libero_sessions.txt

if [ ! -s /tmp/libero_sessions.txt ]; then
    echo "No LIBERO sessions found."
    exit 0
fi

while IFS= read -r line; do
    # session 名字在冒号前
    SESSION_NAME=$(echo $line | cut -d: -f1)
    echo "Killing session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
done < /tmp/libero_sessions.txt

rm /tmp/libero_sessions.txt
echo "All LIBERO tmux sessions have been killed."