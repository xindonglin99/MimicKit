#!/bin/bash
set -e

SESSION="deepmimic_exps"
ARG_DIR="args/exps"
ARG_FILES=(${ARG_DIR}/*_args.txt)

# Kill old session if exists
tmux kill-session -t $SESSION 2>/dev/null || true

# Create new tmux session (detached)
tmux new-session -d -s $SESSION

# Activate conda environment inside tmux
tmux send-keys -t $SESSION "conda activate mimickit" C-m

# Send each experiment as its own command
for ARG_FILE in "${ARG_FILES[@]}"; do
    tmux send-keys -t $SESSION "python mimickit/run.py --arg_file ${ARG_FILE} --mode train --visualize false" C-m
    # Add a wait after each command to ensure sequential execution
    tmux send-keys -t $SESSION "wait" C-m
done

# Attach to tmux
tmux attach -t $SESSION
