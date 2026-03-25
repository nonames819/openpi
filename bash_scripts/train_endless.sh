#!/bin/bash

uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py pi05_libero --save_interval=500 --exp_name=pi05_libero_pytorch_8gpu_bs256 --resume --num_train_steps=7000

python $CHD/scripts/large_matrix.py