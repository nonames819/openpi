[[tool.uv.index]]
name = "sii"
url = "http://nexus.sii.shaipower.online/repository/pypi/simple"
default = true
# train

## fine-tune pi0.5 (一开始训练进度慢是正常现象)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment

# pytorch
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_pytorch.py pi05_libero_debug --exp_name pi05_libero_pytorch --overwrite

## debug
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=debug_pi05_libero 

## convert
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir checkpoints/pi05_libero/my_experiment/29999 \
    --config_name pi05_libero \
    --output_path checkpoints/pi05_libero_pytorch/my_experiment/29999

uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ./cache/openpi-assets/checkpoints/pi0_base \
    --config_name pi0_libero \
    --output_path ./cache/openpi-assets/checkpoints_pytorch/pi0_base

# eval 

## libero
```
# Run the simulation (pick this and set export MUJOCO_GL=osmesa)
source examples/libero/.venv/bin/activate
MUJOCO_GL=osmesa PYTHONPATH=$PWD/third_party/libero python examples/libero/eval_with_log.py

# Run the server (modify config in python file to select path)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 uv run scripts/serve_policy.py --env LIBERO



# below are official instructions
# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
MUJOCO_GL=osmesa PYTHONPATH=$PWD/third_party/libero python examples/libero/main.py

# Run the server
uv run scripts/serve_policy.py --env LIBERO

uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```