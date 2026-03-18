from huggingface_hub import snapshot_download

# 下载整个 libero 数据集（包含所有子集，较大）
local_dir = "./libero_data"
snapshot_download(
    repo_id="physical-intelligence/libero",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False  # 直接复制文件（非软链接），便于后续使用
)