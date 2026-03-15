from huggingface_hub import snapshot_download

# 下载整个数据集
snapshot_download(
    repo_id="openvla/modified_libero_rlds",
    repo_type="dataset",
    local_dir="./modified_libero_rlds",
    resume_download=True
)