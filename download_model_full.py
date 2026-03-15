from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="casperhansen/llama-3.3-70b-instruct-awq",
    local_dir="/mnt/ssd_combined/models/llama-3.3-70b-awq",
    local_dir_use_symlinks=False,
    resume_download=True,
    ignore_patterns=None,          # ничего не игнорируем
    max_workers=8                  # ускорит загрузку
)