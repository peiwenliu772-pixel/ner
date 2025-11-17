
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="hfl/chinese-bert-wwm",
    cache_dir="./pre_models/bert-wwm" 
)
print(f"模型已下载到:{model_path}")
