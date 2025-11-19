from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="hfl/chinese-bert-wwm",
    local_dir="./pre_models/chinese-bert-wwm",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "pytorch_model.bin",   # PyTorch 权重
        "config.json",         # 模型结构
        "vocab.txt",           # tokenizer 词表
        "tokenizer.json"       # tokenizer 配置
    ],
    resume_download=True

)

print(f"模型已下载到: {model_path}")
# google-bert/bert-base-chinese