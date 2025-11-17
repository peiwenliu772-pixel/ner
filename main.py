# main.py
import json
import os
import random
import numpy as np
import torch
from torch.optim import AdamW
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from data_process import load_data
from model import BertNER
import time
from config_loader import load_config
from tqdm import tqdm
from spacy.training.iob_utils import iob_to_biluo, tags_to_entities


# -----------------------------
# 0) 随机数
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# 1) BIO → 实体抽取
# -----------------------------
def get_entities_from_bio(tags):
    """
    将 BIO 标签序列转为实体列表 [(entity_type, start_idx, end_idx), ...]
    end_idx 为闭区间
    """
    # 先把 BIO 转成 BILUO
    biluo_tags = iob_to_biluo(tags)
    # 用 spaCy 提取实体
    entities = tags_to_entities(biluo_tags)
    # spaCy 返回的 end 是开区间 → 改为闭区间
    return [(label, start, end - 1) for label, start, end in entities]

# -----------------------------
# 2) 评估
# -----------------------------
def getPRF(pred_list, true_list):
    TP, pre, true = 0, 0, 0
    for pred_entities, true_entities in zip(pred_list, true_list):
        TP += len(pred_entities & true_entities)
        pre += len(pred_entities)
        true += len(true_entities)
        precision = TP / pre if pre != 0 else 0
        recall = TP / true if true != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return precision, recall, f1


@torch.no_grad()
def evaluate(myconfig, model, loader):
    """
    返回字典：{"precision": ..., "recall": ..., "f1": ...}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id2label = myconfig.id2label
    model_was_training = model.training
    model.eval()

    pred_list, true_list = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=True):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
        preds = torch.argmax(logits, dim=-1)

        for pred_seq, true_seq in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            # 去掉 pad/ignore
            pred_tags = [id2label[p] for p, t in zip(pred_seq, true_seq) if t != -100]
            true_tags = [id2label[t] for t in true_seq if t != -100]

            pred_entities = set(get_entities_from_bio(pred_tags))
            true_entities = set(get_entities_from_bio(true_tags))
            pred_list.append(pred_entities)
            true_list.append(true_entities)
    precision, recall, f1 = getPRF(pred_list, true_list)

    if model_was_training:
        model.train()
    return {"precision": precision, "recall": recall, "f1": f1}


# -----------------------------
# 3) 训练一个 epoch
# -----------------------------
def train_one_epoch(myconfig, model, loader, optimizer, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_loss = 0.0
    start = time.time()
    for batch in tqdm(loader, desc="Training", leave=True):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
    print(f"单步耗时：{time.time() - start:.3f}秒")
    return total_loss / len(loader.dataset)


# -----------------------------
# 4) 主流程
# -----------------------------
def main():
    myconfig = load_config("./ner_config/msra_bert_wwm.json")
    print(f"当前数据集: {myconfig.dataset}")
    print(f"加载模型:{myconfig.pretrained_model_name}")

    set_seed(myconfig.seed)
    myconfig.output_dir = os.path.join("output", myconfig.dataset, myconfig.pretrained_model_name)
    os.makedirs(myconfig.output_dir, exist_ok=True)
    log_path = os.path.join(myconfig.output_dir, "train_log.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizerFast.from_pretrained(myconfig.pretrained_model_name, cache_dir=myconfig.cache_dir)
   

    # DataLoader
    
    train_loader, dev_loader, test_loader, label2id, id2label = load_data(myconfig, tokenizer)
    print(f"训练集样本数：{len(train_loader.dataset)}")
    print(f"单epoch训练步数:{len(train_loader)}")
    # 模型
    myconfig.label2id = label2id
    myconfig.id2label = id2label
    myconfig.num_labels = len(label2id)
    model = BertNER(myconfig).to(device)

    # 优化器   权重衰减
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": myconfig.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=myconfig.learning_rate)

    # 线性 warmup + 线性下降（按 step）
    total_steps = len(train_loader) * myconfig.epochs
    warmup_steps = int(total_steps * myconfig.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 训练 & 验证（保存 dev 最优）
    best = -1.0
    for ep in range(1, myconfig.epochs + 1):
        train_loss = train_one_epoch(myconfig, model, train_loader, optimizer, scheduler)
        dev_metrics = evaluate(myconfig, model, dev_loader)
        main_score = dev_metrics.get("f1", 0.0)

        print(f"[Epoch {ep}] train_loss={train_loss:.4f} dev={dev_metrics}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Epoch {ep} | train_loss={train_loss:.4f} | dev={dev_metrics}\n")
        if main_score > best:
            best = main_score
            torch.save(model.state_dict(), os.path.join(myconfig.output_dir, "pytorch_model.bin"))
            tokenizer.save_pretrained(myconfig.output_dir)
            # 保存 myconfig 便于之后从目录加载
            model.bert.config.save_pretrained(myconfig.output_dir)
            print(f" save best to {myconfig.output_dir} (score={best:.4f})")

    # 测试：加载最佳
    model.load_state_dict(torch.load(os.path.join(myconfig.output_dir, "pytorch_model.bin"), map_location=device))
    test_metrics = evaluate(myconfig, model, test_loader)
    print(f"测试结果:{test_metrics}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"测试结果:{test_metrics}\n")

if __name__ == "__main__":
    main()
