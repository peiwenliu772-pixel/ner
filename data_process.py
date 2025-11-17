import torch
from torch.utils.data import Dataset, DataLoader
import json

def read_bio_file(file_path):
    """
    使用 split 方法读取 BIO 文件
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read().strip()  # 去掉文件首尾空行

    sentences, labels = [], []

    # 按双换行拆分句子
    raw_sentences = data.split("\n\n")

    for sent in raw_sentences:
        lines = sent.split("\n")  # 按行拆分
        cur_words = [line.split("\t")[0] for line in lines if "\t" in line]
        cur_labels = [line.split("\t")[1] for line in lines if "\t" in line]

        # 避免空句子加入
        if cur_words:
            sentences.append(cur_words)
            labels.append(cur_labels)

    return sentences, labels


# ===========================================================
# 2. 完整 BIO 标签集合
# ===========================================================
def build_label_map_bio(json_path):
     with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        label_list = data["labels"]
        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for i, label in enumerate(label_list)}
        return label2id, id2label


# ===========================================================
# 3. 对齐标签到 tokenizer tokens
# ===========================================================
def labels_with_tokens(words, labels, tokenizer, label2id, max_seq_len=256):
    encoding = tokenizer(
        words,
        is_split_into_words=True,  # 告诉分词器：words 已经是分好词/字的列表
        truncation=True,
        max_length=max_seq_len,
        return_attention_mask=True,
    )

    word_ids = encoding.word_ids()  # 每个 token 来自第几个原始词（None=特殊位）
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:  # [CLS]/[SEP]/[PAD]
            aligned_labels.append(-100)  # loss 忽略
        elif word_idx != previous_word_idx:  # 该词的第一个 subword
            tag = labels[word_idx]
            if tag not in label2id:  # 异常标签兜底为 'O'
                tag = 'O'
            aligned_labels.append(label2id[tag])
        else:  # 同一个词的后续 subword → 忽略
            aligned_labels.append(-100)
        previous_word_idx = word_idx

    encoding["labels"] = aligned_labels
    return encoding


# ===========================================================
# 4. NER Dataset
# ===========================================================
class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, label2id, max_seq_len=256):
        self.sentences, self.labels = read_bio_file(data_path)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "words": self.sentences[idx],
            "labels": self.labels[idx]
        }

    def collate_fn(self, batch):
        all_input_ids, all_attention_mask, all_labels = [], [], []
        # 逐样本对齐
        for item in batch:
            encoding = labels_with_tokens(
                item["words"], item["labels"], self.tokenizer, self.label2id, self.max_seq_len
            )
            all_input_ids.append(torch.tensor(encoding["input_ids"], dtype=torch.long))
            all_attention_mask.append(torch.tensor(encoding["attention_mask"], dtype=torch.long))
            all_labels.append(torch.tensor(encoding["labels"], dtype=torch.long))

        # pad 到 batch 最大长度
        input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(all_attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ===========================================================
# 5. DataLoader
# ===========================================================
def load_data(config, tokenizer):
     
    label2id, id2label = build_label_map_bio(config.labels_path)
    train_dataset = NERDataset(config.train_path, tokenizer, label2id, config.max_seq_len)
    dev_dataset = NERDataset(config.dev_path, tokenizer, label2id, config.max_seq_len)
    test_dataset = NERDataset(config.test_path, tokenizer, label2id, config.max_seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2,
        collate_fn=train_dataset.collate_fn)

    dev_loader = DataLoader(
        dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2,
        collate_fn=dev_dataset.collate_fn)

    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2,
        collate_fn=test_dataset.collate_fn)

    return train_loader, dev_loader, test_loader, label2id, id2label


# ===========================================================
# 6. 模块自检
# ===========================================================
if __name__ == "__main__":
    from transformers import AutoTokenizer


    class Config:
        train_path = "data/train.txt"
        dev_path = "data/dev.txt"
        test_path = "data/test.txt"
        labels_map_bio_path = "data/labels.json"
        max_seq_len = 256
        batch_size = 8


    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    train_loader, dev_loader, test_loader, label2id, id2label = load_data(Config, tokenizer)
    print(" 标签字典:", label2id)

    batch = next(iter(train_loader))
    print("input_ids:", batch["input_ids"].shape)
    print("labels:", batch["labels"].shape)