import torch
import torch.nn as nn
from transformers import BertModel

class BertNER(nn.Module):
    def __init__(self, myconfig):
        super().__init__()
        # self.cfg = myconfigfig
        self.num_labels=myconfig.num_labels
        # self.bert = BertModel.from_pretrained(myconfig.pretrained_model_path)
        self.bert = BertModel.from_pretrained(myconfig.pretrained_model_name, cache_dir=myconfig.cache_dir)
        self.dropout = nn.Dropout(myconfig.dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size,self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(self.dropout(x))
        loss = None  #只需要推理
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))   #[B*L, C],[B*L]
        return {"loss": loss, "logits": logits}

