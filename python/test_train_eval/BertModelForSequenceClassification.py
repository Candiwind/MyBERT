import torch.nn as nn
from MyBertModel import *

class BertModelForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(BertModelForSequenceClassification, self).__init__()
        self.bert = MyBertModel(config)
        # 加载预训练参数
        if config.use_pretrained:
            self.bert.load_pretrain(config.max_len, config.bert_dir + 'pytorch_model.bin')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                ):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits