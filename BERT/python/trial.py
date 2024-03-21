from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import torch
import os
import json
# import numpy as np

with open("./config.json") as j:
    config = json.load(j)

if not os.path.exists("../infer"):
    os.mkdir("../infer")
# 存储推理模型参数的路径
if not os.path.exists("../infer/models"):
    os.mkdir("../infer/models")

# 加载分词器和模型
checkpoint = config["Checkpoint"]
tokenizer = BertTokenizer.from_pretrained(checkpoint)
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.eval()

# 加载数据集并进行编码
dataset = load_dataset("glue", "sst2")
encoded_dataset = dataset.map(lambda examples: tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128), batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 准备数据加载器
eval_dataset = encoded_dataset["validation"]
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

# print( eval_dataset["label"].shape )
unique_labels, counts = torch.unique(eval_dataset["label"], return_counts=True)
# 输出唯一标签的数量，即标签种类的个数
print(f"Label num: {len(unique_labels)}")
print(f"Num of each label: {counts}")

