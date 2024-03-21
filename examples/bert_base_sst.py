from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import torch
import numpy as np

# 加载分词器和模型
#tokenizer = BertTokenizer.from_pretrained('assemblyai/bert-large-uncased-sst2')
#model = BertForSequenceClassification.from_pretrained('assemblyai/bert-large-uncased-sst2', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('EhsanAghazadeh/bert-based-uncased-sst2-e1')
model = BertForSequenceClassification.from_pretrained('EhsanAghazadeh/bert-based-uncased-sst2-e1', num_labels=2)
model.eval()

# 加载数据集并进行编码
dataset = load_dataset("glue", "sst2")
encoded_dataset = dataset.map(lambda examples: tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128), batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 准备数据加载器
eval_dataset = encoded_dataset["validation"]
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 初始化准确率计算
correct_predictions = 0
total_predictions = 0

# 评估模型性能
for batch in eval_dataloader:
    # 分别处理输入数据和标签
    inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
    labels = batch['label'].to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    # 更新正确预测的计数
    correct_predictions += (predictions == labels).sum().item()
    total_predictions += labels.size(0)

# 计算总准确率
accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy}")
