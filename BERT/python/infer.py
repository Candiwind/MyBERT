from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import torch
import os
import shutil
import evaluate



# ================================ 涉及推理任务配置信息的一般都在这里 ====================================
checkpoint = "EhsanAghazadeh/bert-based-uncased-sst2-e1"
dataset = load_dataset("glue", "sst2")
# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(checkpoint)
# 加载数据集并进行编码
def preprocess_function_batch(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)
encoded_dataset = dataset.map(preprocess_function_batch, batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
# 加载评估标准
metric = evaluate.load("accuracy")
# ============================================= end ===================================================



if not os.path.exists("../infer"):
    os.mkdir("../infer")
# 存储推理模型参数的路径
if not os.path.exists("../infer/models"):
    os.mkdir("../infer/models")



# 准备数据加载器
eval_dataset = encoded_dataset["validation"]
unique_labels, counts = torch.unique(eval_dataset["label"], return_counts=True)
num_labels = len(unique_labels)
# 输出唯一标签的数量，即标签种类的个数
print(f"Label num: {num_labels}")
print(f"Num of each label: {counts}")
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
model.eval()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

save_checkpoint = checkpoint.replace("/", "_")
model_save_path = f"../infer/models/{save_checkpoint}"
if os.path.exists(model_save_path):
    shutil.rmtree(model_save_path)
    os.mkdir(model_save_path)
model.save_pretrained(model_save_path)

# 初始化准确率计算
correct_predictions = 0
total_predictions = 0

# 评估模型性能
for batch in eval_dataloader:
    # 分别处理输入数据和标签
    inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
    labels = batch['label'].to(device)
    
    with torch.no_grad():
        test_predictions = model(**inputs)

    logits = test_predictions.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=labels)

# 计算总准确率
metric_result = metric.compute()
print(metric_result)

