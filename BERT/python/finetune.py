from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
import os
import evaluate

# ================================ 涉及微调任务配置信息的一般都在这里 ====================================
checkpoint = "bert-base-uncased"
task = "imdb"
dataset = load_dataset("imdb")
# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(checkpoint)
# 加载数据集并进行编码
test_size = 0.3
splitted_datasets = dataset["train"].train_test_split(test_size=test_size)
def preprocess_function_batch(examples):
    return tokenizer(examples["text"], truncation=True)
splitted_datasets_encoded = splitted_datasets.map(preprocess_function_batch, batched=True)
dataset_test_encoded = dataset["test"].map(preprocess_function_batch, batched=True)
# 加载评估标准
metric = evaluate.load("accuracy")
# Trainer参数
model_output_dir = f"../finetune/checkpoints/{checkpoint}-finetuned-{task}"
model_save_dir = f"../finetune/models/{checkpoint}-finetuned-{task}"
args = TrainingArguments(
    output_dir=model_output_dir,
    evaluation_strategy="steps",
    eval_steps=50,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="tensorboard"
)
# ============================================= end ===================================================



if not os.path.exists("../finetune"):
    os.mkdir("../finetune")
# 存储微调后模型参数的路径
if not os.path.exists("../finetune/models"):
    os.mkdir("../finetune/models")
# 存储微调过程信息的路径
if not os.path.exists("../finetune/checkpoints"):
    os.mkdir("../finetune/checkpoints")


unique_labels = set(dataset["test"]["label"])
num_labels = len(unique_labels)
print(f"Label num: {num_labels}")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    results = metric.compute(predictions=predictions, references=labels)
    return results

def model_init():
    return BertForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=splitted_datasets_encoded["train"],
    eval_dataset=splitted_datasets_encoded["test"].shuffle(42).select(range(150)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.model.save_pretrained(model_save_dir)

test_predictions = trainer.predict(dataset_test_encoded)
test_predictions_argmax = np.argmax(test_predictions[0], axis=1)
test_references = np.array(dataset["test"]["label"])
metric_result = metric.compute(predictions=test_predictions_argmax, references=test_references)
print(metric_result)

