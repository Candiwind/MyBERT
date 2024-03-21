
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


import warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset#https://github.com/huggingface/datasets

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[100])

print(raw_train_dataset.features)

from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)
print(tokenized_datasets["train"][0])

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]#取到所有的列
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}#不需要这些列
print([len(x) for x in samples["input_ids"]])#每一个样本的长度

batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})

from transformers import TrainingArguments
training_args = TrainingArguments("test-trainer")
print(training_args)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

import numpy as np
preds = np.argmax(predictions.predictions, axis=-1)

from datasets import load_metric
metric = load_metric("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)



training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

