# # import os
# # import torch
# # from torch.utils.data import Dataset, DataLoader


# # class ProcessDataset(Dataset):
# #     def __init__(self, path, tokenizer, train_max_len):
# #         self.path = path
# #         self.tokenizer = tokenizer
# #         self.train_max_len = train_max_len
# #         self.features = self.get_features()
# #         self.nums = len(self.features)

# #     def get_features(self):
# #         features = []
# #         with open(self.path, 'r') as fp:
# #             lines = fp.read().strip().split('\n')
# #             for i, line in enumerate(lines):
# #                 line = line.split('\t')
# #                 text = line[0]
# #                 label = line[1]
# #                 inputs = self.tokenizer.encode_plus(
# #                     text=text,
# #                     max_length=self.train_max_len,
# #                     padding="max_length",
# #                     truncation="only_first",
# #                     return_attention_mask=True,
# #                     return_token_type_ids=True,
# #                 )
# #                 if i < 3:
# #                     print("input_ids:", str(inputs['input_ids']))
# #                     print("token_type_ids:", str(inputs['token_type_ids']))
# #                     print("attention_mask:", str(inputs['attention_mask']))
# #                     print("label:", label)
# #                 features.append(
# #                     (
# #                         inputs['input_ids'],
# #                         inputs['token_type_ids'],
# #                         inputs['attention_mask'],
# #                         int(label),
# #                     )
# #                 )
# #         return features

# #     def __len__(self):
# #         return self.nums

# #     def __getitem__(self, item):
# #         data = {
# #             "token_ids": torch.tensor(self.features[item][0]).long(),
# #             "token_type_ids": torch.tensor(self.features[item][1]).long(),
# #             "attention_masks": torch.tensor(self.features[item][2]).long(),
# #             "labels": torch.tensor(self.features[item][3]).long(),
# #         }
# #         return data


# # def readIMDB(path, seg="train"):
# #     pos_or_neg = ["pos", "neg"]
# #     data = []
# #     for label in pos_or_neg:
# #         files = os.listdir(os.path.join(path, seg, label))
# #         for file in files:
# #             with open(os.path.join(path, seg, label, file), "r", encoding="utf-8") as f:
# #                 review = f.read().replace("\n", "")
# #                 if label == "pos":
# #                     data.append([review, 1])
# #                 elif label == "neg":
# #                     data.append([review, 0])
# #     return data


# # if __name__ == "__main__":
# #     train_data = readIMDB("aclImdb")
# #     test_data = readIMDB("aclImdb", "test")
    
# import warnings
# warnings.filterwarnings("ignore")
# from datasets import load_dataset
# from transformers import AutoTokenizer
# from transformers import DataCollatorWithPadding
# from transformers import AutoModelForSequenceClassification




# def main():
    
#     # # Load datasets.
#     # raw_datasets = load_dataset("imdb")
    
#     # # Load tokenizer.
#     checkpoint = "bert-base-uncased"
#     # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
#     # # Tokenize. 
#     # # inputs = tokenizer("This is the first sentence.", "This is the second one.")
#     # # print(inputs)
#     # # print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
#     # def preprocess_function(examples):
#     #     return tokenizer(examples["text"], truncation=True)
#     # tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    
#     # # Remove unnecessary data keys.
#     # samples = tokenized_datasets["train"][:]
#     # samples = {k: v for k, v in samples.items() if k not in ["text"]}  # 不需要这些列
#     # # print([len(x) for x in samples["input_ids"]])  # 每一个样本的长度
    
#     # # Collate data. 
#     # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#     # batch = data_collator(samples)
    
#     # Load pretrained model.
#     # Similar to: model = resnet18(label_num) + model.load_state_dict(torch.load("../model/resnet-224x224.pth"))
#     model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
#     # print(dir(model))
#     # print(model.named_modules)
#     # print(model.named_parameters)  # Load parameters of each layer here.
#     # print(model.num_labels)
#     # print(model.num_parameters)
#     print(model.named_parameters())
    

from transformers import BertTokenizer
from Config import *
from MyBertModel import *


if __name__ == '__main__':
    BERT_PATH = './models/bert-base-uncased/'
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    config = Config(tokenizer.vocab_size)
    model = MyBertModel(config)
    model.load_pretrain(512, './models/bert-base-uncased/pytorch_model.bin')
    model.eval()
    model.to(config.device)
    for name, param in model.named_parameters():
        print(name)

