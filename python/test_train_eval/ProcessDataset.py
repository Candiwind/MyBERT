import os
import torch
from torch.utils.data import Dataset

class ProcessDataset(Dataset):
    def __init__(self, data, tokenizer, train_max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.train_max_len = train_max_len
        self.feeatures = self.get_features()
        self.nums = len(self.feeatures)

    def get_features(self):
        features = []
        for review, label in self.data:
            inputs = self.tokenizer.encode_plus(
                text=review,
                max_length=self.train_max_len,
                padding="max_length",
                truncation="only_first",
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            features.append(
                (
                    inputs['input_ids'],
                    inputs['token_type_ids'],
                    inputs['attention_mask'],
                    int(label),
                )
            )
        return features

    def __len__(self):
        return self.nums

    def __getitem__(self, item):
        data = {
            "token_ids": torch.tensor(self.feeatures[item][0]).long(),
            "token_type_ids": torch.tensor(self.feeatures[item][1]).long(),
            "attention_masks": torch.tensor(self.feeatures[item][2]).long(),
            "labels": torch.tensor(self.feeatures[item][3]).long(),
        }
        return data

def readIMDB(path, seg='train'):
    pos_or_neg =['pos', 'neg']
    data = []
    for label in pos_or_neg:
        files = os.listdir(os.path. join(path, seg, label))
        for file in files:
            with open(os.path.join(path, seg, label, file),'r', encoding='utf-8') as rf:
                review=rf.read().replace('\n', '')
                if label =='pos':
                    data.append([review, 1])
                elif label == 'neg':
                    data.append([review, 0])
    return data


