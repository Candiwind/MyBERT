import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import f1_score, classification_report

from BertModelForSequenceClassification import *

class Trainer:
    def __init__(self, args, train_loader, dev_loader, test_loader):
        self.args = args
        self.device = args.device
        self.model = BertModelForSequenceClassification(args)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model.to(self.device)

    def load_ckp(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    def train(self):
        accumulation_steps = 8
        best_dev_micro_f1 = 0.0
        loss_list = []
        dev_list = []
        pth_temp = self.args.namefig + ".txt"
        f = open(pth_temp, 'w+')
        stopa = 0
        stopb = 0
        for epoch in range(self.args.train_epochs):
            # if stopa == 1:
            #     break
            for batch_idx, train_data in enumerate(self.train_loader):
                # f.flush()
                # stopb = stopb + 1
                # if stopb > 20:
                #     stopa = 1
                #     break
                self.model.train()
                token_ids = train_data['token_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                labels = train_data['labels'].to(self.device)
                train_outputs = self.model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(train_outputs, labels)
                loss = loss / accumulation_steps  # 梯度积累
                loss.backward()
                # msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                #         epoch + 1, batch_idx, len(self.train_loader), 100. *
                #         batch_idx / len(self.train_loader), loss.item()
                #     )
                # print(msg)
                if ((batch_idx + 1) % accumulation_steps) == 0:
                    # 每 8 次更新一下网络中的参数
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if ((batch_idx + 1) % accumulation_steps) == 1:
                    loss_list.append([loss.item(), batch_idx])
                    msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                        epoch + 1, batch_idx, len(self.train_loader), 100. *
                        batch_idx / len(self.train_loader), loss.item()
                    )
                    print(msg)
                    print(msg, file=f)
                if batch_idx == len(self.train_loader) - 1:
                    # 在每个 Epoch 的最后输出一下结果
                    dev_loss, dev_outputs, dev_targets = self.dev()
                    accuracy, precision, recall, f1 = self.get_metrics(dev_outputs, dev_targets)
                    dev_list.append([accuracy, precision, recall, f1])
                    msg = "loss：{:.4f} accuracy：{:.4f} precision：{:.4f} recall：{:.4f} f1：{:.4f}".format(
                        dev_loss, accuracy, precision, recall, f1)
                    print(msg)
                    print(msg, file=f)
                    if f1 > best_dev_micro_f1:
                        msg = "------------>保存当前最好的模型"
                        print(msg)
                        print(msg, file=f)
                        best_dev_micro_f1 = f1
                        checkpoint_path = os.path.join(self.args.output_dir, 'best.pt')
                        torch.save(self.model.state_dict(), checkpoint_path)
                
        pth_temp = self.args.namefig + "loss_list.pkl"
        with open(pth_temp, 'wb') as f:
            pickle.dump(loss_list, f)
        pth_temp = self.args.namefig + "dev_list.pkl"
        with open(pth_temp, 'wb') as f:
            pickle.dump(dev_list, f)
        f.close()
        checkpoint_path = os.path.join(self.args.output_dir, 'bestresult.pt')
        torch.save(self.model.state_dict(), checkpoint_path)


    def dev(self):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(self.dev_loader):
                token_ids = dev_data['token_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                labels = dev_data['labels'].to(self.device)
                outputs = self.model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(labels.cpu().detach().numpy().tolist())
        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path):
        model = self.model
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(self.test_loader):
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                labels = test_data['labels'].to(self.device)
                outputs = model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(labels.cpu().detach().numpy().tolist())
        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text, args):
        model = self.model
        checkpoint = os.path.join(args.output_dir, 'best.pt')
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            inputs = tokenizer.encode_plus(text=text,
                                           add_special_tokens=True,
                                           max_length=args.train_max_len,
                                           truncation='longest_first',
                                           padding="max_length",
                                           return_token_type_ids=True,
                                           return_attention_mask=True,
                                           return_tensors='pt')
            token_ids = inputs['input_ids'].to(self.device)
            attention_masks = inputs['attention_mask'].to(self.device)
            token_type_ids = inputs['token_type_ids'].to(self.device)
            outputs = model(token_ids, attention_masks, token_type_ids)
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
            if len(outputs) != 0:
                return outputs[0]
            else:
                return '未识别'

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        precision = precision_score(targets, outputs, average='micro')
        recall = precision_score(targets, outputs, average='micro')
        micro_f1 = f1_score(targets, outputs, average='micro')
        return accuracy, precision, recall, micro_f1

    def get_classification_report(self, outputs, targets):
        report = classification_report(targets, outputs)
        return report

