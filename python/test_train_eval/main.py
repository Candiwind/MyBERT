from transformers import BertTokenizer
import pickle
from torch.utils.data import DataLoader
import random

from Config import *
from Trainer import *
from ProcessDataset import *


if __name__ == '__main__':
    random.seed(1)
    torch.manual_seed(1)
    
    BERT_PATH = './models/bert-base-uncased/'
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    train_max_len = 512
    datasetName = "aclImdb_v1"
    train_data = readIMDB(f"./data/{datasetName}")
    test_data = readIMDB(f"./data/{datasetName}", seg="test")
    train_dataset = ProcessDataset(train_data, tokenizer, train_max_len)
    test_set = ProcessDataset(test_data, tokenizer, train_max_len)
    train_set, check_set = torch.utils.data.random_split(train_dataset, [20000, 5000])
    
    with open(f'./data/{datasetName}/train_set_save.pkl', 'wb') as f:
        pickle.dump(train_set, f)
    with open(f'./data/{datasetName}/check_set_save.pkl', 'wb') as f:
        pickle.dump(check_set, f)
    with open(f'./data/{datasetName}/test_set_save.pkl', 'wb') as f:
        pickle.dump(test_set, f)
    
    # 加载字典、词集合
    # with open('./data/train_set_save.pkl', 'rb') as f:
    #     train_dataset = pickle.load(f)
    with open(f'./data/{datasetName}/check_set_save.pkl', 'rb') as f:
        check_dataset = pickle.load(f)
    with open(f'./data/{datasetName}/test_set_save.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    with open(f'./data/{datasetName}/train_set_save.pkl', 'rb') as f:
        train_dataset_temp = pickle.load(f)
    # 构建数据集和数据迭代器，设定 batch_size 大小为 4
    train_loader = DataLoader(dataset=train_dataset_temp,
                              batch_size=4,
                              shuffle=True)
    eval_loader = DataLoader(dataset=check_dataset,
                             batch_size=4,
                             shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=4,
                             shuffle=False)
    ## 调试超参数
    # test_parameter = [[0, 2e-5], [0.1, 2e-5], [0.5, 2e-5], [0.1, 2e-4], [0.1, 2e-6]]
    # for i in range(len(test_parameter)):
    #     hidden_dropout_prob, lr = test_parameter[i]
    #     config = Config(tokenizer.vocab_size, BERT_PATH, hidden_dropout_prob, lr)
    #     trainer = Trainer(config, train_loader, eval_loader, test_loader)
    #     trainer.train()
    ## 使用最优参数在Trainset上重新训练模型
    result_parameter = [[0.1, 5e-6]]
    hidden_dropout_prob, lr = result_parameter[0]
    config = Config(tokenizer.vocab_size, BERT_PATH, hidden_dropout_prob, lr)
    trainer = Trainer(config, train_loader, eval_loader, test_loader)
    trainer.train()
    # 在测试集上测试模型性能
    if not os.path.exists("./checkpoints/"):
        os.mkdir("./checkpoints/")
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    checkpoint_path = os.path.join(config.output_dir, 'bestresult.pt')
    total_loss, test_outputs, test_targets = trainer.test(checkpoint_path)
    report = trainer.get_classification_report(test_targets, test_outputs)
    print(report)
