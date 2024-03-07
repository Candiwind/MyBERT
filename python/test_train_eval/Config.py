import torch

class Config:
    def __init__(self, vocab_size, bert_dir, hidden_dropout_prob, lr):
        self.vocab_size = vocab_size
        self.hidden_size = 768
        self.attention_head_num = 12
        self.attention_head_size = self.hidden_size // self.attention_head_num
        self.intermediate_size = 3072
        self.num_hidden_layers = 12
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.AttentionMask = True
        self.max_len = 512
        self.layer_norm_eps = 1e-12
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = hidden_dropout_prob
        # 以下的是训练的一些参数
        self.bert_dir = bert_dir
        self.train_max_len = 512  # 32
        self.batch_size = 4
        self.train_epochs = 5
        self.lr = lr
        self.num_labels = 2
        self.namefig = 'dropout%flr%f' % (hidden_dropout_prob, lr)
        self.output_dir = './checkpoints/' + self.namefig + "/"
        self.use_pretrained = True
        
        