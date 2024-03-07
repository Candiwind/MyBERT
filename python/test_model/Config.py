import torch

class Config:
    def __init__(self, vocab_size):
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

