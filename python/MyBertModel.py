import torch
import torch.nn as nn

from BertEmbeddings import *
from BertPooler import *
from FeedForward import *
from GELU import *
from MultiHeadSelfAttention import *
from Transformer import *

class MyBertModel(nn.Module):
    def __init__(self, config):
        super(MyBertModel, self).__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.attention_head_num = config.attention_head_num
        self.attention_head_size = config.attention_head_size
        self.intermediate_size = config.intermediate_size
        self.num_hidden_layers = config.num_hidden_layers
        self.device = self.device
        self.AttentionMask = config.AttentionMask
        self.max_len = config.max_len
        # BertEmbeddings
        self.roberta_emd = BertEmbeddings(vocab_size=self.vocab_size, max_len=self.max_len,
                                            hidden_size=self.hidden_size, device=self.device
        )
        # BertEncoder
        self.transformer_blocks = nn.ModuleList(
            Transformer(
                hidden_size=self.hidden_size,
                attention_head_num=self.attention_head_num,
                attention_head_size=self.attention_head_size,
                intermediate_size=self.intermediate_size
            ).to(self.device)
            for _ in range(self.num_hidden_layers)
        )
        # BertPooler
        self.pooler = BertPooler(self.hidden_size)

    def forward(self, input_token, segment_ids, attention_mask):
        embedding_x = self.roberta_emd(input_token, segment_ids)
        if self.AttentionMask:
            attention_mask = self.gen_attention_masks(attention_mask).to(self.device)
        else:
            attention_mask = None
        feedforward_x = None
        for i in range(self.num_hidden_layers):
            if i == 0:
                feedforward_x = self.transformer_blocks[i](embedding_x, attention_mask)
            else:
                feedforward_x = self.transformer_blocks[i](feedforward_x, attention_mask)
        sequence_output = feedforward_x
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output



