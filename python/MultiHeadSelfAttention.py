import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 attention_head_num,
                 attention_head_size,
                 dropout_prob=0.1
                 ):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention_head_num = attention_head_num
        self.attention_head_size = attention_head_size
        self.out_dim = attention_head_num * attention_head_size
        self.q_dense = nn.Linear(self.out_dim, self.out_dim)
        self.k_dense = nn.Linear(self.out_dim, self.out_dim)
        self.v_dense = nn.Linear(self.out_dim, self.out_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_prob)
        self.o_dense = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, x, attention_mask):
        qx = x
        kx = x
        vx = x
        q = self.q_dense(qx)
        k = self.k_dense(kx)
        v = self.v_dense(vx)
        # 1. batch_size*seq_len*embedding_size -> batch_size*seq_len*head*head_size
        # 2. batch_size*seq_len*head*head_size -> (.T)batch_size*head*seq_len*head_size
        shape = list(x.size())
        batch_size = shape[0]
        seq_len = shape[1]
        q = q.view([batch_size, seq_len, self.attention_head_num, self.attention_head_size])
        q = q.transpose(1, 2)
        k = k.view([batch_size, seq_len, self.attention_head_num, self.attention_head_size])
        k = k.transpose(1, 2)
        v = v.view([batch_size, seq_len, self.attention_head_num, self.attention_head_size])
        v = v.transpose(1, 2)
        # q*k.T -> [batch_size, head, seq_len, seq_len]
        attention_scores = torch.matmul(q, k.transpose(2, 3))
        attention_scores = attention_scores / math.sqrt(float(self.attention_head_size))
        # In order to avoid the influence of padding '0's on softmax result, a negative value should be added to them
        # attention_mask: [batch_size, seq_len, seq_len]
        if attention_mask is not None:
            add_mask = (1.0 - attention_mask) * 1e5
            add_mask = add_mask[:, None, :, :]
            attention_scores -= add_mask
        attention_scores = self.softmax(attention_scores)
        attention_scores = self.dropout(attention_scores)
        attention_scores = torch.matmul(attention_scores, v)
        attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()
        attention_scores = attention_scores.view([batch_size, seq_len, self.out_dim])
        attention_scores = self.o_dense(attention_scores)
        return attention_scores






