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
        self.device = config.device
        self.AttentionMask = config.AttentionMask
        self.max_len = config.max_len
        # 申明网络
        self.bert_emd = BertEmbeddings(vocab_size=self.vocab_size, max_len=self.max_len,
                                             hidden_size=self.hidden_size, device=self.device)
        self.bert_encoder = nn.ModuleList(
            Transformer(
                hidden_size=self.hidden_size,
                attention_head_num=self.attention_head_num,
                attention_head_size=self.attention_head_size,
                intermediate_size=self.intermediate_size).to(self.device)
            for _ in range(self.num_hidden_layers)
        )
        self.pooler = BertPooler(self.hidden_size)

    def gen_attention_masks(self, attention_mask):
        size = list(attention_mask.size())
        batch = size[0]
        max_len = size[1]
        process_attention_mask = torch.zeros(batch, max_len, max_len, requires_grad=False)
        true_len = torch.sum(attention_mask, dim=1)
        for i in range(batch):
            process_attention_mask[i, :true_len[i], :true_len[i]] = 1
        return process_attention_mask

    def load_local2target(self):
        local2target_emb = {
            'bert_emd.token_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
            'bert_emd.type_embeddings.weight': 'bert.embeddings.token_type_embeddings.weight',
            'bert_emd.pos_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
            'bert_emd.emb_normalization.weight': 'bert.embeddings.LayerNorm.gamma',
            'bert_emd.emb_normalization.bias': 'bert.embeddings.LayerNorm.beta'
        }
        local2target_encoder = {
            'bert_encoder.%s.multi_attention.q_dense.weight': 'bert.encoder.layer.%s.attention.self.query.weight',
            'bert_encoder.%s.multi_attention.q_dense.bias': 'bert.encoder.layer.%s.attention.self.query.bias',
            'bert_encoder.%s.multi_attention.k_dense.weight': 'bert.encoder.layer.%s.attention.self.key.weight',
            'bert_encoder.%s.multi_attention.k_dense.bias': 'bert.encoder.layer.%s.attention.self.key.bias',
            'bert_encoder.%s.multi_attention.v_dense.weight': 'bert.encoder.layer.%s.attention.self.value.weight',
            'bert_encoder.%s.multi_attention.v_dense.bias': 'bert.encoder.layer.%s.attention.self.value.bias',
            'bert_encoder.%s.multi_attention.o_dense.weight': 'bert.encoder.layer.%s.attention.output.dense.weight',
            'bert_encoder.%s.multi_attention.o_dense.bias': 'bert.encoder.layer.%s.attention.output.dense.bias',
            'bert_encoder.%s.attention_layernorm.weight': 'bert.encoder.layer.%s.attention.output.LayerNorm.gamma',
            'bert_encoder.%s.attention_layernorm.bias': 'bert.encoder.layer.%s.attention.output.LayerNorm.beta',
            'bert_encoder.%s.feedforward.dense1.weight': 'bert.encoder.layer.%s.intermediate.dense.weight',
            'bert_encoder.%s.feedforward.dense1.bias': 'bert.encoder.layer.%s.intermediate.dense.bias',
            'bert_encoder.%s.feedforward.dense2.weight': 'bert.encoder.layer.%s.output.dense.weight',
            'bert_encoder.%s.feedforward.dense2.bias': 'bert.encoder.layer.%s.output.dense.bias',
            'bert_encoder.%s.feedforward_layernorm.weight': 'bert.encoder.layer.%s.output.LayerNorm.gamma',
            'bert_encoder.%s.feedforward_layernorm.bias': 'bert.encoder.layer.%s.output.LayerNorm.beta',
        }
        local2target_pooler = {
            "pooler.dense.weight": "bert.pooler.dense.weight",
            "pooler.dense.bias": "bert.pooler.dense.bias",
        }
        return local2target_emb, local2target_encoder, local2target_pooler

    def load_pretrain(self, sen_length, path):
        local2target_emb, local2target_encoder, local2target_pooler = self.load_local2target()
        pretrain_model_dict = torch.load(path)
        if sen_length == 512:
            finetune_model_dict = self.state_dict()
            new_parameter_dict = {}
            # 加载embedding层参数
            for key in local2target_emb:
                local = key
                target = local2target_emb[key]
                new_parameter_dict[local] = pretrain_model_dict[target]
            # 加载encoder层参数
            for i in range(self.num_hidden_layers):
                for key in local2target_encoder:
                    local = key % i
                    target = local2target_encoder[key] % i
                    new_parameter_dict[local] = pretrain_model_dict[target]
            # 加载pooler层参数
            for key in local2target_pooler:
                local = key
                target = local2target_pooler[key]
                new_parameter_dict[local] = pretrain_model_dict[target]
            finetune_model_dict.update(new_parameter_dict)
            self.load_state_dict(finetune_model_dict)
        else:
            raise Exception('输入预训练模型的长度错误')

    def forward(self, input_token, segment_ids, attention_mask):
        embedding_x = self.bert_emd(input_token, segment_ids)
        if self.AttentionMask:
            attention_mask = self.gen_attention_masks(attention_mask).to(self.device)
        else:
            attention_mask = None
        feedforward_x = None
        for i in range(self.num_hidden_layers):
            if i == 0:
                feedforward_x = self.bert_encoder[i](embedding_x, attention_mask)
            else:
                feedforward_x = self.bert_encoder[i](feedforward_x, attention_mask)
        sequence_output = feedforward_x
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output
