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

    def load_local2target(self):
        local2target_emb = {
            "roberta_emd.token_embeddings.weight": "bert.embeddings.word_embeddings.weight",
            "roberta_emd.position_embeddings.weight": "bert.embeddings.position_embeddings.weight",
            "roberta_emd.type_embeddings.weight": "bert.embeddings.token_type_embeddings.weight",
            "roberta_emd.emb_normalization.weight": "bert.embeddings.LayerNorm.gamma",
            "roberta_emd.emb_normalization.bias": "bert.embeddings.LayerNorm.beta"   # 
        }
        local2target_transformer = {
            "transformer_blocks.%s.multi_attention.q_dense.weight": "bert.encoder.layer.%s.attention.self.query.weight",
            "transformer_blocks.%s.multi_attention.q_dense.bias": "bert.encoder.layer.%s.attention.self.query.bias",
            "transformer_blocks.%s.multi_attention.k_dense.weight": "bert.encoder.layer.%s.attention.self.key.weight",
            "transformer_blocks.%s.multi_attention.k_dense.bias": "bert.encoder.layer.%s.attention.self.key.bias",
            "transformer_blocks.%s.multi_attention.v_dense.weight": "bert.encoder.layer.%s.attention.self.value.weight",
            "transformer_blocks.%s.multi_attention.v_dense.bias": "bert.encoder.layer.%s.attention.self.value.bias",
            "transformer_blocks.%s.multi_attention.o_dense.weight": "bert.encoder.layer.%s.attention.output.dense.weight",
            "transformer_blocks.%s.multi_attention.o_dense.weight": "bert.encoder.layer.%s.attention.output.dense.bias",
            "transformer_blocks.%s.attention_layernorm.weight": "bert.encoder.layer.%s.attention.output.LayerNorm.weight",
            "transformer_blocks.%s.attention_layernorm.bias": "bert.encoder.layer.%s.attention.output.LayerNorm.bias",
            "transformer_blocks.%s.feedforward.dense1.weight": "bert.encoder.layer.%s.intermediate.dense.weight",
            "transformer_blocks.%s.feedforward.dense1.bias": "bert.encoder.layer.%s.intermediate.dense.bias",
            "transformer_blocks.%s.feedforward.dense2.weight": "bert.encoder.layer.%s.output.dense.weight",
            "transformer_blocks.%s.feedforward.dense2.bias": "bert.encoder.layer.%s.output.dense.bias",
            "transformer_blocks.%s.feedforward_layernorm.weight": "bert.encoder.layer.%s.output.LayerNorm.weight",
            "transformer_blocks.%s.feedforward_layernorm.bias": "bert.encoder.layer.%s.output.LayerNorm.bias"
        }
        local2target_pooler = {
            "pooler.dense.weight": "bert.pooler.dense.weight",
            "pooler.dense.bias": "bert.pooler.dense.bias"
        }
        return local2target_emb, local2target_transformer, local2target_pooler

    def load_pretrain(self, sen_length, path):
        local2target_emb, local2target_transformer, local2target_pooler = self.get_local2target()
        pretrain_model_dict = torch.load(path)
        if sen_length == 512:
            finetune_model_dict = self.state_dict()
            new_parameter_dict = {}
            # Load embedding parameters
            for key in local2target_emb:
                local = key
                target = local2target_emb[key]
                new_parameter_dict[local] = pretrain_model_dict[target]
            # Load transformerblock parameters
            for i in range(self.num_hidden_layers):
                for key in local2target_transformer:
                    local = key % i
                    target = local2target_transformer[key] % i
                    new_parameter_dict[local] = pretrain_model_dict[target]
            # Load pooler parameters
            for key in local2target_pooler:
                local = key
                target = local2target_pooler[key]
                new_parameter_dict[local] = pretrain_model_dict[target]
            finetune_model_dict.update(new_parameter_dict)
            self.load_state_dict(finetune_model_dict)
        else:
            raise Exception("Input-len Error: Pretrained-len is 512. ")

