import torch
import torch.nn as nn

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size, device, dropout_prob=0.1):
        self.device = device
        self.max_len = max_len
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.type_embeddings = nn.Embedding(2, hidden_size)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.emb_normalization = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input_token, segment_ids):
        token_embeddings = self.token_embeddings(input_token)
        type_embeddings = self.type_embeddings(segment_ids)
        position_ids = []
        input_count = list(input_token.size())[0]
        max_len = list(input_token.size())[1]
        for _ in range(input_count):
            tmp = [x for x in range(max_len)]
            position_ids.append(tmp)
        position_ids = torch.tensor(position_ids).to(self.device)
        position_embeddings = self.position_embeddings(position_ids)
        embedding_x = token_embeddings + type_embeddings + position_embeddings
        embedding_x = self.emb_normalization(embedding_x)
        embedding_x = self.emb_dropout(embedding_x)
        return embedding_x


