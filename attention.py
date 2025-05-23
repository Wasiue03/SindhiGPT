import pandas as pd

df = pd.read_csv('/content/only_articles.csv', encoding='iso-8859-1')

sindhi_articles = df['article'].head(1000)

with open('sindhi_articles.txt', 'w', encoding='utf-8') as f:
    for article in sindhi_articles:
        f.write(f"{article}\n")


import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys   = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries= nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys   = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries= query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys   = self.keys(keys)
        queries= self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


def create_vocab(text):
    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    return stoi, itos

# Load your Sindhi text
with open('sindhi_articles.txt', 'r', encoding='utf-8') as f:
    text = f.read()

stoi, itos = create_vocab(text)
