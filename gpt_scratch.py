#Sindhi

import pandas as pd
sind = pd.read_csv("/content/only_articles.csv")
sind.head()

import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

try:
    text_df = pd.read_csv("/content/only_articles.csv", encoding='utf-8')
except UnicodeDecodeError:

    try:
        text_df = pd.read_csv("/content/only_articles.csv", encoding='utf-16')
    except:
        text_df = pd.read_csv("/content/only_articles.csv", encoding='windows-1256')


text = ''.join(text_df['article'].astype(str).tolist())  


chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")
print("Sample characters:", chars[:20])  

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    """Encode string to list of integers"""
    return [stoi[c] for c in s]

def decode(l):
    """Decode list of integers to string"""
    return ''.join([itos[i] for i in l])

test_str = "ڪتاب"  
if len(chars) > 0:  
    encoded = encode(test_str)
    decoded = decode(encoded)
    print(f"\nTest string: {test_str}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print("Decoding correct?", test_str == decoded)

data = torch.tensor(encode(text), dtype=torch.long)

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Hyperparameters
block_size = 8
batch_size = 32
max_iters = 1000
eval_interval = 100
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'




# Data loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10)
        for k in range(10):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Instantiate model
model = BigramLanguageModel(vocab_size)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate sample text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print("📝 Generated Sindhi text:\n", decode(generated))


