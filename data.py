#%%
import torch
import numpy as np 
import torch.nn as nn
from torch.nn import functional as F 

data_file =  "./input.txt"
batch_size = 4
block_size = 8 
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'mps'
eval_iters = 200


def read_data(data_file = data_file):
    with open(data_file, "r") as f:
        lines = f.read()
    print(f"lenth of data in characters {len(lines)}")
    return lines

def text_decoder(text):
    token_decoder = {i: ch for i, ch in enumerate(CHARS)}
    decode = lambda x: ''.join([token_decoder[c] for c in x])
    return decode(text)

def text_encoder(text):
    token_encoder = {ch: i for i, ch in enumerate(CHARS)}
    encode = lambda x: [token_encoder[c] for c in x]
    return encode(text)

def train_val_split(text, train_perc = 0.9):
    data = torch.tensor(text_encoder(text), dtype=torch.long)
    n = int(train_perc*len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(split: str):
    data = TRAIN_DATA if split == "train" else VAL_DATA
    # randomly sample integers from size of data and return tensor of size batch_size 
    idx = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y

text = read_data()
CHARS = sorted(list(set(text)))
vocab_size = len(CHARS)
TRAIN_DATA, VAL_DATA = train_val_split(text)

#%%
batch_size = 4 # paralell comp
block_size = 8 # context for predictions

xb, yb = get_batch("train")
yb.shape

#%%
# baseline model == bigram model
n_embd = 32

class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weights = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf")) #(B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        out = weights @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        # each token reads off the logits for the next token from the embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(head_size=n_embd) # same as C
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets = None):
        B, T = idx.shape
        # idx and targets are both tensors of B, T
        # returns score for next token of size B, T, C (where C == vocab size)
        token_embeddings = self.token_embedding_table(idx) #(B, T, C) where C == n_embd
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device) #(T, C)
        x = token_embeddings + position_embeddings # will broadcast position_embeddings from T, C to 1, T, C so it can sum
        x = self.sa_head(x)
        logits = self.lm_head(x) #(B, T, C) where C == vocab size
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx = B x T x C
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1:, :] # becomes B x C where -1 == predictions for next token
            probs = F.softmax(logits, dim = -1) # along C in B x C
            # sample from distribution
            idx_next = torch.multinomial(probs[0], num_samples=1) #B x 1
            idx = torch.cat([idx, idx_next], dim=1) # B X T C
        return idx

m = BigramLanguageModel(vocab_size=vocab_size)
logits, loss = m(idx=xb, targets=yb)
inp = torch.zeros([1,1], dtype=torch.long)
print(text_decoder(m.generate(idx=inp, max_new_tokens=100)[0].tolist()))
# %%
# simple training loop
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for steps in range(max_iters):
    xb, yb = get_batch("train")
    logits, loss = m(idx=xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

#%%
#self-attention trick
torch.manual_seed(12)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        cur_x = x[b, :t+1] #t, C
        xbow[b, t] = torch.mean(cur_x, 0)

#%%
# matrix implementation
wei = torch.tril(torch.ones(T, T)) #atm weights are just 1, i.e all historical tokens are weighted the same
wei = wei/wei.sum(1, keepdim=True) # normalise to sum to 1
xbow2 = wei @ x # (T, T) @ (B, T, C) --> torch creates new batch d for wei and does batched matrix multiplication --> (B, T, C)
xbow2.shape

#%%
# use softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1) # will still calculate weighted average for now, can be adapted to any input tril (which will be learned in the future)
xbow3 = wei @ x
wei
