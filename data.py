#%%
import torch
import numpy as np 

data_file =  "./input.txt"

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
import torch.nn as nn
from torch.nn import functional as F 

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        # each token reads off the logits for the next token from the embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets = None):
        # idx and targets are both tensors of B, T
        # returns score for next token of size B, T, C (where C == vocab size)
        logits = self.token_embedding_table(idx)
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
            logits, loss = self(idx)
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
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(100):
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
