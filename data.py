#%%
import torch
import numpy as np 

# develop a function to read the data 
data_file =  "./input.txt"

def read_data(data_file = data_file):
    with open(data_file, "r") as f:
        lines = f.read()
    print(f"lenth of data in characters {len(lines)}")
    return lines

text = read_data()
# develop an encoding for generating tokens from words
chars = sorted(list(set(text)))
vocab_size = len(chars)
token_encoder = {ch: i for i, ch in enumerate(chars)}
token_decoder = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [token_encoder[c] for c in x]
decode = lambda x: ''.join([token_decoder[c] for c in x])
print(encode("hii there"))
print(decode(encode("hii there")))
#%%
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# develop a function for generating blocks of tokens
block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]
print(x)
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"cur context {context}, target is {target}")

#%%
# develop a function for generating batches of blocks of tokens
batch_size = 4 # paralell comp
block_size = 8 # context for predictions

def get_batch(split: str):
    data = train_data if split == "train" else val_data
    # randomly sample integers from size of data and return tensor of size batch_size 
    idx = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y

xb, yb = get_batch("train")
len(xb)
len(yb)
yb

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
        # returns score for next token of size B, T, C (vocab size)
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
print(decode(m.generate(idx=inp, max_new_tokens=100)[0].tolist()))
# %%
