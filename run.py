import os
import torch
import torch.nn as nn
import torch.nn.functional as F

mps_device = torch.device("cpu") 

if torch.cuda.is_available():
    torch.device("cuda")
elif torch.backends.mps.is_available():
    torch.device("mps")
else torch.device("cpu") 

# hyper param
batch_size = 32
block_size = 256
n_embd = 384
learn_rate = 3e-4
max_steps = 5000
eval_interval = 500
eval_iters = 250
n_head = 6
n_layer = 6
dropout = 0.2


# read data
with open("input.txt", "r") as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

# define encoder and decoder
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False, device=mps_device)
        self.query = nn.Linear(n_embd, head_size, bias=False, device=mps_device)
        self.value = nn.Linear(n_embd, head_size, bias=False, device=mps_device)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size, device=mps_device))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd, device=mps_device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, device=mps_device),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd, device=mps_device),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd, device=mps_device)
        self.ln2 = nn.LayerNorm(n_embd, device=mps_device)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, device=mps_device)
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd, device=mps_device
        )
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd, device=mps_device)
        self.lm_head = nn.Linear(n_embd, vocab_size, device=mps_device)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=mps_device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        print('\nINFINITE SHAKESPEARE\n')
        for iter in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            print(decode(idx_next[0].tolist()), end='', flush=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()
model.to(mps_device)


load_dir = './'

# Specify the filename for the parameters file
params_filename = 'parameters'

# Join the directory and filename to create the full path
params_path = os.path.join(load_dir, params_filename)

# Load the parameters from the specified path
model_state_dict = torch.load(params_path)
model.load_state_dict(model_state_dict)
model.eval()

model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device= mps_device), max_new_tokens=10000)
