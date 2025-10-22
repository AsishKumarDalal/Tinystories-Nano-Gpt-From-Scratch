

import torch
import torch.nn as nn
import numpy as np

from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")

EOT_TOKEN_ID = 50256

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 128, # Context length
    "emb_dim": 384,         # Embedding dimension
    "n_heads": 6,          # Number of attention heads
    "n_layers": 6,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    return total

total_params = count_parameters(model_2)
print(f"Total parameters: {total_params:,}")

import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm

enc = tiktoken.get_encoding("gpt2")



def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(EOT_TOKEN_ID)# encode_ordinary ignores any special tokens
    out = {'ids': ids, 'len': len(ids)}

    return out

if not os.path.exists("train.bin"):
    tokenized = ds.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=8,
        )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):

            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])

            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

def get_batch(split):

    if split == 'train':
        data = np.memmap('/kaggle/input/small-llm-tinystories/train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('/kaggle/input/small-llm-tinystories/validation.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':

        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

class LayerNorm(nn.Module):
  def __init__(self,emb_dim,bias=True):
    super().__init__()
    self.eps=1e-5
    self.scale=nn.Parameter(torch.ones(emb_dim))
    self.shift=nn.Parameter(torch.zeros(emb_dim)) if bias else torch.zeros(emb_dim)
  def forward(self,x):
     mean=x.mean(dim=-1, keepdim=True)
     variance=x.var(dim=-1, keepdim=True, unbiased=False)
     norm_x = (x - mean) / torch.sqrt(variance + self.eps)
     return self.scale * norm_x + self.shift
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(
          nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
          GELU(),
          nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
        )
    def forward(self,x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff= FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
    def forward(self,x):
        placeholder=x
        x= self.norm1(x)
        x=self.att(x)
        x=placeholder+x
        placeholder2=x
        x=self.norm2(x)
        x=self.ff(x)
        x=placeholder2+x
        return x

class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self,in_idx,y=None):
        batch_size, seq_len = in_idx.shape
        token_embedding=self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x=token_embedding+pos_embeds
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm (x)
        x= self.out_head(x)
        return x

    @torch.no_grad
    def generate(self, idx, max_new_tokens=250, context_size=128,
                         temperature=0.0, top_k=None, eos_id=EOT_TOKEN_ID):
                """
                Generate text with early stopping at EOT token.
                Stops immediately when EOT token is generated.
                """
                for _ in range(max_new_tokens):
                    idx_cond = idx[:, -context_size:]

                    with torch.no_grad():
                        logits = self(idx_cond)

                    logits = logits[:, -1, :]

                    # Top-k filtering
                    if top_k is not None:
                        top_logits, _ = torch.topk(logits, top_k)
                        min_val = top_logits[:, -1]
                        logits = torch.where(
                            logits < min_val,
                            torch.tensor(float("-inf")).to(logits.device),
                            logits
                        )

                    # Temperature sampling
                    if temperature > 0.0:
                        logits = logits / temperature
                        probs = torch.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                    else:
                        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

                    # STOP AT EOT TOKEN
                    if eos_id is not None and idx_next.item() == eos_id:
                        idx = torch.cat((idx, idx_next), dim=1)
                        break

                    idx = torch.cat((idx, idx_next), dim=1)

                return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(split, model, device, num_batches=5):
    """
    Calculate average loss over num_batches using get_batch function

    Args:
        split: 'train' or 'validation'
        model: the model to evaluate
        device: device to run on
        num_batches: number of batches to average over
    """
    total_loss = 0.

    for i in range(num_batches):
        input_batch, target_batch = get_batch(split)
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches

block_size = 128      # context window length
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_model(model, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader('train', model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader('validation', model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

# Remove the DataLoader section and use this instead:

def train_model_simple(model, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,
                       steps_per_epoch=1000):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for step in range(steps_per_epoch):
            # Use the get_batch function instead of DataLoader
            input_batch, target_batch = get_batch('train')

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample__(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

def generate_and_print_sample__(model, tokenizer, device, start_context):
    """Generate and print a sample text"""
    model.eval()
    context_size = 128
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = model.generate(
            idx=encoded,
            max_new_tokens=300,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# Note:
# Uncomment the following code to calculate the execution time
import time
start_time = time.time()

torch.manual_seed(123)
model = GPT(GPT_CONFIG_124M)
model.to(device)
# Ensure all parameters require gradients
model.train()

# Verify gradients are enabled (optional but good for debugging)
print("Checking if model parameters require grad...")
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"WARNING: {name} does not require grad!")
        param.requires_grad = True
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10

train_losses, val_losses, tokens_seen = train_model_simple(
    model, optimizer, device,
    num_epochs=num_epochs, eval_freq=100, eval_iter=5,
    start_context="a boy got a ball", tokenizer=tokenizer
)

# Note:
# Uncomment the following code to show the execution time
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# Save
checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()

}
torch.save(checkpoint, 'checkpoint.pth')

# Load
# checkpoint = torch.load('checkpoint.pth')
# model_2.load_state_dict(checkpoint['model_state_dict'])
# optimizer_2 .load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# model_2.train()

generate_and_print_sample__(model,tokenizer,device,"a boy was playing then his friend comes")