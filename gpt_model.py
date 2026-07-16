import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size, embed_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_size)
  
  def forward(self, tokens):
    # tokens -> (batch_size, block_size)
    x =  self.embedding(tokens)
    # x -> (batch_size, block_size, embed_size)
    return x

def rotate_half(x):
  x1 = x[..., ::2]
  x2 = x[..., 1::2]

  x = torch.stack((-x2,x1),dim=-1)

  return x.flatten(-2)

def precompute_freqs(head_size, block_size):
  theta = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))

  positions = torch.arange(block_size).float()

  freqs = torch.outer(positions, theta)
  freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)
  return freqs

class MultiHeadCausalSelfAttention(nn.Module):
  def __init__(self, embed_size, num_heads, block_size, dropout):
    super().__init__()
    self.embed_size = embed_size
    self.num_heads = num_heads
    self.attn_dropout = nn.Dropout(dropout)
    self.resid_dropout = nn.Dropout(dropout)
    assert embed_size % num_heads == 0 # embed_size must be divisible by num_heads
    self.head_size = embed_size // num_heads
    self.c_attn = nn.Linear(embed_size, 3 * embed_size)
    self.c_proj = nn.Linear(embed_size, embed_size)

    freqs = precompute_freqs(self.head_size, block_size)
    self.register_buffer("cos_cached", freqs.cos())
    self.register_buffer("sin_cached", freqs.sin())


  def forward(self, x):
    B, T, C = x.shape
    # x: (B, T, C)

    q, k, v = self.c_attn(x).split(self.embed_size, dim=2)
    # c_attn -> (B,T,3C) -> split -> Q(B,T,C) K(B,T,C) V(B,T,C)

    q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
    k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
    v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
    # (B, T, 768) -> (B,T, 12, 64) -> (B, 12, T, 64)

    cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)
    sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)
    # cos,sin: (1, 1, T, D)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    # q,k: (B, H, T, D)

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=self.attn_dropout.p if self.training else 0.0,
        is_causal=True,)
    # Q(B, H, T, D), K(B, H, T, D), V(B, H, T, D)
    # -> Flash Attention -> (B, H, T, D)

    out = out.transpose(1, 2).contiguous().view(B, T, C) 
    # (B,H,T,D) -> (B,T,H,D) -> (B,T,C)

    out = self.c_proj(out)
    # (B,T,C) -> c_proj -> (B,T,C)

    out = self.resid_dropout(out)
    # (B,T,C) -> (B,T,C)
    return out

class MLP(nn.Module):
  def __init__(self, embed_size,dropout):
    super().__init__()
    self.c_fc = nn.Linear(embed_size, 4 * embed_size)
    self.gelu = nn.GELU()
    self.c_proj = nn.Linear(4 * embed_size, embed_size)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    x = self.c_fc(x)       # (B,T,C)  -> (B,T,4C)
    x = self.gelu(x)       # (B,T,4C) -> (B,T,4C)
    x = self.c_proj(x)     # (B,T,4C) -> (B,T,C)
    x = self.dropout(x)    # (B,T,C)  -> (B,T,C)
    return x
  
class DecoderBlock(nn.Module):
  def __init__(self, embed_size, num_heads, block_size, dropout):
    super().__init__()
    self.layer_norm1 = nn.LayerNorm(embed_size)
    self.attn = MultiHeadCausalSelfAttention(embed_size, num_heads, block_size, dropout)
    self.layer_norm2 = nn.LayerNorm(embed_size)
    self.mlp = MLP(embed_size, dropout)

  def forward(self, x):
    x = x + self.attn(self.layer_norm1(x)) 
    x = x + self.mlp(self.layer_norm2(x))
    return x
  
class GPT(nn.Module):
  def __init__(self, vocab_size, embed_size, block_size, dropout, num_heads, num_layers):
    super().__init__()
    self.token_embedding = TokenEmbedding(vocab_size, embed_size)
    self.dropout = nn.Dropout(dropout)
    self.blocks = nn.ModuleList(
        [DecoderBlock(embed_size, num_heads, block_size, dropout) 
        for _ in range(num_layers)])
    self.ln_f = nn.LayerNorm(embed_size)
    self.lm_head = nn.Linear(embed_size, vocab_size, bias=False) # (B,T,C) -> (B,T,V(vocab_size))
    # Weight Tying
    self.lm_head.weight = self.token_embedding.embedding.weight # (V,C)

  def forward(self, tokens):
    # tokens -> (B, T)

    x = self.token_embedding(tokens) # -> (B, T, C)
    x = self.dropout(x)

    for block in self.blocks:
      x = block(x)

    x = self.ln_f(x)

    logits = self.lm_head(x) # (B,T,C) -> (B,T,V(vocab_size))
    
    return logits