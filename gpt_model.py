import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size, embed_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_size)
  
  def forward(self, tokens):
    # tokens -> (batch_size, block_size)
    x =  self.embedding(tokens)
    # x -> (batch_size, block_size, embed_size)
    return x

class PositionalEmbedding(nn.Module):
  def __init__(self, block_size, embed_size):
    super().__init__()
    self.position_embedding_table = nn.Embedding(block_size, embed_size)

  def forward(self, tokens):
    T = tokens.shape[1] # tokens -> (B, T)
    positions = torch.arange(T, device=tokens.device) # -> (T,)
    x = self.position_embedding_table(positions) # -> (T, C)
    return x
  
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
    self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())
  

  def forward(self, x):
    B, T, C = x.shape # -> (B, T, C)
    q, k, v = self.c_attn(x).split(self.embed_size, dim=2)
    # c_attn -> (B,T,3C) -> split -> Q(B,T,C) K(B,T,C) V(B,T,C)
    q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
    k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
    v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
    # (B, T, 768) -> (B,T, 12, 64) -> (B, 12, T, 64)
    # (B, T, C) -> (B(batch), T(seq_len), H(num_heads), D(head_size)) -> (B,H,T,D)
    # And why do we want (B,H,T,D)? :To perform matrix multiplication (Q @ K.T) independently and in parallel for each head.
    attention_scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5) # (B,H,T,D) @ (B,H,D,T) -> (B,H,T,T)
    attention_scores = attention_scores.masked_fill(self.mask[:T, :T] == 0, float("-inf")) # (B,H,T,T)
    attention_probs = attention_scores.softmax(dim=-1) # (B,H,T,T) -> (B,H,T,T)
    attention_probs = self.attn_dropout(attention_probs) # (B,H,T,T) -> (B,H,T,T)
    out = attention_probs @ v # (B,H,T,T) @ (B,H,T,D) -> (B,H,T,D)
    out = out.transpose(1, 2).contiguous().view(B, T, C) # (B,H,T,D) -> (B,T,H,D) -> (B,T,C)
    out = self.c_proj(out) # (B,T,C) -> c_proj -> (B,T,C)
    out = self.resid_dropout(out) # (B,T,C) -> (B,T,C)
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
    self.position_embedding = PositionalEmbedding(block_size, embed_size)
    self.dropout = nn.Dropout(dropout)
    self.blocks = nn.ModuleList(
        [DecoderBlock(embed_size, num_heads, block_size, dropout) 
        for _ in range(num_layers)])
    self.ln_f = nn.LayerNorm(embed_size)
    self.lm_head = nn.Linear(embed_size, vocab_size, bias=False) # (B,T,C) -> (B,T,V(vocab_size))

  def forward(self, tokens):
    # tokens = (B, T)
    token_embeddings = self.token_embedding(tokens) # -> (B, T, C)
    position_embeddings = self.position_embedding(tokens) # -> (T, C)
    x = token_embeddings + position_embeddings # (B,T,C) + (T,C) -> (B,T,C)
    x = self.dropout(x)
    for block in self.blocks:
      x = block(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # (B,T,C) -> (B,T,V(vocab_size))
    return logits