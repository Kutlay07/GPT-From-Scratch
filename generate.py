import torch

from config import *
from gpt_model import GPT
from gpt_data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT(
    vocab_size = VOCAB_SIZE,
    embed_size = EMBED_SIZE,
    block_size = BLOCK_SIZE,
    dropout = DROPOUT,
    num_heads = NUM_HEADS,
    num_layers = NUM_LAYERS,
).to(device)

model.load_state_dict(
    torch.load("weights/gpt_model.pt", map_location=device)
)

