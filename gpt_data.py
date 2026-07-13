import kagglehub
import os
from config import BLOCK_SIZE

# Let's download our dataset (WikiText2 !!)

def download_dataset():
  path = kagglehub.dataset_download("vivekmettu/wikitext2-data")
  print(f"Dataset has been downloaded to {path}")
  return path

dataset_path = download_dataset() # move to train.py later

def load_dataset(dataset_path):
  train_path = os.path.join(dataset_path, "train.txt")

  with open(train_path, "r", encoding="utf-8") as f:
    text = f.read()
  return text

import tiktoken
enc = tiktoken.get_encoding("gpt2") # -> list[int] 

def tokenize(text):
   
  token_ids = enc.encode(text)
  return token_ids

def decode(token_ids):
    text = enc.decode(token_ids)
    return text

import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):
  def __init__(self, token_ids, block_size):
    self.token_ids = token_ids
    self.block_size = block_size

  def __len__(self):
    return len(self.token_ids) - self.block_size

  def __getitem__(self, idx):
    x = self.token_ids[idx: idx+self.block_size]
    y = self.token_ids[idx+1: idx +1+ self.block_size]
    x = torch.tensor(x, dtype = torch.long)
    y = torch.tensor(y, dtype = torch.long)
    return x,y 