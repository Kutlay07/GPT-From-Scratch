from gpt_model import GPT
from config import *
from gpt_data import *

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import math

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Folders
# =========================
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("weights", exist_ok=True)

# =========================
# Model
# =========================
model = GPT(
    vocab_size=VOCAB_SIZE,
    embed_size=EMBED_SIZE,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
).to(device)



# =========================
# Dataset
# =========================
dataset_path = download_dataset()
text = load_dataset(dataset_path)
text = clean_text(text)
token_ids = tokenize(text)

dataset = GPTDataset(token_ids, BLOCK_SIZE,stride=STRIDE)

train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,)

# =========================
# Optimizer
# =========================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================
# SCHEDULER
# ==========================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1

        if self.step_num < self.warmup_steps:
            lr = self.max_lr * self.step_num / self.warmup_steps

        else:
            progress = (
                (self.step_num - self.warmup_steps)
                / (self.total_steps - self.warmup_steps)
            )

            cosine = 0.5 * (1 + math.cos(math.pi * progress))

            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = 0.0

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, state_dict):
        self.step_num = state_dict["step_num"]

total_steps = len(train_loader) * EPOCHS

scheduler = WarmupCosineScheduler(
    optimizer=optimizer,
    warmup_steps=WARMUP_STEPS,
    total_steps=total_steps,
    max_lr=LEARNING_RATE,
    min_lr=MIN_LR,
)

# --- GradScaler ---
scaler = GradScaler(enabled=device.type == "cuda")

# =========================
# Resume Training
# =========================

CHECKPOINT_PATH = "/kaggle/input/datasets/kutlay07/checkpoint-loss-3-1393-pt/checkpoint_loss_3_1393.pt"

best_loss = float("inf")

if os.path.exists(CHECKPOINT_PATH):

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )
        
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    best_loss = checkpoint["loss"]

    print(f"Checkpoint loaded successfully!")
    print(f"Resuming from loss: {best_loss:.4f}\n")

else:

    print("No checkpoint found. Starting from scratch.")

# =========================
# Compile Model (PyTorch 2.x)
# =========================
if hasattr(torch, "compile"):
    model = torch.compile(model)
    print("torch.compile enabled!")

# =========================
# Training
# =========================

model.train()

for epoch in range(EPOCHS):

    epoch_loss = 0.0

    for batch_idx, (x, y) in enumerate(train_loader):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)

            B, T, V = logits.shape

            logits = logits.reshape(B * T, V)
            targets = y.reshape(B * T)

            loss = F.cross_entropy(logits, targets)

        scaler.scale(loss).backward()
        
        # Since we are using AMP, convert the gradients back to normal
        scaler.unscale_(optimizer)
        
        # Gradient Clipping (to prevent the exploding gradients)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()

        epoch_loss += loss.item()

        if batch_idx % 100 == 0:
            
            current_lr = optimizer.param_groups[0]["lr"]
            
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
                f"LR: {current_lr:.6f}"
            )

    avg_loss = epoch_loss / len(train_loader)
    model.eval()

    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            with autocast(
                device_type=device.type,
                enabled = device.type == "cuda"
            ):
                logits = model(x)

                B,T,V = logits.shape
                
                logits = logits.reshape(B * T, V)
                targets = y.reshape(B * T)
                
                loss = F.cross_entropy(logits, targets)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    perplexity = math.exp(avg_val_loss)
    model.train()
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss      : {avg_loss:.4f}")
    print(f"Validation Loss : {avg_val_loss:.4f}")
    print(f"Perplexity      : {perplexity:.2f}")
    # =========================
    # Save checkpoint
    # =========================
    checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch+1}.pt"
    
    save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": save_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
        },
        checkpoint_path,
    )

    # Verify checkpoint
    try:
        torch.load(checkpoint_path, map_location="cpu")
        print("Checkpoint verified successfully.")
    except Exception as e:
        print(f"Checkpoint verification failed: {e}")

    # Save best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_model_path = "weights/best_model.pt"

        save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(save_model.state_dict(), best_model_path)

        try:
            torch.load(best_model_path, map_location="cpu")
            print("Best model saved and verified.")
        except Exception as e:
            print(f"Best model verification failed: {e}")

    print(f"Best Loss: {best_loss:.4f}\n")

# =========================
# Save final model
# =========================
last_model_path = "weights/last_model.pt"

save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
torch.save(save_model.state_dict(), last_model_path)

try:
    torch.load(last_model_path, map_location="cpu")
    print("Final model saved and verified.")
except Exception as e:
    print(f"Final model verification failed: {e}")