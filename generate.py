import torch
import torch.nn.functional as F

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
    torch.load("weights/best_model.pt", map_location=device))
model.eval()

def generate(prompt, max_new_tokens, temperature=0.0, top_k=None, top_p=None):
    """
    Generate text using the trained GPT model.

    Args:
        prompt (str): Input prompt.
        max_new_tokens (int): Number of tokens to generate.
        temperature (float): Sampling temperature. 0 = greedy decoding.
        top_k (int | None): Restrict sampling to the k most probable tokens.
        top_p (float | None): Nucleus sampling threshold.

    Returns:
        str: Generated text.
    """
    model.eval()
    token_ids = tokenize(prompt)
    tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    tokens = tokens.to(device)
    
    past_kvs = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            if step == 0:
                input_tokens = tokens[:, -BLOCK_SIZE:]
                
            else:
                input_tokens = tokens[:, -1:]

            logits, past_kvs = model(
                input_tokens,
                past_kvs=past_kvs,
                use_cache=True,
            )
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            else:
                logits = logits / temperature
        
                # Top-k Sampling
                if top_k is not None:
                    values, _ = torch.topk(logits, top_k)
                    logits[logits < values[:, [-1]]] = -float("inf")

                # Top-p (Nucleus Sampling)
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True,)

                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=-1)
    token_ids = tokens.squeeze(0).tolist()
    text = decode(token_ids)
    return text

# Example of usage 
# if __name__ == "__main__":
#     output = generate("The", 100, temperature=0.8, top_k=20, top_p=0.9)
#     print(output)