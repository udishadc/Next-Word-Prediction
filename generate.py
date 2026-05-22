import torch
import numpy as np
from config import SEQ_LEN


def generate_text(seed_text, max_phrase_len, model, w2i, out_i2w, temperature=0.7):
    """
    Predict next words autoregressively from seed_text.
    Returns the full generated string and per-step top-5 probabilities.
    """
    device = next(model.parameters()).device
    model.eval()

    current_words   = seed_text.lower().split()
    words_and_probs = []

    with torch.no_grad():
        for _ in range(max_phrase_len):

            token_ids = [w2i.get(w, 1) for w in current_words]
            if len(token_ids) < SEQ_LEN:
                token_ids = [0] * (SEQ_LEN - len(token_ids)) + token_ids
            else:
                token_ids = token_ids[-SEQ_LEN:]

            x      = torch.tensor([token_ids], dtype=torch.long).to(device)
            logits = model(x)

            # temperature scaling
            logits = logits / temperature
            probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

            # top 5 candidates from OUTPUT vocab
            top5_indices = np.argsort(probs)[-10:][::-1]
            top5 = [
                {"word": out_i2w.get(int(idx), "<OOV>"),
                 "probability": float(probs[idx])}
                for idx in top5_indices
                if out_i2w.get(int(idx), "<OOV>") != "<OOV>"
            ][:5]

            # penalize recently used words to reduce repetition
            recent_words = set(current_words[-5:])
            top5_probs = np.array([
                t["probability"] * 0.2 if t["word"] in recent_words
                else t["probability"]
                for t in top5
            ])
            top5_probs /= top5_probs.sum()
            chosen_idx  = np.random.choice(len(top5), p=top5_probs)
            next_word   = top5[chosen_idx]["word"]

            words_and_probs.append({
                "input_context": " ".join(current_words[-5:]),
                "chosen_word":   next_word,
                "top5":          top5
            })

            current_words.append(next_word)

    output_text = " ".join(current_words)
    return output_text, words_and_probs


def print_step_breakdown(seed_text, steps):
    """Pretty print the top 5 breakdown for one generation example."""
    print("=" * 65)
    print("STEP-BY-STEP TOP 5 PREDICTIONS")
    print("=" * 65)
    for i, step in enumerate(steps):
        print(f"\nStep {i+1:02d} | Chosen: '{step['chosen_word']}'")
        print(f"  Context: '...{step['input_context']}'")
        print("  Top 5:")
        for rank, candidate in enumerate(step["top5"], 1):
            marker = " <- chosen" if candidate["word"] == step["chosen_word"] else ""
            print(f"    {rank}. '{candidate['word']:<15}' "
                  f"{candidate['probability']:.4f}{marker}")
