import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_PATH, PATIENCE

logger = logging.getLogger("sherlock")


class SherlockDataset(Dataset):
    """
    PyTorch Dataset — wraps sequences so DataLoader can
    batch and shuffle them automatically during training.
    """
    def __init__(self, sequences):
        self.X = torch.tensor(sequences[:, :-1], dtype=torch.long)
        self.y = torch.tensor(sequences[:, -1],  dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_perplexity(loss):
    """
    Perplexity = exp(cross-entropy loss).
    Lower is better. Target is below 250.
    """
    return np.exp(loss)


def train_model(model, train_seqs, test_seqs):
    """Full training loop with early stopping and LR scheduling."""

    train_dataset = SherlockDataset(train_seqs)
    test_dataset  = SherlockDataset(test_seqs)

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")
    model.to(device)

    best_val_loss    = float('inf')
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [],
               "train_acc" : [], "val_acc" : [],
               "val_top5_acc": []}

    for epoch in range(EPOCHS):

        # -- Training --------------------------------------------------
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss    += loss.item()
            preds          = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total   += y_batch.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc  = train_correct / train_total

        # -- Validation ------------------------------------------------
        model.eval()
        val_loss, val_correct, val_top5_correct, val_total = 0, 0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits  = model(X_batch)
                loss    = criterion(logits, y_batch)
                val_loss    += loss.item()
                preds        = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                top5_preds   = logits.topk(5, dim=1).indices
                val_top5_correct += (top5_preds == y_batch.unsqueeze(1)).any(dim=1).sum().item()
                val_total   += y_batch.size(0)

        avg_val_loss     = val_loss / len(test_loader)
        avg_val_acc      = val_correct / val_total
        avg_val_top5_acc = val_top5_correct / val_total

        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)
        history["val_top5_acc"].append(avg_val_top5_acc)

        logger.info(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {avg_train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {avg_val_acc:.4f} | "
            f"Val Top-5 Acc: {avg_val_top5_acc:.4f} | "
            f"Perplexity: {compute_perplexity(avg_val_loss):.2f}"
        )

        # -- Early stopping --------------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info(f"  Model saved.")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}.")
                break

    # -- Plot training curves ------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_acc"], label="Train Accuracy", color="steelblue")
    ax1.plot(history["val_acc"],   label="Val Accuracy",   color="coral")
    ax1.set_title("Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history["train_loss"], label="Train Loss", color="steelblue")
    ax2.plot(history["val_loss"],   label="Val Loss",   color="coral")
    ax2.set_title("Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()
    logger.info("Training curves saved to training_curves.png")

    return history
