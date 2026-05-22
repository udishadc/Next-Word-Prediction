import torch
import torch.nn as nn
import torch.nn.functional as F
from config import EMBED_DIM, LSTM_UNITS, ATTN_UNITS, DENSE_UNITS, DROPOUT


class BahdanauAttention(nn.Module):
    """Additive attention: learns alignment weights over all LSTM timesteps."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, ATTN_UNITS)
        self.W2 = nn.Linear(hidden_dim, ATTN_UNITS)
        self.v  = nn.Linear(ATTN_UNITS, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_dim)
        score   = self.v(torch.tanh(self.W1(lstm_out) + self.W2(lstm_out)))
        # score: (batch, seq_len, 1)
        weights = torch.softmax(score, dim=1)
        # weighted sum across timesteps
        context = (weights * lstm_out).sum(dim=1)
        # context: (batch, hidden_dim)
        return context, weights


class SherlockModel(nn.Module):
    """Embedding -> 2-layer BiLSTM -> Bahdanau Attention -> Dense -> output logits."""
    def __init__(self, input_vocab_size, output_vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=input_vocab_size,
            embedding_dim=EMBED_DIM,
            padding_idx=0
        )
        self.dropout1 = nn.Dropout(DROPOUT)

        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=LSTM_UNITS,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT
        )

        # bidirectional doubles the output size
        self.attention = BahdanauAttention(hidden_dim=LSTM_UNITS * 2)

        self.dense    = nn.Linear(LSTM_UNITS * 2, DENSE_UNITS)
        self.dropout2 = nn.Dropout(DROPOUT)
        self.output   = nn.Linear(DENSE_UNITS, output_vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) — indices from input vocab
        x = self.embedding(x)           # (batch, seq_len, embed_dim)
        x = self.dropout1(x)

        x, _ = self.lstm(x)             # (batch, seq_len, lstm_units*2)

        context, attn_weights = self.attention(x)   # (batch, lstm_units*2)

        x = F.relu(self.dense(context))  # (batch, dense_units)
        x = self.dropout2(x)
        x = self.output(x)               # (batch, output_vocab_size)
        return x
