# Hyperparameters and settings

SEQ_LEN          = 10     # context window: how many words to look back
INPUT_MIN_FREQ   = 50     # input vocab: ~300 words → patterns repeat across train/val
OUTPUT_MIN_FREQ  = 2000   # output vocab: ~6 most common words
EMBED_DIM        = 64
LSTM_UNITS       = 128    # bidir → 256, fast on CPU
ATTN_UNITS       = 128
DENSE_UNITS      = 128
DROPOUT          = 0.15   # mild: prevents memorizing unique contexts without slowing too much
BATCH_SIZE       = 512
EPOCHS           = 300    # ~50 min at 10 sec/epoch
LEARNING_RATE    = 0.005
TRAIN_SPLIT      = 0.8
DATA_URL         = "https://www.gutenberg.org/files/1661/1661-0.txt"
SAVE_PATH        = "sherlock.txt"
MODEL_PATH       = "sherlock_model.pt"
PATIENCE         = 15
