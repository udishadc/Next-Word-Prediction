import os
import re
import urllib.request
import numpy as np
from collections import Counter
from config import SEQ_LEN, INPUT_MIN_FREQ, OUTPUT_MIN_FREQ, DATA_URL, SAVE_PATH, TRAIN_SPLIT


def download_sherlock():
    """Download and cache the text from Project Gutenberg."""
    if not os.path.exists(SAVE_PATH):
        print("Downloading Sherlock Holmes...")
        urllib.request.urlretrieve(DATA_URL, SAVE_PATH)
    else:
        print("Using cached Sherlock Holmes text.")
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters.")
    return text


def clean_text(text):
    """Strip Gutenberg header/footer, lowercase, remove punctuation, collapse whitespace."""
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end   = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start != -1:
        text = text[start + 60:]
    if end != -1:
        text = text[:end]
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(words):
    """
    Build two vocabularies: a large input vocab for encoding context windows,
    and a small output vocab of the most frequent target words.
    """
    counts = Counter(words)

    # --- Input vocabulary ---
    input_vocab = ["<PAD>", "<OOV>"] + [
        w for w, c in counts.most_common() if c >= INPUT_MIN_FREQ
    ]
    w2i = {w: i for i, w in enumerate(input_vocab)}
    i2w = {i: w for i, w in enumerate(input_vocab)}

    # --- Output vocabulary ---
    output_words = [w for w, c in counts.most_common() if c >= OUTPUT_MIN_FREQ]
    out_w2i = {w: i for i, w in enumerate(output_words)}
    out_i2w = {i: w for i, w in enumerate(output_words)}

    print(f"Input  vocab size : {len(input_vocab):,}")
    print(f"Output vocab size : {len(output_words):,}")
    print(f"Output words      : {output_words[:20]} ...")
    return input_vocab, w2i, i2w, out_w2i, out_i2w


def make_sequences(token_ids, i2w, out_w2i):
    """Sliding window sequences; only keeps samples whose target is in the output vocab."""
    sequences = []
    for i in range(SEQ_LEN, len(token_ids)):
        target_word = i2w.get(token_ids[i], "<OOV>")
        if target_word in out_w2i:
            out_target = out_w2i[target_word]
            sequences.append(list(token_ids[i - SEQ_LEN: i]) + [out_target])
    sequences = np.array(sequences, dtype=np.int32)
    print(f"Total sequences  : {len(sequences):,}")
    return sequences


def train_test_split(sequences):
    """Shuffle and split 80/20 with a fixed seed."""
    rng = np.random.default_rng(seed=42)
    shuffled = sequences.copy()
    rng.shuffle(shuffled)
    split      = int(len(shuffled) * TRAIN_SPLIT)
    train_seqs = shuffled[:split]
    test_seqs  = shuffled[split:]
    print(f"Train sequences  : {len(train_seqs):,}")
    print(f"Test sequences   : {len(test_seqs):,}")
    return train_seqs, test_seqs


def load_data():
    """Download, clean, tokenize, and split the corpus."""
    raw_text = download_sherlock()
    clean    = clean_text(raw_text)
    words    = clean.split()
    print(f"Total words      : {len(words):,}")

    input_vocab, w2i, i2w, out_w2i, out_i2w = build_vocab(words)
    token_ids = [w2i.get(w, 1) for w in words]
    sequences = make_sequences(token_ids, i2w, out_w2i)
    train_seqs, test_seqs = train_test_split(sequences)

    return input_vocab, w2i, i2w, out_w2i, out_i2w, train_seqs, test_seqs
