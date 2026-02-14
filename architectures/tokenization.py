"""
Tokenization utilities for the in-house protein language model.
Defines amino acid vocabulary, special tokens, and padding behavior.
"""

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Amino acid vocabulary
# ----------------------------

# AA vocab: 1–20 for canonical residues
aa_vocab = {aa: i + 1 for i, aa in enumerate(list("ACDEFGHIKLMNPQRSTVWY"))}

# Special tokens
aa_vocab["X"] = 21         # unknown residue
aa_vocab["<EOS>"] = 22     # end-of-sequence marker
aa_vocab["<CLS>"] = 23     # classification/summary token
aa_vocab["<MASK>"] = 24  # mask token

# Padding token
PAD = 0
MASK_ID = aa_vocab["<MASK>"]
EOS_ID = aa_vocab["<EOS>"]
CLS_ID = aa_vocab["<CLS>"]

# ----------------------------
# Sequence length settings
# ----------------------------

# Model input length (WITHOUT CLS)
MAX_LEN = 600
MAX_LEN_WITH_CLS = MAX_LEN + 1

# Vocabulary size for Embedding layer
# Covers PAD=0 plus all non-zero token ids
vocab_size = max(aa_vocab.values()) + 1

# ----------------------------
# Tokenization helpers
# ----------------------------

def tokenize(seq, vocab=aa_vocab):
    """
    Convert an amino acid sequence into token ids and append EOS.
    Padding is handled separately.
    """
    tokens = [vocab.get(ch, vocab["X"]) for ch in str(seq)]
    tokens.append(vocab["<EOS>"])
    return tokens

def encode_and_pad(seqs, max_len=MAX_LEN):
    """
    Tokenize sequences (append EOS) then pad/truncate to max_len.
    Returns np.int32 array of shape (N, max_len).
    """
    tok = [tokenize(s) for s in seqs]
    X = pad_sequences(
        tok,
        maxlen=max_len,
        dtype="int32",
        padding="post",
        truncating="post",
        value=PAD
    )
    return np.asarray(X, dtype=np.int32)

def is_bla(sequence, model, threshold = 0.7, max_len = MAX_LEN):
    """Predict whether a protein sequence is a beta-lactamase."""
    x = encode_and_pad([sequence], max_len=max_len)
    prob = float(model.predict(x, verbose=0)[0][0])

    return {
        "is_bla": int(prob >= threshold),
        "probability": prob,
        "threshold": threshold
    }

# prepend CLS token
def prepend_cls(tokens):
    """
    Prepend CLS token to a batch of token ids.

    tokens: (B, L)
    returns: (B, L+1)
    """
    B = tf.shape(tokens)[0]
    cls_col = tf.fill([B, 1], tf.cast(CLS_ID, tf.int32))
    tokens = tf.cast(tokens, tf.int32)
    return tf.concat([cls_col, tokens], axis=1)