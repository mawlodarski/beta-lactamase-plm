#in-house transformer-based protein language model
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Layer, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention
)
import numpy as np
from tokenization import PAD, CLS_ID

class AttnMaskExpand(Layer):
    """
    Builds an attention mask that ignores padded 0 tokens.

    Returns a boolean mask shaped (B, 1, L) to broadcast across heads and queries.
    """
    def call(self, tokens):
        # tokens: (B, L)
        # mask:   (B, 1, L)
        return tf.not_equal(tokens, PAD)[:, tf.newaxis, :]

    def get_config(self):
        return super().get_config()

class PositionalAdder(Layer):
    """
    Adds learned positional embeddings to token embeddings.
    """
    def __init__(self, max_len, emb_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.pos_embed = Embedding(input_dim=max_len, output_dim=emb_dim, name="pos_embed")

    def call(self, inputs, tok_embed):
        # inputs:   (B, L) token ids
        # tok_embed:(B, L, D)
        B = tf.shape(inputs)[0]
        L = tf.shape(inputs)[1]
        pos_ids = tf.range(L)[tf.newaxis, :]      # (1, L)
        pos = self.pos_embed(pos_ids)             # (1, L, D)
        pos = tf.repeat(pos, repeats=B, axis=0)   # (B, L, D)
        return tok_embed + pos

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "emb_dim": self.emb_dim})
        return cfg

class Encoder(Layer):
    """
    One transformer encoder block
    Supports returning attention scores for interpretability.
    """
    def __init__(self, emb_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads}).")

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=emb_dim // num_heads,
            name="mha"
        )

        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(emb_dim)],
            name="ffn"
        )

        self.norm1 = LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = LayerNormalization(epsilon=1e-6, name="norm2")
        self.dropout1 = Dropout(dropout, name="dropout1")
        self.dropout2 = Dropout(dropout, name="dropout2")

    def call(self, x, mask=None, training=False, return_attn=False):
        """
        x:    (B, L, D)
        mask: (B, 1, L) boolean (broadcastable)
        """
        if return_attn:
            attn_out, attn_scores = self.mha(
                query=x,
                value=x,
                key=x,
                attention_mask=mask,
                return_attention_scores=True,
                training=training
            )
        else:
            attn_out = self.mha(
                query=x,
                value=x,
                key=x,
                attention_mask=mask,
                training=training
            )
            attn_scores = None

        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.norm1(x + attn_out)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.norm2(out1 + ffn_out)

        if return_attn:
            # attn_scores shape is typically (B, heads, L, L)
            return out2, attn_scores

        return out2

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "emb_dim": self.emb_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        })
        return cfg

# prepend CLS token
def prepend_cls(tokens):
    """
    Prepend CLS token to a batch of token ids.
    """
    B = tf.shape(tokens)[0]
    cls_col = tf.fill([B, 1], tf.cast(CLS_ID, tf.int32))
    tokens = tf.cast(tokens, tf.int32)
    return tf.concat([cls_col, tokens], axis=1)

# -----------------------------
# BL vs non-BL Transformer model
# -----------------------------

class TransformerBLDetector(Model):
    """
    Binary classifier: beta-lactamase (1) vs non-beta-lactamase (0).
    """
    def __init__(self,
                 vocab_size,
                 emb_dim,
                 num_heads,
                 ff_dim,
                 max_len,        
                 num_layers,
                 dropout=0.1,
                 name="bl_detector",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_len = max_len
        self.max_len_with_cls = max_len + 1
        self.num_layers = num_layers
        self.dropout = dropout

        # Token embedding and positional embedding
        self.tok_embed = Embedding(vocab_size, emb_dim, name="tok_embed")
        self.pos_add = PositionalAdder(self.max_len_with_cls, emb_dim, name="pos_add")

        # Attention mask (ignores PAD tokens)
        self.mask_layer = AttnMaskExpand(name="attn_mask")

        # Encoder stack
        self.encoders = [
            Encoder(emb_dim, num_heads, ff_dim, dropout=dropout, name=f"encoder_{i}")
            for i in range(num_layers)
        ]

        # Classification head
        self.head_dense = Dense(256, activation="relu", name="head_dense")
        self.head_drop = Dropout(dropout, name="head_dropout")
        self.out = Dense(1, activation="sigmoid", name="bla_output")

    def build(self, input_shape):
        """
        Creates weights and enables model.summary().
        """
        dummy_tokens = tf.zeros((1, self.max_len), dtype=tf.int32)
        _ = self.call(dummy_tokens, training=False, return_attn=False, return_embedding=False)
        super().build(input_shape)

    def call(self, tokens, training=False, return_attn=False, return_embedding=False):
        """
        tokens: (B, max_len) int32 token ids, already padded/truncated to max_len
                (these should include EOS where applicable; CLS is added here)

        return_attn: if True, returns per-layer attention score tensors
        return_embedding: if True, returns CLS embedding vector per sequence (for UMAP)
        """
        # Prepend CLS and build mask on the resulting token sequence
        tokens_cls = prepend_cls(tokens)  # (B, L+1)
        mask = self.mask_layer(tokens_cls)  # (B, 1, L+1)

        # Embed + add positions
        tok = self.tok_embed(tokens_cls)          # (B, L+1, D)
        x = self.pos_add(tokens_cls, tok)         # (B, L+1, D)

        attentions = [] if return_attn else None

        # Encoder stack (optionally collect attention)
        for enc in self.encoders:
            if return_attn:
                x, attn = enc(x, mask=mask, training=training, return_attn=True)
                attentions.append(attn)
            else:
                x = enc(x, mask=mask, training=training, return_attn=False)

        # CLS embedding is the sequence-level representation
        cls_emb = x[:, 0, :]  # (B, D)

        # Binary prediction head
        h = self.head_dense(cls_emb)
        h = self.head_drop(h, training=training)
        y = self.out(h)

        out = {"bla_output": y}

        if return_embedding:
            out["embedding"] = cls_emb

        if return_attn:
            out["attn"] = attentions

        return out

    def get_sequence_embedding(self, tokens, batch_size=256):
        """
        Convenience method to get CLS embeddings for UMAP.
        tokens: array/tensor of shape (N, max_len)
        returns: numpy array of shape (N, emb_dim)
        """
        ds = tf.data.Dataset.from_tensor_slices(tokens).batch(batch_size)
        embs = []
        for batch in ds:
            out = self(batch, training=False, return_attn=False, return_embedding=True)
            embs.append(out["embedding"])
        return tf.concat(embs, axis=0).numpy()

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "vocab_size": self.vocab_size,
            "emb_dim": self.emb_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "max_len": self.max_len,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "name": self.name,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        allowed = {
            "vocab_size", "emb_dim", "num_heads", "ff_dim",
            "max_len", "num_layers", "dropout", "name"
        }
        filtered = {k: v for k, v in config.items() if k in allowed}
        return cls(**filtered)

# -----------------------------
# Ambler Transformer model
# -----------------------------

class TransformerAmblerModel(Model):
    """
    Transformer-based Ambler class classifier (form CLS token).

    The model predicts the Ambler class of a protein sequence using a CLS
    sequence embedding.
    """
    def __init__(self,
                vocab_size,
                n_ambler_classes,
                emb_dim,
                num_heads,
                ff_dim,
                max_len,
                num_layers,
                dropout=0.1,
                name="ambler_classifier",
                **kwargs):
        super().__init__(name=name, **kwargs)

        self.vocab_size = vocab_size
        self.n_ambler_classes = n_ambler_classes
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_len = max_len
        self.max_len_with_cls = max_len + 1
        self.num_layers = num_layers
        self.dropout = dropout

        # Shared embedding + positional add + mask
        self.tok_embed = Embedding(vocab_size, emb_dim, name="tok_embed")
        self.pos_add = PositionalAdder(self.max_len_with_cls, emb_dim, name="pos_add")
        self.mask_layer = AttnMaskExpand(name="attn_mask")

        # Shared encoder stack
        self.encoders = [
            Encoder(emb_dim, num_heads, ff_dim, dropout=dropout, name=f"encoder_{i}")
            for i in range(num_layers)
        ]

        # Ambler head
        self.head_dense = Dense(256, activation="relu", name="head_dense")
        self.head_drop = Dropout(dropout, name="head_dropout")
        self.ambler_out = Dense(n_ambler_classes, activation="softmax", name="ambler_output")

    def build(self, input_shape):
        dummy_tokens = tf.zeros((1, self.max_len), dtype=tf.int32)
        _ = self.call(dummy_tokens, training=False, return_attn=False, return_embedding=False)
        super().build(input_shape)

    @staticmethod
    def _prepend_cls(tokens):
        """
        Prepend CLS token to a batch of token ids.
        tokens: (B, L) -> (B, L+1)
        """
        B = tf.shape(tokens)[0]
        cls_col = tf.fill([B, 1], tf.cast(CLS_ID, tf.int32))
        tokens = tf.cast(tokens, tf.int32)
        return tf.concat([cls_col, tokens], axis=1)

    def call(self, tokens, training=False, return_attn=False, return_embedding=False):
        """
        tokens: (B, max_len) int32 token ids (padded/truncated), should include EOS.
        """
        tokens_cls = self._prepend_cls(tokens)              # (B, L+1)
        mask = tf.not_equal(tokens_cls, PAD)[:, tf.newaxis, :]  # (B, 1, L+1)

        tok = self.tok_embed(tokens_cls)                    # (B, L+1, D)
        x = self.pos_add(tokens_cls, tok)                   # (B, L+1, D)

        attentions = [] if return_attn else None

        for enc in self.encoders:
            if return_attn:
                x, attn = enc(x, mask=mask, training=training, return_attn=True)
                attentions.append(attn)
            else:
                x = enc(x, mask=mask, training=training, return_attn=False)

        cls_emb = x[:, 0, :]                                # (B, D)

        h = self.head_dense(cls_emb)
        h = self.head_drop(h, training=training)
        y = self.ambler_out(h)                              # (B, n_ambler_classes)

        out = {"ambler_output": y}

        if return_embedding:
            out["embedding"] = cls_emb

        if return_attn:
            out["attn"] = attentions

        return out

    def get_sequence_embedding(self, tokens, batch_size=256):
        """
        Extract CLS embeddings for UMAP.
        tokens: (N, max_len)
        returns: (N, emb_dim)
        """
        ds = tf.data.Dataset.from_tensor_slices(tokens).batch(batch_size)
        embs = []
        for batch in ds:
            out = self(batch, training=False, return_attn=False, return_embedding=True)
            embs.append(out["embedding"])
        return tf.concat(embs, axis=0).numpy()

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "vocab_size": self.vocab_size,
            "n_ambler_classes": self.n_ambler_classes,
            "emb_dim": self.emb_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "max_len": self.max_len,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "name": self.name,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        allowed = {
            "vocab_size", "n_ambler_classes",
            "emb_dim", "num_heads", "ff_dim",
            "max_len", "num_layers", "dropout", "name"
        }
        filtered = {k: v for k, v in config.items() if k in allowed}
        return cls(**filtered)

# -----------------------------
# Enzyme-type Transformer model (Serine vs Metallo)
# -----------------------------

class TransformerEnzymeModel(Model):
    """
    Transformer-based enzyme-type classifier (Serine vs Metallo).

    The model predicts Enzyme_type of a protein sequence using a CLS
    sequence embedding.

    Interpretability:
      - `return_attn=True` returns per-layer attention score tensors (B, heads, L, L)
      - `return_embedding=True` returns the CLS embedding vector (B, D) for UMAP
    """
    def __init__(self,
                vocab_size,
                n_enzyme_classes,
                emb_dim,
                num_heads,
                ff_dim,
                max_len,
                num_layers,
                dropout=0.1,
                name="enzyme_classifier",
                **kwargs):
        super().__init__(name=name, **kwargs)

        self.vocab_size = vocab_size
        self.n_enzyme_classes = n_enzyme_classes
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_len = max_len
        self.max_len_with_cls = max_len + 1
        self.num_layers = num_layers
        self.dropout = dropout

        # Shared embedding + positional add + mask
        self.tok_embed = Embedding(vocab_size, emb_dim, name="tok_embed")
        self.pos_add = PositionalAdder(self.max_len_with_cls, emb_dim, name="pos_add")
        self.mask_layer = AttnMaskExpand(name="attn_mask")

        # Shared encoder stack
        self.encoders = [
            Encoder(emb_dim, num_heads, ff_dim, dropout=dropout, name=f"encoder_{i}")
            for i in range(num_layers)
        ]

        # Enzyme head
        self.head_dense = Dense(256, activation="relu", name="head_dense")
        self.head_drop = Dropout(dropout, name="head_dropout")
        self.enzyme_out = Dense(n_enzyme_classes, activation="softmax", name="enzyme_output")

    def build(self, input_shape):
        dummy_tokens = tf.zeros((1, self.max_len), dtype=tf.int32)
        _ = self.call(dummy_tokens, training=False, return_attn=False, return_embedding=False)
        super().build(input_shape)

    @staticmethod
    def _prepend_cls(tokens):
        """
        Prepend CLS token to a batch of token ids.
        tokens: (B, L) -> (B, L+1)
        """
        B = tf.shape(tokens)[0]
        cls_col = tf.fill([B, 1], tf.cast(CLS_ID, tf.int32))
        tokens = tf.cast(tokens, tf.int32)
        return tf.concat([cls_col, tokens], axis=1)

    def call(self, tokens, training=False, return_attn=False, return_embedding=False):
        """
        tokens: (B, max_len) int32 token ids (padded/truncated), should include EOS.
        """
        tokens_cls = self._prepend_cls(tokens)                   # (B, L+1)
        mask = tf.not_equal(tokens_cls, PAD)[:, tf.newaxis, :]   # (B, 1, L+1)

        tok = self.tok_embed(tokens_cls)                         # (B, L+1, D)
        x = self.pos_add(tokens_cls, tok)                        # (B, L+1, D)

        attentions = [] if return_attn else None

        for enc in self.encoders:
            if return_attn:
                x, attn = enc(x, mask=mask, training=training, return_attn=True)
                attentions.append(attn)
            else:
                x = enc(x, mask=mask, training=training, return_attn=False)

        cls_emb = x[:, 0, :]                                     # (B, D)

        h = self.head_dense(cls_emb)
        h = self.head_drop(h, training=training)
        y = self.enzyme_out(h)                                   # (B, n_enzyme_classes)

        out = {"enzyme_output": y}

        if return_embedding:
            out["embedding"] = cls_emb

        if return_attn:
            out["attn"] = attentions

        return out

    def get_sequence_embedding(self, tokens, batch_size=256):
        """
        Extract CLS embeddings for UMAP.
        tokens: (N, max_len)
        returns: (N, emb_dim)
        """
        ds = tf.data.Dataset.from_tensor_slices(tokens).batch(batch_size)
        embs = []
        for batch in ds:
            out = self(batch, training=False, return_attn=False, return_embedding=True)
            embs.append(out["embedding"])
        return tf.concat(embs, axis=0).numpy()

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "vocab_size": self.vocab_size,
            "n_enzyme_classes": self.n_enzyme_classes,
            "emb_dim": self.emb_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "max_len": self.max_len,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "name": self.name,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        allowed = {
            "vocab_size", "n_enzyme_classes",
            "emb_dim", "num_heads", "ff_dim",
            "max_len", "num_layers", "dropout", "name"
        }
        filtered = {k: v for k, v in config.items() if k in allowed}
        return cls(**filtered)


# -----------------------------
# Family Transformer model
# -----------------------------

class TransformerFamilyModel(Model):
    """
    Transformer-based beta-lactamase family classifier.

    The model predicts a family label from a protein sequence using a CLS
    sequence embedding.
    """
    def __init__(self,
                 vocab_size,
                 n_family_classes,
                 emb_dim,
                 num_heads,
                 ff_dim,
                 max_len,        # input length WITHOUT CLS (e.g., 600)
                 num_layers,
                 dropout=0.1,
                 name="family_classifier",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.vocab_size = vocab_size
        self.n_family_classes = n_family_classes
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_len = max_len
        self.max_len_with_cls = max_len + 1
        self.num_layers = num_layers
        self.dropout = dropout

        # Token embedding and positional embedding
        self.tok_embed = Embedding(vocab_size, emb_dim, name="tok_embed")
        self.pos_add = PositionalAdder(self.max_len_with_cls, emb_dim, name="pos_add")

        # Attention mask (ignores PAD tokens)
        self.mask_layer = AttnMaskExpand(name="attn_mask")

        # Encoder stack
        self.encoders = [
            Encoder(emb_dim, num_heads, ff_dim, dropout=dropout, name=f"encoder_{i}")
            for i in range(num_layers)
        ]

        # Classification head
        self.head_dense = Dense(256, activation="relu", name="head_dense")
        self.head_drop = Dropout(dropout, name="head_dropout")
        self.family_out = Dense(n_family_classes, activation="softmax", name="family_output")

    def build(self, input_shape):
        dummy_tokens = tf.zeros((1, self.max_len), dtype=tf.int32)
        _ = self.call(dummy_tokens, training=False, return_attn=False, return_embedding=False)
        super().build(input_shape)

    @staticmethod
    def _prepend_cls(tokens):
        """
        Prepend CLS token to a batch of token ids.
        tokens: (B, L) -> (B, L+1)
        """
        B = tf.shape(tokens)[0]
        cls_col = tf.fill([B, 1], tf.cast(CLS_ID, tf.int32))
        tokens = tf.cast(tokens, tf.int32)
        return tf.concat([cls_col, tokens], axis=1)

    def call(self, tokens, training=False, return_attn=False, return_embedding=False):
        """
        tokens: (B, max_len) int32 token ids (padded/truncated, including EOS)
        """
        tokens_cls = self._prepend_cls(tokens)                 # (B, L+1)
        mask = tf.not_equal(tokens_cls, PAD)[:, tf.newaxis, :] # (B, 1, L+1)

        tok = self.tok_embed(tokens_cls)                       # (B, L+1, D)
        x = self.pos_add(tokens_cls, tok)                      # (B, L+1, D)

        attentions = [] if return_attn else None

        for enc in self.encoders:
            if return_attn:
                x, attn = enc(x, mask=mask, training=training, return_attn=True)
                attentions.append(attn)
            else:
                x = enc(x, mask=mask, training=training, return_attn=False)

        # CLS embedding is the sequence-level representation
        cls_emb = x[:, 0, :]                                   # (B, D)

        h = self.head_dense(cls_emb)
        h = self.head_drop(h, training=training)
        y = self.family_out(h)                                 # (B, n_family_classes)

        out = {"family_output": y}

        if return_embedding:
            out["embedding"] = cls_emb

        if return_attn:
            out["attn"] = attentions

        return out

    def get_sequence_embedding(self, tokens, batch_size=256):
        """
        Extract CLS embeddings for UMAP.
        tokens: (N, max_len)
        returns: (N, emb_dim)
        """
        ds = tf.data.Dataset.from_tensor_slices(tokens).batch(batch_size)
        embs = []
        for batch in ds:
            out = self(batch, training=False, return_attn=False, return_embedding=True)
            embs.append(out["embedding"])
        return tf.concat(embs, axis=0).numpy()

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "vocab_size": self.vocab_size,
            "n_family_classes": self.n_family_classes,
            "emb_dim": self.emb_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "max_len": self.max_len,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "name": self.name,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        allowed = {
            "vocab_size", "n_family_classes",
            "emb_dim", "num_heads", "ff_dim",
            "max_len", "num_layers", "dropout", "name"
        }
        filtered = {k: v for k, v in config.items() if k in allowed}
        return cls(**filtered)