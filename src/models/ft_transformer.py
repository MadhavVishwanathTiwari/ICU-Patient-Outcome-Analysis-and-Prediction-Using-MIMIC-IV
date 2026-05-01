"""
FT-Transformer — MIMIC-IV Clinical Prediction Tournament
=========================================================
A PyTorch implementation of the Feature Tokenizer + Transformer architecture
from Gorishniy et al. (2021): "Revisiting Deep Learning Models for Tabular Data"
https://arxiv.org/abs/2106.11959

Design notes for this pipeline
-------------------------------
• All features entering this model are already *numerical* (StandardScaler +
  SimpleImputer have been applied upstream in run_full_tournament.py).  There
  are therefore no categorical embeddings — only the linear "Feature Tokenizer"
  that projects each scalar x_i → d_token-dimensional vector via a learned
  weight vector w_i and bias b_i:

      token_i  =  x_i * W_i  +  b_i        W_i ∈ ℝ^d_token

  This is equivalent to a per-feature linear layer with no shared weights,
  giving the Transformer a rich, interaction-aware starting representation.

• A learnable [CLS] token is prepended; its final representation is used for
  the classification head (same convention as BERT).

• PyTorch is used deliberately — avoids TF/Keras graph conflicts with the
  existing Custom MLP (TF/Keras).  The two frameworks coexist safely as long
  as neither attempts to import the other's session/graph internals during
  the same fold.

• Runtime budget: conservative defaults (n_blocks=2, d_token=64, n_heads=4,
  epochs=50, patience=5) mirror the stacking MLP's philosophy — tractable
  across 48 tournament slots without sacrificing signal.

Public API
----------
build_ftt(input_dim, task_type, n_classes, **kwargs) → FTTransformerModel
train_ftt(model, X_tr, y_tr, device) → model   (in-place, returns self)
predict_ftt(model, X, device, task_type) → np.ndarray
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ── Sub-modules ──────────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Projects each numerical feature independently to a d_token-dimensional
    embedding.  Weight matrix W has shape (n_features, d_token); each column
    is used for exactly one feature, so features never share parameters here.

    forward(x)  x : (B, F)  →  tokens : (B, F, d_token)
    """

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        # One weight vector per feature, one global bias per dimension
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) → unsqueeze to (B, F, 1) and multiply by weight (F, d_token)
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return tokens                              # (B, F, d_token)


class TransformerBlock(nn.Module):
    """
    Standard Pre-LN Transformer encoder block:
        LayerNorm → MultiHeadAttention → residual
        LayerNorm → FFN               → residual

    Pre-LN (norm before attention/FFN) is more stable for tabular depths and
    matches the original FT-Transformer paper's reported best configuration.

    Each residual connection uses its own Dropout instance with the rate that
    matches that path's hyperparameter (attn_dropout for attention,
    ffn_dropout for FFN), keeping the two axes of regularisation independent.
    """

    def __init__(self, d_token: int, n_heads: int,
                 ffn_d_hidden: int, attn_dropout: float, ffn_dropout: float):
        super().__init__()
        assert d_token % n_heads == 0, "d_token must be divisible by n_heads"

        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=n_heads,
            dropout=attn_dropout, batch_first=True
        )
        # FIX (Bug 2): separate Dropout instances with the correct rate per path.
        # Previously a single self.drop = Dropout(ffn_dropout) was shared, which
        # silently applied the FFN dropout rate to the post-attention residual too.
        self.attn_drop = nn.Dropout(attn_dropout)   # post-attention residual
        self.ffn_drop  = nn.Dropout(ffn_dropout)    # post-FFN residual

        self.norm2 = nn.LayerNorm(d_token)
        self.ffn   = nn.Sequential(
            nn.Linear(d_token, ffn_d_hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_d_hidden, d_token),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_token)   S = n_features + 1  (includes [CLS])
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.attn_drop(h)   # attn_dropout rate

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.ffn_drop(h)    # ffn_dropout rate
        return x


class FTTransformerModel(nn.Module):
    """
    Full FT-Transformer for tabular binary / multiclass classification.

    Architecture
    ------------
    1. FeatureTokenizer   : (B, F)         → (B, F, d_token)
    2. Prepend [CLS]      : (B, F, d)      → (B, F+1, d)
    3. N × TransformerBlock
    4. Extract CLS token  : (B, F+1, d)    → (B, d)
    5. LayerNorm + Head   : (B, d)         → (B, n_out)
       binary   : n_out = 1,  no activation  (BCEWithLogitsLoss)
       multiclass: n_out = C, no activation  (CrossEntropyLoss)

    Parameters
    ----------
    n_features   : number of input features
    task_type    : 'binary' | 'multiclass'
    n_classes    : number of target classes (ignored for binary)
    d_token      : embedding dimension per feature
    n_heads      : number of attention heads  (must divide d_token)
    n_blocks     : number of Transformer encoder layers
    ffn_d_hidden : hidden width of FFN sublayer  (typically 4 × d_token)
    attn_dropout : dropout on attention weights and post-attention residual
    ffn_dropout  : dropout inside FFN and post-FFN residual
    """

    def __init__(self, n_features: int, task_type: str, n_classes: int,
                 d_token: int = 64, n_heads: int = 4,
                 n_blocks: int = 2, ffn_d_hidden: int = 128,
                 attn_dropout: float = 0.1, ffn_dropout: float = 0.1):
        super().__init__()
        self.task_type = task_type
        self.n_classes = n_classes

        self.tokenizer = FeatureTokenizer(n_features, d_token)

        # Learnable [CLS] token — shape (1, 1, d_token), broadcast over batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, ffn_d_hidden,
                             attn_dropout, ffn_dropout)
            for _ in range(n_blocks)
        ])

        self.head_norm = nn.LayerNorm(d_token)
        n_out = 1 if task_type == 'binary' else n_classes
        self.head = nn.Linear(d_token, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        tokens = self.tokenizer(x)                               # (B, F, d)
        cls    = self.cls_token.expand(x.size(0), -1, -1)       # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)                 # (B, F+1, d)

        for block in self.blocks:
            tokens = block(tokens)

        cls_out = tokens[:, 0, :]                                # (B, d)
        cls_out = self.head_norm(cls_out)
        return self.head(cls_out)                                 # (B, n_out)


# ── Training helpers ──────────────────────────────────────────────────────────

def _compute_class_weights_torch(y: np.ndarray, n_classes: int,
                                  device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency class weights for imbalanced clinical targets.

    Returns a float32 tensor on `device` of shape (n_classes,).

    For class k:
        cw[k] = total_samples / (n_classes × count_k)

    This gives a HIGHER weight to LESS frequent classes. For binary tasks the
    returned tensor has shape (2,): cw[0] for the majority class, cw[1] for
    the minority class; cw[1] > cw[0] whenever positives are the minority.
    """
    counts  = np.bincount(y.astype(int), minlength=n_classes).astype(float)
    counts  = np.maximum(counts, 1)           # guard against empty classes
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_ftt(input_dim: int, task_type: str, n_classes: int,
              d_token: int = 64, n_heads: int = 4,
              n_blocks: int = 2, ffn_d_hidden: int = 128,
              attn_dropout: float = 0.1, ffn_dropout: float = 0.1
              ) -> FTTransformerModel:
    """
    Constructs a fresh (randomly initialised) FT-Transformer.
    Called once per fold — no weight sharing between folds.
    """
    return FTTransformerModel(
        n_features=input_dim,
        task_type=task_type,
        n_classes=n_classes,
        d_token=d_token,
        n_heads=n_heads,
        n_blocks=n_blocks,
        ffn_d_hidden=ffn_d_hidden,
        attn_dropout=attn_dropout,
        ffn_dropout=ffn_dropout,
    )


def train_ftt(model: FTTransformerModel,
              X_tr: np.ndarray,
              y_tr: np.ndarray,
              device: torch.device,
              epochs: int = 50,
              batch_size: int = 256,
              lr: float = 1e-3,
              weight_decay: float = 1e-4,
              patience: int = 5,
              val_fraction: float = 0.15) -> FTTransformerModel:
    """
    Trains the FT-Transformer with early stopping.

    Parameters match the stacking MLP budget (50 epochs, patience 5) so OOF
    runtime stays tractable across 48 tournament slots.  AdamW is preferred
    over plain Adam for transformers — weight decay regularises the attention
    and FFN weights without affecting LayerNorm/bias parameters.

    Parameters
    ----------
    model        : FTTransformerModel (on CPU; moved to device internally)
    X_tr, y_tr   : scaled numpy arrays for the current fold's training split
    device       : torch.device  ('cuda' | 'cpu' | 'mps')
    epochs       : maximum training epochs
    batch_size   : mini-batch size
    lr           : learning rate for AdamW
    weight_decay : L2 regularisation coefficient (applied via AdamW)
    patience     : early-stopping patience on validation loss
    val_fraction : fraction of X_tr held out as internal validation

    Returns
    -------
    model : the trained FTTransformerModel (best weights restored)
    """
    model = model.to(device)
    n_val   = max(1, int(len(X_tr) * val_fraction))
    n_train = len(X_tr) - n_val

    # Reproducible split — same seed used everywhere in the pipeline
    rng   = np.random.default_rng(42)
    idx   = rng.permutation(len(X_tr))
    tr_i, val_i = idx[:n_train], idx[n_train:]

    X_t = torch.tensor(X_tr[tr_i],  dtype=torch.float32)
    y_t = torch.tensor(y_tr[tr_i],  dtype=torch.float32 if model.task_type == 'binary'
                                                        else torch.long)
    X_v = torch.tensor(X_tr[val_i], dtype=torch.float32, device=device)
    y_v = torch.tensor(y_tr[val_i], dtype=torch.float32 if model.task_type == 'binary'
                                                        else torch.long, device=device)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                        shuffle=True, pin_memory=(device.type == 'cuda'))

    n_cls = 2 if model.task_type == 'binary' else model.n_classes
    cw    = _compute_class_weights_torch(y_tr, n_cls, device)

    if model.task_type == 'binary':
        # FIX (Bug 1): pos_weight must be > 1 to up-weight the minority positive
        # class. _compute_class_weights_torch gives cw[1] > cw[0] when positives
        # are rare, so the correct ratio is cw[1] / cw[0] (≈ count_neg / count_pos).
        #
        # The original code had cw[0] / cw[1] which is count_pos / count_neg < 1,
        # actively DOWN-weighting the positive class — the opposite of the intent.
        pos_weight = (cw[1] / cw[0]).unsqueeze(0)   # shape (1,)  value > 1 for minority positives
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion  = nn.CrossEntropyLoss(weight=cw)

    # AdamW — separate decay groups: no decay for bias/LayerNorm parameters
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and not _no_decay(n)]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and _no_decay(n)]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params,    'weight_decay': weight_decay},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=lr
    )

    best_val_loss  = float('inf')
    best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    wait           = 0

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            if model.task_type == 'binary':
                loss = criterion(logits.squeeze(-1), y_batch)
            else:
                loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            if model.task_type == 'binary':
                val_loss = criterion(val_logits.squeeze(-1), y_v).item()
            else:
                val_loss = criterion(val_logits, y_v).item()
        model.train()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait          = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best weights
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model


def _no_decay(param_name: str) -> bool:
    """Returns True for parameters that should not receive weight decay."""
    return any(nd in param_name for nd in ('bias', 'norm', 'cls_token'))


def predict_ftt(model: FTTransformerModel,
                X: np.ndarray,
                device: torch.device,
                task_type: str,
                batch_size: int = 512) -> np.ndarray:
    """
    Returns calibrated probability predictions from a trained FT-Transformer.

    Binary     → np.ndarray shape (N,)      sigmoid-converted probability of class 1
    Multiclass → np.ndarray shape (N, C)    softmax probabilities over C classes
    """
    model.eval()
    X_t    = torch.tensor(X, dtype=torch.float32)
    # FIX (Bug 3): enable pin_memory on CUDA for faster host→device transfers,
    # consistent with the training DataLoader.
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False,
                        pin_memory=(device.type == 'cuda'))
    all_probs = []

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits  = model(X_batch)
            if task_type == 'binary':
                probs = torch.sigmoid(logits.squeeze(-1))   # (B,)
            else:
                probs = F.softmax(logits, dim=-1)           # (B, C)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def get_device() -> torch.device:
    """
    Returns the best available device: CUDA → MPS → CPU.
    MPS support is included for Apple Silicon (M-series) machines running
    the MIMIC-IV pipeline locally before cloud deployment.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
