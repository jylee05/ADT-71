import math
from dataclasses import dataclass

import torch
import torch.nn as nn


class SequencePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self._build(max_len)

    def _build(self, max_len: int) -> None:
        pe = torch.zeros(max_len, self.d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        if hasattr(self, "pe"):
            self.pe = pe
        else:
            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            self._build(x.size(1) * 2)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)


@dataclass
class RefinerConfig:
    drum_channels: int = 7
    feature_dim: int = 2
    hidden_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1

    # clean/noisy adaptive gate
    gate_init_bias: float = -1.0


class DrumRefiner(nn.Module):
    """Context-aware edit refiner.

    Input:
      fm_logits: (B, T, C*2), range usually in [-1, 1]
      cond_feats: (B, T, F) optional audio condition features
    Output:
      dict with
        - edit_logits: (B, T, C, 3) for keep/delete/add
        - vel_residual: (B, T, C)
        - gate: (B, 1, 1) adaptive strength scalar in [0,1]
    """

    def __init__(self, cfg: RefinerConfig, cond_dim: int = 0):
        super().__init__()
        self.cfg = cfg
        self.channels = cfg.drum_channels
        self.input_dim = cfg.drum_channels * cfg.feature_dim
        merged_in = self.input_dim + cond_dim

        self.in_proj = nn.Linear(merged_in, cfg.hidden_dim)
        self.pos = SequencePositionalEncoding(cfg.hidden_dim, max_len=4096, dropout=cfg.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

        self.edit_head = nn.Linear(cfg.hidden_dim, self.channels * 3)  # keep/delete/add
        self.vel_head = nn.Linear(cfg.hidden_dim, self.channels)  # residual for velocity only

        self.gate_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
        nn.init.constant_(self.gate_head[-1].bias, cfg.gate_init_bias)

    def forward(self, fm_logits: torch.Tensor, cond_feats: torch.Tensor | None = None) -> dict:
        if cond_feats is not None:
            x = torch.cat([fm_logits, cond_feats], dim=-1)
        else:
            x = fm_logits

        h = self.in_proj(x)
        h = self.pos(h)
        h = self.encoder(h)

        edit_logits = self.edit_head(h).view(h.size(0), h.size(1), self.channels, 3)
        vel_residual = self.vel_head(h)

        pooled = h.mean(dim=1)
        gate = torch.sigmoid(self.gate_head(pooled)).unsqueeze(1)  # (B,1,1)

        return {
            "edit_logits": edit_logits,
            "vel_residual": vel_residual,
            "gate": gate,
        }


def make_edit_labels(base_onset: torch.Tensor, target_onset: torch.Tensor) -> torch.Tensor:
    """Create keep/delete/add labels.

    base_onset, target_onset: (B, T, C) in {-1, +1} (or thresholded values)
    Returns labels in {0:keep, 1:delete, 2:add}
    """
    base_hit = (base_onset > 0).long()
    tgt_hit = (target_onset > 0).long()

    keep = (base_hit == tgt_hit)
    delete = (base_hit == 1) & (tgt_hit == 0)

    labels = torch.zeros_like(base_hit)
    labels[delete] = 1
    labels[~keep & ~delete] = 2
    return labels


def apply_edits(
    base_logits: torch.Tensor,
    edit_logits: torch.Tensor,
    vel_residual: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Apply keep/delete/add edit decisions + velocity residual.

    base_logits: (B,T,C*2) interleaved [on,vel]
    returns refined logits with same shape.
    """
    b, t, d = base_logits.shape
    c = d // 2
    view = base_logits.view(b, t, c, 2)

    base_on = view[..., 0]
    base_vel = view[..., 1]

    edit_cls = edit_logits.argmax(dim=-1)

    # keep: retain base onset score
    # delete: force off
    # add: force on
    on_delete = torch.full_like(base_on, -1.0)
    on_add = torch.full_like(base_on, 1.0)

    edited_on = torch.where(edit_cls == 1, on_delete, base_on)
    edited_on = torch.where(edit_cls == 2, on_add, edited_on)

    # adaptive residual blend
    refined_on = base_on + gate * (edited_on - base_on)
    refined_vel = torch.clamp(base_vel + gate.squeeze(1) * vel_residual, -1.0, 1.0)

    out = torch.stack([refined_on, refined_vel], dim=-1).reshape(b, t, d)
    return out
