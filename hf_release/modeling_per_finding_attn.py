"""Standalone model definition for the 96slices_per_finding_attn checkpoints (no repo code needed).

A 2D backbone reads every 3-slice window of a study; a per-finding attention head pools the windows into one
vector per finding and reads out 12 logits (one per abnormality, apply sigmoid for probabilities).

    model = load_model("path/to/convnext_base_v5_96slices_per_finding_attn")
    logits = model(x)   # x: (B * 94, 3, 384, 384), the 94 windows of each study in order
"""
import json
from pathlib import Path

import timm
import torch
import torch.nn as nn
from safetensors.torch import load_file


class PerFindingAttnModel(nn.Module):
    """Backbone over a bag of windows, softmax attention over the windows per finding -> 12 logits.

    The layer names (backbone, norm, att, clsW, clsb) must match the keys in model.safetensors, so they stay as
    they were in training."""

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg["model"]
        window_cfg = cfg["input"]["window_bag"]

        # the image model that turns one 3-slice window into a feature vector (no classifier on top)
        self.backbone = timm.create_model(model_cfg["name"], pretrained=False, num_classes=0,
                                          in_chans=3, drop_rate=model_cfg.get("drop_rate", 0.0))
        feature_dim = self.backbone.num_features       # length of each window's feature vector (1024)
        num_findings = len(cfg["tasks"])               # 12

        # how many windows each study is cut into: a random 12 in training, all 94 at inference
        self.windows_train = int(window_cfg["train_k"])
        self.windows_eval = int(window_cfg["eval_k"])
        self.windows_per_study = self.windows_eval     # current mode; train() / eval() switch it

        self.norm = nn.LayerNorm(feature_dim)          # normalises each window's features
        # att: scores how relevant each window is, separately for each finding
        self.att = nn.Sequential(nn.Linear(feature_dim, 256), nn.Tanh(), nn.Dropout(0.2),
                                 nn.Linear(256, num_findings))
        # clsW / clsb: one linear read-out per finding (weights and bias)
        self.clsW = nn.Parameter(torch.zeros(num_findings, feature_dim))
        self.clsb = nn.Parameter(torch.zeros(num_findings))

    def train(self, mode: bool = True):
        """PyTorch calls this for model.train() and model.eval(); also switch the windows per study."""
        self.windows_per_study = self.windows_train if mode else self.windows_eval
        return super().train(mode)

    def forward(self, windows):
        """windows: (num_studies * windows_per_study, 3, H, W), each study's windows in order."""
        features = self.backbone(windows)                                   # (studies*windows, F)
        # regroup by study, then normalise:
        # view: (studies*windows, F) -> (studies, windows, F), e.g. (188, 1024) -> (2, 94, 1024) for 2 studies
        features = self.norm(features.view(-1, self.windows_per_study, features.shape[-1]))
        # per finding, a softmax over the windows: weights that sum to 1 for each finding
        attention = torch.softmax(self.att(features), dim=1)                # (studies, windows, 12)
        # weighted average of the window features, one vector per finding
        pooled = torch.einsum("bkn,bkf->bnf", attention, features)          # (studies, 12, F)
        # each finding's own linear read-out -> one logit per finding
        # 12 separate Linear(F -> 1), one per finding:
        # (studies, 12, F) * (12, F) -> (studies, 12, F) -> sum -> (studies, 12) + (12) -> (studies, 12)
        return (pooled * self.clsW).sum(-1) + self.clsb


def build_model(cfg: dict) -> nn.Module:
    """The model with random weights, from a config dict (load_model below also loads the weights)."""
    return PerFindingAttnModel(cfg)


def load_model(folder, device="cpu"):
    """Build the model from config.json and load model.safetensors from the same folder."""
    folder = Path(folder)
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model = build_model(cfg)
    model.load_state_dict(load_file(folder / "model.safetensors"))   # strict: every weight must match
    return model.to(device).eval()                                    # eval(): inference mode, 94 windows
