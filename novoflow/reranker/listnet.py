"""
Learned Permutation Softmax (ListNet) Reranker.
Reranks multi-candidate beams using physical fragment ladder evidence, mass error, and flow log-probabilities.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from novoflow.reranker.features import FEATURE_NAMES, extract_features


DEFAULT_LINEAR_WEIGHTS: dict[str, float] = {
    "cand_length_log_prob": 0.12,
    "mean_log_prob": 1.02,
    "total_log_prob": 0.02,
    "frag_score": 4.02,
    "explained_intensity": 0.02,
    "ladder_score": 2.02,
    "theo_coverage": 1.52,
    "ion_balance": -0.02,
    "unmatched_penalty": -0.02,
    "term_bonus": 1.02,
    "ppm_error": -0.07,
    "mass_gate_20ppm": 2.02,
    "mass_gate_10ppm": 0.02,
    "mass_gaussian": 1.02,
    "len_diff_expected": -0.02,
    "cand_length": -0.02,
}


class ListNetReranker:
    """
    Linear ListNet permutation scoring model.
    Computes candidate scores s = w^T x.
    """
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(weights) if weights is not None else dict(DEFAULT_LINEAR_WEIGHTS)
        self.feature_names = FEATURE_NAMES

    def score_features(self, feat_dict: dict[str, float]) -> float:
        """Compute scalar score for a single feature dictionary."""
        score = 0.0
        for name, w in self.weights.items():
            score += w * feat_dict.get(name, 0.0)
        return float(score)

    def rerank_candidates(
        self,
        candidate_records: list[dict],
    ) -> list[dict]:
        """
        Rerank a list of candidate dictionaries for a single spectrum.
        Each candidate dict must contain 'sequence' and optionally features.
        Returns candidates sorted in descending order of score.
        """
        scored = []
        for cand in candidate_records:
            if "features" in cand:
                s = self.score_features(cand["features"])
            else:
                s = float(cand.get("score", 0.0))
            cand_copy = dict(cand)
            cand_copy["novoflow_score"] = s
            scored.append(cand_copy)

        scored.sort(key=lambda x: x["novoflow_score"], reverse=True)
        return scored

    def fit(
        self,
        features_list: list[list[dict[str, float]]],
        labels_list: list[list[float]],
        epochs: int = 15,
        lr: float = 0.01,
        device: str = "cpu",
    ):
        """
        Fit linear weights using ListNet Top-1 cross-entropy loss.
        features_list: list of spectra, each with K candidate feature dictionaries.
        labels_list: list of spectra, each with K binary 1.0/0.0 match labels.
        """
        n_feat = len(self.feature_names)
        linear = nn.Linear(n_feat, 1, bias=False).to(device)
        with torch.no_grad():
            w_init = torch.tensor([self.weights[fn] for fn in self.feature_names], dtype=torch.float32, device=device)
            linear.weight.copy_(w_init.unsqueeze(0))

        optimizer = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=1e-4)

        X_data = []
        Y_data = []
        for feats, labels in zip(features_list, labels_list):
            if len(feats) == 0:
                continue
            x_arr = [[f[name] for name in self.feature_names] for f in feats]
            X_data.append(torch.tensor(x_arr, dtype=torch.float32, device=device))
            Y_data.append(torch.tensor(labels, dtype=torch.float32, device=device))

        linear.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in zip(X_data, Y_data):
                if y.sum() <= 0:
                    continue
                optimizer.zero_grad()
                preds = linear(x).squeeze(-1)
                p_pred = F.softmax(preds, dim=-1)
                p_true = F.softmax(y * 5.0, dim=-1)
                loss = -(p_true * torch.log(p_pred + 1e-12)).sum()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        with torch.no_grad():
            fitted_w = linear.weight.squeeze(0).cpu().tolist()
            for fn, w in zip(self.feature_names, fitted_w):
                self.weights[fn] = w
