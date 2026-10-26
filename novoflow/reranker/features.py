"""
Candidate Feature Extraction for Reranking.
Extracts spectral, physical mass, and flow likelihood features for peptide candidates.
"""

from __future__ import annotations
import math
from novoflow.constants import calc_peptide_mass, MASS_H2O


FEATURE_NAMES = [
    "cand_length_log_prob",
    "mean_log_prob",
    "total_log_prob",
    "frag_score",
    "explained_intensity",
    "ladder_score",
    "theo_coverage",
    "ion_balance",
    "unmatched_penalty",
    "term_bonus",
    "ppm_error",
    "mass_gate_20ppm",
    "mass_gate_10ppm",
    "mass_gaussian",
    "len_diff_expected",
    "cand_length",
]


def extract_features(
    candidate_seq: str,
    precursor_mass: float,
    expected_length: float,
    length_log_prob: float = 0.0,
    mean_log_prob: float = 0.0,
    total_log_prob: float = 0.0,
    frag_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Extract 16 discriminative physical and probabilistic features from a candidate.
    """
    if frag_metrics is None:
        frag_metrics = {
            "frag_score": 0.0,
            "explained_intensity": 0.0,
            "ladder_score": 0.0,
            "theo_coverage": 0.0,
            "ion_balance": 0.0,
        }

    cand_len = len(candidate_seq)
    calc_mass = calc_peptide_mass(candidate_seq)
    delta_m = calc_mass - precursor_mass
    ppm = (abs(delta_m) / max(100.0, precursor_mass)) * 1e6

    mass_gate_20 = 1.0 if ppm <= 20.0 else 0.0
    mass_gate_10 = 1.0 if ppm <= 10.0 else 0.0
    mass_gaussian = math.exp(-0.5 * min(100.0, (ppm / 10.0) ** 2))

    # Terminal cleavage bonus: typical tryptic peptides end in K or R
    term_bonus = 1.0 if candidate_seq and candidate_seq[-1] in ("K", "R") else 0.0

    return {
        "cand_length_log_prob": float(length_log_prob),
        "mean_log_prob": float(mean_log_prob),
        "total_log_prob": float(total_log_prob),
        "frag_score": float(frag_metrics.get("frag_score", 0.0)),
        "explained_intensity": float(frag_metrics.get("explained_intensity", 0.0)),
        "ladder_score": float(frag_metrics.get("ladder_score", 0.0)),
        "theo_coverage": float(frag_metrics.get("theo_coverage", 0.0)),
        "ion_balance": float(frag_metrics.get("ion_balance", 0.0)),
        "unmatched_penalty": float(1.0 - frag_metrics.get("explained_intensity", 0.0)),
        "term_bonus": float(term_bonus),
        "ppm_error": float(ppm),
        "mass_gate_20ppm": float(mass_gate_20),
        "mass_gate_10ppm": float(mass_gate_10),
        "mass_gaussian": float(mass_gaussian),
        "len_diff_expected": float(abs(cand_len - expected_length)),
        "cand_length": float(cand_len),
    }
