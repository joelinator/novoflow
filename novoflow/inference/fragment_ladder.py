"""
Physical Fragment Ladder Matching and Scoring Engine.
Computes theoretical b- and y-ion ladders, neutral losses, ladder continuity,
and matched spectral intensity fractions.
"""

from __future__ import annotations
import math
import torch
from novoflow.constants import (
    AA_MASSES,
    MASS_PROTON,
    MASS_H2O,
    MASS_NH3,
    MASS_CO,
    calc_peptide_mass,
)


def compute_theoretical_ions(
    peptide_seq: str,
    max_charge: int = 2,
    include_neutral_losses: bool = True,
) -> dict[str, list[float]]:
    """
    Generate all theoretical b- and y-ions for a peptide string.
    Returns dictionary with ion series keys and lists of m/z values.
    """
    residues = []
    i = 0
    n = len(peptide_seq)
    while i < n:
        if peptide_seq[i:i+5] == "M(ox)":
            residues.append("M(ox)")
            i += 5
        elif peptide_seq[i] in AA_MASSES:
            residues.append(peptide_seq[i])
            i += 1
        else:
            residues.append(peptide_seq[i])
            i += 1

    L = len(residues)
    res_masses = [AA_MASSES.get(r, 110.0) for r in residues]

    # Prefix cumulative masses
    prefix_masses = []
    curr = 0.0
    for m in res_masses:
        curr += m
        prefix_masses.append(curr)

    total_mass = prefix_masses[-1]

    b1_mz = []
    y1_mz = []
    b2_mz = []
    y2_mz = []
    b_h2o = []
    b_nh3 = []
    a_ions = []
    y_h2o = []
    y_nh3 = []

    for idx in range(L - 1):
        b_neutral = prefix_masses[idx]
        y_neutral = total_mass - prefix_masses[idx] + MASS_H2O

        # 1+ ions: [M + H]+
        b1 = b_neutral + MASS_PROTON
        y1 = y_neutral + MASS_PROTON
        b1_mz.append(b1)
        y1_mz.append(y1)

        # 2+ ions: [M + 2H]2+
        if max_charge >= 2:
            b2_mz.append((b_neutral + 2 * MASS_PROTON) / 2.0)
            y2_mz.append((y_neutral + 2 * MASS_PROTON) / 2.0)

        if include_neutral_losses:
            b_h2o.append(b1 - MASS_H2O)
            b_nh3.append(b1 - MASS_NH3)
            a_ions.append(b1 - MASS_CO)
            y_h2o.append(y1 - MASS_H2O)
            y_nh3.append(y1 - MASS_NH3)

    return {
        "b1": b1_mz,
        "y1": y1_mz,
        "b2": b2_mz,
        "y2": y2_mz,
        "b_h2o": b_h2o,
        "b_nh3": b_nh3,
        "a": a_ions,
        "y_h2o": y_h2o,
        "y_nh3": y_nh3,
        "total_mass": total_mass + MASS_H2O,
        "num_residues": L,
    }


def score_fragment_ladder(
    peptide_seq: str,
    exp_mz: list[float] | torch.Tensor,
    exp_int: list[float] | torch.Tensor,
    tolerance_ppm: float = 20.0,
    tolerance_da: float = 0.05,
) -> dict[str, float]:
    """
    Score a candidate peptide against an experimental tandem mass spectrum.
    Returns physical metrics:
        - frag_score: composite weighted score
        - explained_intensity: fraction of total MS2 intensity explained by theoretical ions
        - theo_coverage: fraction of (L-1) peptide cleavage sites covered by at least one b or y ion
        - ladder_score: fraction of consecutive cleavage pairs confirmed
        - b_matched: number of unique b-cleavage sites matched
        - y_matched: number of unique y-cleavage sites matched
        - ion_balance: balance between b and y series support
    """
    if isinstance(exp_mz, torch.Tensor):
        exp_mz = exp_mz.cpu().tolist()
    if isinstance(exp_int, torch.Tensor):
        exp_int = exp_int.cpu().tolist()

    if len(exp_mz) == 0:
        return {
            "frag_score": 0.0,
            "explained_intensity": 0.0,
            "theo_coverage": 0.0,
            "ladder_score": 0.0,
            "b_matched": 0,
            "y_matched": 0,
            "ion_balance": 0.0,
        }

    total_intensity = sum(exp_int)
    if total_intensity <= 0:
        total_intensity = 1.0

    theo = compute_theoretical_ions(peptide_seq)
    L = theo["num_residues"]
    if L <= 1:
        return {
            "frag_score": 0.0,
            "explained_intensity": 0.0,
            "theo_coverage": 0.0,
            "ladder_score": 0.0,
            "b_matched": 0,
            "y_matched": 0,
            "ion_balance": 0.0,
        }

    b_hits = [False] * (L - 1)
    y_hits = [False] * (L - 1)
    matched_peak_indices = set()

    def match_ion_list(ion_list: list[float], hit_recorder: list[bool] | None = None):
        for idx, target_mz in enumerate(ion_list):
            tol = max(tolerance_da, target_mz * tolerance_ppm * 1e-6)
            for p_i, (m, _) in enumerate(zip(exp_mz, exp_int)):
                if abs(m - target_mz) <= tol:
                    matched_peak_indices.add(p_i)
                    if hit_recorder is not None and idx < len(hit_recorder):
                        hit_recorder[idx] = True

    # Match primary b and y series
    match_ion_list(theo["b1"], b_hits)
    match_ion_list(theo["y1"], y_hits)
    # Secondary and neutral losses
    match_ion_list(theo["b2"], b_hits)
    match_ion_list(theo["y2"], y_hits)
    match_ion_list(theo["b_h2o"])
    match_ion_list(theo["y_h2o"])
    match_ion_list(theo["a"])

    matched_intensity = sum(exp_int[i] for i in matched_peak_indices)
    explained_intensity = matched_intensity / total_intensity

    # Theoretical site coverage: site is covered if either b_i or y_{L-1-i} is observed
    sites_covered = 0
    for i in range(L - 1):
        if b_hits[i] or y_hits[L - 2 - i]:
            sites_covered += 1
    theo_coverage = sites_covered / max(1, (L - 1))

    # Ladder continuity: consecutive sites
    consecutive_pairs = 0
    possible_pairs = max(1, L - 2)
    for i in range(L - 2):
        if (b_hits[i] and b_hits[i + 1]) or (y_hits[i] and y_hits[i + 1]):
            consecutive_pairs += 1
    ladder_score = consecutive_pairs / possible_pairs

    b_matched = sum(b_hits)
    y_matched = sum(y_hits)
    total_matches = b_matched + y_matched
    if total_matches > 0:
        ion_balance = 1.0 - abs(b_matched - y_matched) / total_matches
    else:
        ion_balance = 0.0

    # Composite physical ladder score
    frag_score = (
        0.40 * theo_coverage
        + 0.35 * ladder_score
        + 0.15 * explained_intensity
        + 0.10 * ion_balance
    )

    return {
        "frag_score": frag_score,
        "explained_intensity": explained_intensity,
        "theo_coverage": theo_coverage,
        "ladder_score": ladder_score,
        "b_matched": b_matched,
        "y_matched": y_matched,
        "ion_balance": ion_balance,
    }
