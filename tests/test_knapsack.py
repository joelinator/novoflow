import pytest
import torch
from novoflow.reachability.knapsack import ExactReachabilityDP, get_reachable_lengths
from novoflow.constants import calc_peptide_mass, MASS_H2O


def test_reachability_dp_initialization():
    dp = ExactReachabilityDP(max_len=30, max_mass=3500.0, bin_size=0.05, tol_da=0.5)
    assert dp.dp_table is not None
    assert dp.pooled_table is not None
    assert dp.dp_table.shape[0] == 31


def test_reachability_exact_peptide():
    dp = ExactReachabilityDP(max_len=30, max_mass=3500.0, bin_size=0.02, tol_da=0.5)
    pep = "PEPTIDE"
    total_mass = calc_peptide_mass(pep)
    residue_mass = total_mass - MASS_H2O
    L = len(pep)  # 7

    # The exact residue mass must be reachable in 7 steps
    reachable = dp.is_reachable(L, residue_mass).item()
    assert reachable is True

    # Impossible mass: 100 Da cannot be formed by 15 amino acids (min mass ~ 15 * 57 = 855 Da)
    impossible = dp.is_reachable(15, 100.0).item()
    assert impossible is False


def test_get_reachable_lengths():
    dp = ExactReachabilityDP(max_len=30, max_mass=3500.0, bin_size=0.05, tol_da=0.5)
    pep = "AGSTV"
    mass = calc_peptide_mass(pep)
    valid_lens = get_reachable_lengths(mass, dp, min_len=3, max_len=10)
    assert 5 in valid_lens
