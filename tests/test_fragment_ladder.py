import pytest
from novoflow.inference.fragment_ladder import compute_theoretical_ions, score_fragment_ladder
from novoflow.constants import calc_peptide_mass, MASS_PROTON, AA_MASSES


def test_theoretical_ions():
    pep = "PEPTIDE"
    theo = compute_theoretical_ions(pep)
    assert theo["num_residues"] == 7
    assert len(theo["b1"]) == 6  # 6 b-ions for 7 residues
    assert len(theo["y1"]) == 6  # 6 y-ions for 7 residues

    # First b-ion should be P + MASS_PROTON
    expected_b1 = AA_MASSES["P"] + MASS_PROTON
    assert abs(theo["b1"][0] - expected_b1) < 1e-4


def test_score_fragment_ladder_perfect_spectrum():
    pep = "PEPTIDE"
    theo = compute_theoretical_ions(pep)

    # Construct synthetic spectrum with all b1 and y1 ions
    syn_mz = theo["b1"] + theo["y1"]
    syn_int = [100.0] * len(syn_mz)

    scores = score_fragment_ladder(pep, syn_mz, syn_int, tolerance_da=0.05)
    assert scores["theo_coverage"] == 1.0
    assert scores["ladder_score"] == 1.0
    assert scores["explained_intensity"] == 1.0
    assert scores["frag_score"] > 0.85
