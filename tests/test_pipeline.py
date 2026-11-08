import pytest
import torch
from novoflow.pipeline import NovoFlowPipeline
from novoflow.io.mgf import MGFSpectrum
from novoflow.constants import calc_peptide_mass, calc_precursor_mz
from novoflow.inference.fragment_ladder import compute_theoretical_ions


def test_pipeline_execution_cpu():
    # Construct synthetic spectrum for a known peptide
    pep = "PEPTIDE"
    mass = calc_peptide_mass(pep)
    charge = 2
    mz = calc_precursor_mz(mass, charge)
    theo = compute_theoretical_ions(pep)

    spec = MGFSpectrum(
        title="Synthetic_PEPTIDE",
        precursor_mz=mz,
        charge=charge,
        mz_array=theo["b1"] + theo["y1"],
        intensity_array=[50.0] * (len(theo["b1"]) + len(theo["y1"])),
        rt=35.0,
        scan_num=1,
    )

    # Initialize CPU pipeline with 3 flow steps for fast test execution
    pipeline = NovoFlowPipeline(
        device="cpu",
        flow_steps=3,
        top_k_lengths=2,
        samples_per_length=1,
    )

    results = pipeline.predict_spectra([spec])
    assert len(results) == 1
    res = results[0]
    assert "sequence" in res
    assert len(res["sequence"]) > 0
    assert "novoflow_score" in res
    assert "q_value" in res
    assert "pep" in res
