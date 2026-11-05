import os
import tempfile
import pytest
from novoflow.io.mgf import MGFSpectrum, read_mgf, write_mgf
from novoflow.io.export import export_tsv, export_csv, export_mztab


def test_mgf_roundtrip():
    spec = MGFSpectrum(
        title="Test_Scan_1",
        precursor_mz=500.25,
        charge=2,
        mz_array=[150.1, 250.2, 350.3],
        intensity_array=[10.0, 50.0, 20.0],
        rt=120.5,
        scan_num=101,
    )

    with tempfile.NamedTemporaryFile(suffix=".mgf", delete=False, mode="w") as tmp:
        tmp_path = tmp.name

    try:
        write_mgf(tmp_path, [spec])
        read_back = list(read_mgf(tmp_path))
        assert len(read_back) == 1
        s = read_back[0]
        assert s.title == "Test_Scan_1"
        assert abs(s.precursor_mz - 500.25) < 1e-4
        assert s.charge == 2
        assert len(s.mz_array) == 3
        assert len(s.intensity_array) == 3
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_export_formats():
    identifications = [
        {
            "raw_file": "sample.raw",
            "scan_num": 100,
            "title": "scan_100",
            "rt": 12.3,
            "precursor_mz": 450.2,
            "charge": 2,
            "precursor_neutral_mass": 898.385,
            "sequence": "PEPTIDE",
            "calc_mass": 799.3599,
            "delta_m": 0.001,
            "ppm_error": 1.2,
            "novoflow_score": 8.5,
            "pep": 0.002,
            "q_value": 0.001,
            "frag_score": 0.92,
            "theo_coverage": 1.0,
            "ladder_score": 0.8,
            "explained_intensity": 0.75,
            "b_matched": 6,
            "y_matched": 5,
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = os.path.join(tmpdir, "results.tsv")
        csv_path = os.path.join(tmpdir, "results.csv")
        mztab_path = os.path.join(tmpdir, "results.mztab")

        export_tsv(tsv_path, identifications)
        export_csv(csv_path, identifications)
        export_mztab(mztab_path, identifications)

        assert os.path.exists(tsv_path) and os.path.getsize(tsv_path) > 0
        assert os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        assert os.path.exists(mztab_path) and os.path.getsize(mztab_path) > 0
