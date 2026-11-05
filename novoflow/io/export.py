"""
Export Identifications to Standard Proteomics Formats.
Supports TSV, CSV, and mzTab (HUPO-PSI specification).
"""

from __future__ import annotations
import csv
import os
from typing import Any
from novoflow.constants import calc_peptide_mass, MASS_H2O


STANDARD_COLUMNS = [
    "raw_file",
    "scan_num",
    "title",
    "retention_time_sec",
    "precursor_mz",
    "charge",
    "precursor_neutral_mass",
    "peptide_sequence",
    "calculated_neutral_mass",
    "mass_error_da",
    "mass_error_ppm",
    "novoflow_score",
    "pep",
    "q_value",
    "passes_1pct_fdr",
    "frag_score",
    "theo_coverage",
    "ladder_score",
    "explained_intensity",
    "b_ions_matched",
    "y_ions_matched",
]


def export_tsv(filepath: str, identifications: list[dict[str, Any]]):
    """Write identifications to a tab-separated TSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(STANDARD_COLUMNS)
        for r in identifications:
            row = [
                r.get("raw_file", ""),
                r.get("scan_num", 0),
                r.get("title", ""),
                f"{r.get('rt', 0.0):.2f}",
                f"{r.get('precursor_mz', 0.0):.6f}",
                r.get("charge", 2),
                f"{r.get('precursor_neutral_mass', 0.0):.4f}",
                r.get("sequence", ""),
                f"{r.get('calc_mass', 0.0):.4f}",
                f"{r.get('delta_m', 0.0):.4f}",
                f"{r.get('ppm_error', 0.0):.2f}",
                f"{r.get('novoflow_score', 0.0):.4f}",
                f"{r.get('pep', 1.0):.6f}",
                f"{r.get('q_value', 1.0):.6f}",
                "TRUE" if r.get("q_value", 1.0) <= 0.01 else "FALSE",
                f"{r.get('frag_score', 0.0):.4f}",
                f"{r.get('theo_coverage', 0.0):.4f}",
                f"{r.get('ladder_score', 0.0):.4f}",
                f"{r.get('explained_intensity', 0.0):.4f}",
                r.get("b_matched", 0),
                r.get("y_matched", 0),
            ]
            writer.writerow(row)


def export_csv(filepath: str, identifications: list[dict[str, Any]]):
    """Write identifications to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(STANDARD_COLUMNS)
        for r in identifications:
            row = [
                r.get("raw_file", ""),
                r.get("scan_num", 0),
                r.get("title", ""),
                f"{r.get('rt', 0.0):.2f}",
                f"{r.get('precursor_mz', 0.0):.6f}",
                r.get("charge", 2),
                f"{r.get('precursor_neutral_mass', 0.0):.4f}",
                r.get("sequence", ""),
                f"{r.get('calc_mass', 0.0):.4f}",
                f"{r.get('delta_m', 0.0):.4f}",
                f"{r.get('ppm_error', 0.0):.2f}",
                f"{r.get('novoflow_score', 0.0):.4f}",
                f"{r.get('pep', 1.0):.6f}",
                f"{r.get('q_value', 1.0):.6f}",
                "TRUE" if r.get("q_value", 1.0) <= 0.01 else "FALSE",
                f"{r.get('frag_score', 0.0):.4f}",
                f"{r.get('theo_coverage', 0.0):.4f}",
                f"{r.get('ladder_score', 0.0):.4f}",
                f"{r.get('explained_intensity', 0.0):.4f}",
                r.get("b_matched", 0),
                r.get("y_matched", 0),
            ]
            writer.writerow(row)


def export_mztab(filepath: str, identifications: list[dict[str, Any]], sample_name: str = "NovoFlow_Run"):
    """
    Write identifications to HUPO-PSI mzTab format.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        # Metadata section
        f.write("MTD\tmzTab-version\t1.0.0\n")
        f.write("MTD\tmzTab-mode\tSummary\n")
        f.write("MTD\tmzTab-type\tIdentification\n")
        f.write(f"MTD\tdescription\tNovoFlow De Novo Peptide Sequencing ({sample_name})\n")
        f.write("MTD\tpsm_search_engine_score[1]\t[MS, MS:1001475, NovoFlow Score, ]\n")
        f.write("MTD\tpsm_search_engine_score[2]\t[MS, MS:1001491, NovoFlow PEP, ]\n")
        f.write("MTD\tpsm_search_engine_score[3]\t[MS, MS:1001492, NovoFlow q-value, ]\n")

        # PSM section header
        psm_header = [
            "sequence",
            "PSM_ID",
            "accession",
            "unique",
            "database",
            "database_version",
            "search_engine",
            "search_engine_score[1]",
            "search_engine_score[2]",
            "search_engine_score[3]",
            "charge",
            "exp_mass_to_charge",
            "calc_mass_to_charge",
            "spectra_ref",
            "pre",
            "post",
            "start",
            "end",
        ]
        f.write("PSH\t" + "\t".join(psm_header) + "\n")

        # PSM data rows
        for idx, r in enumerate(identifications, start=1):
            exp_mz = r.get("precursor_mz", 0.0)
            charge = r.get("charge", 2)
            calc_mass = r.get("calc_mass", 0.0)
            calc_mz = (calc_mass + charge * 1.007276) / charge if charge > 0 else 0.0

            psm_row = [
                r.get("sequence", "UNKNOWN"),
                str(idx),
                "null",
                "null",
                "null",
                "null",
                "[MS, MS:1001456, NovoFlow, ]",
                f"{r.get('novoflow_score', 0.0):.4f}",
                f"{r.get('pep', 1.0):.6f}",
                f"{r.get('q_value', 1.0):.6f}",
                str(charge),
                f"{exp_mz:.6f}",
                f"{calc_mz:.6f}",
                f"ms_run[1]:{r.get('title', str(idx))}",
                "-",
                "-",
                "null",
                "null",
            ]
            f.write("PSM\t" + "\t".join(psm_row) + "\n")


def export_identifications(
    identifications: list[dict[str, Any]],
    filepath: str,
    format: str = "tsv",
    sample_name: str = "NovoFlow_Run",
):
    """
    Export identifications to tsv, csv, or mztab based on format or file extension.
    """
    fmt = format.lower().strip(".")
    if fmt == "tsv" or filepath.endswith(".tsv"):
        export_tsv(filepath, identifications)
    elif fmt == "csv" or filepath.endswith(".csv"):
        export_csv(filepath, identifications)
    elif fmt in ("mztab", "mztab.txt") or filepath.endswith(".mztab"):
        export_mztab(filepath, identifications, sample_name=sample_name)
    else:
        export_tsv(filepath, identifications)


