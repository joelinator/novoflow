"""
Mascot Generic Format (MGF) File I/O.
Provides streaming, memory-efficient reading and writing of tandem mass spectra.
"""

from __future__ import annotations
import os
from typing import Generator, Any
from novoflow.constants import MASS_PROTON


class MGFSpectrum:
    """Represents a single tandem mass spectrum parsed from an MGF file."""
    def __init__(
        self,
        title: str,
        precursor_mz: float,
        charge: int,
        mz_array: list[float],
        intensity_array: list[float],
        rt: float = 0.0,
        scan_num: int = 0,
        raw_file: str = "",
    ):
        self.title = title
        self.precursor_mz = precursor_mz
        self.charge = max(1, charge)
        self.mz_array = mz_array
        self.intensity_array = intensity_array
        self.rt = rt
        self.scan_num = scan_num
        self.raw_file = raw_file
        # Precursor neutral mass = (m/z * z) - (z * MASS_PROTON)
        self.precursor_neutral_mass = (self.precursor_mz * self.charge) - (self.charge * MASS_PROTON)


def read_mgf(filepath: str) -> Generator[MGFSpectrum, None, None]:
    """
    Stream spectra from an MGF file one by one.
    Memory-efficient: does not load entire file into RAM.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MGF file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        in_ions = False
        title = ""
        pepmass = 0.0
        charge = 2
        rt = 0.0
        scans = 0
        mz_list = []
        int_list = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == "BEGIN IONS":
                in_ions = True
                title = ""
                pepmass = 0.0
                charge = 2
                rt = 0.0
                scans = 0
                mz_list = []
                int_list = []
            elif line == "END IONS":
                if in_ions and len(mz_list) > 0 and pepmass > 0:
                    yield MGFSpectrum(
                        title=title or f"Scan_{scans}",
                        precursor_mz=pepmass,
                        charge=charge,
                        mz_array=mz_list,
                        intensity_array=int_list,
                        rt=rt,
                        scan_num=scans,
                        raw_file=os.path.basename(filepath),
                    )
                in_ions = False
            elif in_ions:
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.upper()
                    if key == "TITLE":
                        title = val
                    elif key == "PEPMASS":
                        parts = val.split()
                        pepmass = float(parts[0])
                    elif key == "CHARGE":
                        val_clean = val.replace("+", "").replace("-", "")
                        try:
                            charge = int(val_clean)
                        except ValueError:
                            charge = 2
                    elif key == "RTINSECONDS":
                        try:
                            rt = float(val)
                        except ValueError:
                            rt = 0.0
                    elif key == "SCANS":
                        try:
                            scans = int(val)
                        except ValueError:
                            scans = 0
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            m = float(parts[0])
                            i = float(parts[1])
                            mz_list.append(m)
                            int_list.append(i)
                        except ValueError:
                            pass


def read_mgf_batches(
    filepath: str,
    batch_size: int = 128,
) -> Generator[list[MGFSpectrum], None, None]:
    """Stream spectra from an MGF file in batches of batch_size."""
    batch = []
    for spec in read_mgf(filepath):
        batch.append(spec)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_mgf(filepath: str, spectra: list[MGFSpectrum]):
    """Write a list of MGFSpectrum objects to an MGF file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for spec in spectra:
            f.write("BEGIN IONS\n")
            f.write(f"TITLE={spec.title}\n")
            f.write(f"PEPMASS={spec.precursor_mz:.6f}\n")
            f.write(f"CHARGE={spec.charge}+\n")
            if spec.rt > 0:
                f.write(f"RTINSECONDS={spec.rt:.2f}\n")
            if spec.scan_num > 0:
                f.write(f"SCANS={spec.scan_num}\n")
            for m, i in zip(spec.mz_array, spec.intensity_array):
                f.write(f"{m:.4f} {i:.2f}\n")
            f.write("END IONS\n\n")
