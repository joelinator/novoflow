"""
mzML File I/O.
Reads tandem mass spectrometry runs from standard .mzML files.
Supports pyteomics with a built-in XML/base64 streaming fallback.
"""

from __future__ import annotations
import base64
import os
import struct
import zlib
from typing import Generator
from novoflow.constants import MASS_PROTON
from novoflow.io.mgf import MGFSpectrum


def read_mzml(filepath: str) -> Generator[MGFSpectrum, None, None]:
    """
    Stream MS2 spectra from an mzML file.
    Prefers pyteomics if installed, falls back to native streaming XML parser.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"mzML file not found: {filepath}")

    try:
        import pyteomics.mzml as p_mzml
        for spec_dict in p_mzml.read(filepath):
            # Only process MS2 spectra
            ms_level = spec_dict.get("ms level", 0)
            if ms_level != 2:
                continue

            precursors = spec_dict.get("precursorList", {}).get("precursor", [])
            if not precursors:
                continue

            ion_selection = precursors[0].get("selectedIonList", {}).get("selectedIon", [{}])[0]
            precursor_mz = float(ion_selection.get("selected ion m/z", 0.0))
            charge = int(ion_selection.get("charge state", 2))

            if precursor_mz <= 0:
                continue

            scan_title = spec_dict.get("id", "")
            rt = float(spec_dict.get("scanList", {}).get("scan", [{}])[0].get("scan start time", 0.0))
            # If RT in minutes, convert to seconds
            if rt < 300.0 and rt > 0:
                rt = rt * 60.0

            mz_arr = spec_dict.get("m/z array", [])
            int_arr = spec_dict.get("intensity array", [])

            if len(mz_arr) == 0:
                continue

            yield MGFSpectrum(
                title=scan_title,
                precursor_mz=precursor_mz,
                charge=charge,
                mz_array=[float(x) for x in mz_arr],
                intensity_array=[float(x) for x in int_arr],
                rt=rt,
                raw_file=os.path.basename(filepath),
            )
        return
    except ImportError:
        pass

    # Native XML streaming parser fallback
    import xml.etree.ElementTree as ET

    for event, elem in ET.iterparse(filepath, events=("end",)):
        if elem.tag.endswith("spectrum"):
            # Check ms level
            ms_level = 0
            for cv in elem.findall(".//{*}cvParam"):
                if cv.get("name") == "ms level":
                    try:
                        ms_level = int(cv.get("value", 0))
                    except ValueError:
                        pass

            if ms_level == 2:
                scan_title = elem.get("id", "")
                precursor_mz = 0.0
                charge = 2
                rt = 0.0

                for cv in elem.findall(".//{*}selectedIon/{*}cvParam"):
                    if cv.get("name") == "selected ion m/z":
                        precursor_mz = float(cv.get("value", 0.0))
                    elif cv.get("name") == "charge state":
                        charge = int(cv.get("value", 2))

                for cv in elem.findall(".//{*}scan/{*}cvParam"):
                    if cv.get("name") == "scan start time":
                        rt = float(cv.get("value", 0.0))
                        unit = cv.get("unitName", "").lower()
                        if "min" in unit:
                            rt *= 60.0

                # Binary data arrays
                mz_arr = []
                int_arr = []
                for b_arr in elem.findall(".//{*}binaryDataArray"):
                    is_mz = False
                    is_int = False
                    is_zlib = False
                    dtype = "f"  # 32-bit float

                    for cv in b_arr.findall(".//{*}cvParam"):
                        name = cv.get("name", "")
                        if name == "m/z array":
                            is_mz = True
                        elif name == "intensity array":
                            is_int = True
                        elif name == "zlib compression":
                            is_zlib = True
                        elif name == "64-bit float":
                            dtype = "d"

                    b_data = b_arr.find(".//{*}binary")
                    if b_data is not None and b_data.text:
                        raw_bytes = base64.b64decode(b_data.text.strip())
                        if is_zlib:
                            try:
                                raw_bytes = zlib.decompress(raw_bytes)
                            except Exception:
                                pass
                        elem_size = struct.calcsize(f"<{dtype}")
                        count = len(raw_bytes) // elem_size
                        vals = struct.unpack(f"<{count}{dtype}", raw_bytes[: count * elem_size])
                        if is_mz:
                            mz_arr = list(vals)
                        elif is_int:
                            int_arr = list(vals)

                if precursor_mz > 0 and len(mz_arr) > 0:
                    yield MGFSpectrum(
                        title=scan_title,
                        precursor_mz=precursor_mz,
                        charge=charge,
                        mz_array=mz_arr,
                        intensity_array=int_arr,
                        rt=rt,
                        raw_file=os.path.basename(filepath),
                    )

            elem.clear()
