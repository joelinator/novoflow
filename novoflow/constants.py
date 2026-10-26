"""
NovoFlow Constants & Mass Spectrometry Definitions.
Contains monoisotopic masses for canonical amino acids and common PTMs matching MassIVE-KB.
"""

from __future__ import annotations

# Physical Constants (Monoisotopic, Da)
MASS_PROTON = 1.00727646688
MASS_H2O = 18.010564684
MASS_NH3 = 17.026549101
MASS_CO = 27.99491462

# Terminal mass additions for neutral peptides:
# Neutral peptide mass = sum(residue masses) + MASS_H2O
MASS_N_TERM = 1.007825032  # H-
MASS_C_TERM = 17.002739652  # -OH

# Standard Monoisotopic Amino Acid Residue and PTM Masses (Da) from Unimod / NIST
AA_MASSES: dict[str, float] = {
    # 20 Canonical Amino Acids (Cysteine with Carbamidomethylation)
    "A": 71.0371138,
    "C": 160.0306482,  # Carbamidomethylated Cysteine (+57.021464 Da, UNIMOD:4)
    "D": 115.0269431,
    "E": 129.0425931,
    "F": 147.0684139,
    "G": 57.0214637,
    "H": 137.0589119,
    "I": 113.0840640,
    "K": 128.0949630,
    "L": 113.0840640,
    "M": 131.0404846,
    "N": 114.0429274,
    "P": 97.0527639,
    "Q": 128.0585775,
    "R": 156.1011110,
    "S": 87.0320284,
    "T": 101.0476785,
    "V": 99.0684139,
    "W": 186.0793130,
    "Y": 163.0633286,
    # Primary Residue-Level PTMs
    "M(ox)": 147.0353992,    # Methionine Oxidation (+15.9949 Da, UNIMOD:35)
    "C(cam)": 160.0306482,   # Carboxyamidomethylation (+57.0215 Da, UNIMOD:4)
    "N(deam)": 115.0269434,  # Asparagine Deamidation (+0.9840 Da, UNIMOD:7)
    "Q(deam)": 129.0425935,  # Glutamine Deamidation (+0.9840 Da, UNIMOD:7)
    "S(ph)": 166.9983594,    # Serine Phosphorylation (+79.9663 Da, UNIMOD:21)
    "T(ph)": 181.0140095,    # Threonine Phosphorylation (+79.9663 Da, UNIMOD:21)
    "Y(ph)": 243.0296596,    # Tyrosine Phosphorylation (+79.9663 Da, UNIMOD:21)
    # N-Terminal Modifications
    "(+42.01)": 42.010565,   # N-term Acetylation (+42.0106 Da, UNIMOD:1)
    "(+43.01)": 43.005814,   # N-term Carbamylation (+43.0058 Da, UNIMOD:5)
    "(-17.03)": -17.026549,  # N-term Ammonia Loss (-17.0265 Da, UNIMOD:385)
    # Special Tokens
    "<pad>": 0.0,
    "<mask_token>": 0.0,
}

# 20 canonical amino acid residues in stable alphabetical order
CANONICAL_AAS = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

# PTM tokens included in MassIVE-KB model
PTM_TOKENS = [
    "M(ox)",
    "C(cam)",
    "N(deam)",
    "Q(deam)",
    "S(ph)",
    "T(ph)",
    "Y(ph)",
    "(+42.01)",
    "(+43.01)",
    "(-17.03)",
]

# Output tokens predicted by model head (tokens 0..29)
OUTPUT_TOKENS = CANONICAL_AAS + PTM_TOKENS

# Special tokens (indices 30 and 31)
SPECIAL_TOKENS = ["<pad>", "<mask_token>"]

# Full vocabulary (32 tokens) matching pretrained checkpoint
VOCAB = OUTPUT_TOKENS + SPECIAL_TOKENS

VOCAB2IDX = {token: idx for idx, token in enumerate(VOCAB)}
IDX2VOCAB = {idx: token for idx, token in enumerate(VOCAB)}

OUTPUT_VOCAB2IDX = {token: idx for idx, token in enumerate(OUTPUT_TOKENS)}
OUTPUT_IDX2VOCAB = {idx: token for idx, token in enumerate(OUTPUT_TOKENS)}

PAD_IDX = VOCAB2IDX["<pad>"]          # 30
MASK_IDX = VOCAB2IDX["<mask_token>"]  # 31
VOCAB_SIZE = len(VOCAB)               # 32
OUTPUT_VOCAB_SIZE = len(OUTPUT_TOKENS)# 30


def calc_peptide_mass(peptide_seq: str) -> float:
    """Calculate neutral monoisotopic mass of a peptide sequence in Da."""
    mass = MASS_H2O
    i = 0
    n = len(peptide_seq)
    while i < n:
        matched = False
        # Match multi-character PTM tokens first
        for tok in sorted(PTM_TOKENS, key=len, reverse=True):
            if peptide_seq.startswith(tok, i):
                mass += AA_MASSES.get(tok, 0.0)
                i += len(tok)
                matched = True
                break
        if not matched:
            char = peptide_seq[i]
            mass += AA_MASSES.get(char, 110.0)
            i += 1
    return mass


def calc_precursor_mz(peptide_mass: float, charge: int) -> float:
    """Calculate precursor m/z from neutral mass and charge."""
    return (peptide_mass + charge * MASS_PROTON) / charge
