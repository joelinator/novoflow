"""
Exact Dynamic Programming Knapsack Reachability Filter.
Guarantees physical precursor mass conservation in O(1) query time.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from novoflow.constants import AA_MASSES, CANONICAL_AAS, PTM_TOKENS, MASS_H2O, VOCAB2IDX


class ExactReachabilityDP:
    """
    Dynamic programming reachable-mass table for mass knapsack guidance.
    Table dp[k, b] is True if k amino acid residues can sum to mass bin b.
    Precomputes 1D max pooling over tolerance window for O(1) batched lookups.
    """

    _cached_instance: ExactReachabilityDP | None = None

    def __init__(
        self,
        aa_masses: list[float] | dict[str, float] | None = None,
        max_len: int = 50,
        max_mass: float = 5500.0,
        bin_size: float = 0.02,
        tol_da: float = 0.5,
        device: torch.device | str = "cpu",
    ):
        self.max_len = max_len
        self.max_mass = max_mass
        self.bin_size = bin_size
        self.tol_da = tol_da
        self.device = torch.device(device)
        self.num_bins = int(round(max_mass / bin_size)) + 1

        if aa_masses is None:
            masses = [
                AA_MASSES[aa]
                for aa in (CANONICAL_AAS + PTM_TOKENS)
                if aa in AA_MASSES and AA_MASSES[aa] > 10.0
            ]
        elif isinstance(aa_masses, dict):
            masses = list(aa_masses.values())
        else:
            masses = list(aa_masses)

        self.masses = sorted(list(set(float(m) for m in masses if float(m) > 10.0)))
        self.dp_table = self._build_dp_table()
        self.pooled_table = self._pool_for_tolerance(tol_da)

    def _build_dp_table(self) -> torch.Tensor:
        """Construct reachable mass table for all k in [0, max_len]."""
        table = torch.zeros((self.max_len + 1, self.num_bins), dtype=torch.bool, device=self.device)
        table[0, 0] = True

        mass_bins = sorted(
            list(
                set(
                    int(round(m / self.bin_size))
                    for m in self.masses
                    if 0 < int(round(m / self.bin_size)) < self.num_bins
                )
            )
        )

        for k in range(1, self.max_len + 1):
            prev = table[k - 1]
            curr = torch.zeros(self.num_bins, dtype=torch.bool, device=self.device)
            for mb in mass_bins:
                curr[mb:] |= prev[:-mb]
            table[k] = curr

        return table

    def _pool_for_tolerance(self, tol_da: float) -> torch.Tensor:
        """Apply max-pooling across tolerance window for O(1) reachability lookup."""
        w = max(1, int(round(tol_da / self.bin_size)))
        kernel_size = 2 * w + 1
        float_dp = self.dp_table.unsqueeze(0).float()
        pooled = F.max_pool1d(float_dp, kernel_size=kernel_size, stride=1, padding=w).squeeze(0)
        return pooled > 0.5

    def to(self, device: torch.device | str) -> ExactReachabilityDP:
        dev = torch.device(device)
        if self.device != dev:
            self.device = dev
            self.dp_table = self.dp_table.to(dev)
            self.pooled_table = self.pooled_table.to(dev)
        return self

    def is_reachable(
        self,
        k: int | torch.Tensor,
        remaining_mass: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        Query reachability: can k residues sum to remaining_mass within tol_da?
        """
        if not isinstance(remaining_mass, torch.Tensor):
            remaining_mass = torch.tensor(remaining_mass, device=self.device, dtype=torch.float32)
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=self.device, dtype=torch.long)

        k = k.to(device=self.device, dtype=torch.long)
        remaining_mass = remaining_mass.to(device=self.device, dtype=torch.float32)

        k0_valid = (k == 0) & (remaining_mass.abs() <= self.tol_da)
        in_range = (k >= 1) & (k <= self.max_len) & (remaining_mass >= -self.tol_da) & (remaining_mass <= self.max_mass)
        clamped_mass = remaining_mass.clamp(0.0, self.max_mass)
        bins = (clamped_mass / self.bin_size).round().long().clamp(0, self.num_bins - 1)
        clamped_k = k.clamp(0, self.max_len)

        dp_valid = self.pooled_table[clamped_k, bins] & in_range
        return torch.where(k == 0, k0_valid, dp_valid)

    @classmethod
    def get_default(cls, device: torch.device | str = "cpu") -> ExactReachabilityDP:
        if cls._cached_instance is None:
            cls._cached_instance = cls(device=device)
        else:
            cls._cached_instance.to(device)
        return cls._cached_instance


def get_reachable_lengths(
    precursor_neutral_mass: float,
    reachability_dp: ExactReachabilityDP,
    min_len: int = 5,
    max_len: int = 40,
) -> list[int]:
    """Find all peptide lengths that can physically achieve precursor neutral mass."""
    residue_target = max(0.0, precursor_neutral_mass - MASS_H2O)
    reachable = []
    for L in range(min_len, max_len + 1):
        if reachability_dp.is_reachable(L, residue_target).item():
            reachable.append(L)
    return reachable


def filter_logits_by_reachability(
    logits: torch.Tensor,
    pos_idx: int,
    total_len: int,
    current_residue_mass: torch.Tensor,
    target_residue_mass: torch.Tensor,
    reachability_dp: ExactReachabilityDP,
    aa_mass_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Mask out amino acids from logits at pos_idx that cannot physically lead
    to target_residue_mass in remaining (total_len - 1 - pos_idx) steps.
    """
    remaining_k = total_len - 1 - pos_idx
    batch_size, vocab_size = logits.shape
    device = logits.device

    # current_residue_mass: (B,), target_residue_mass: (B,)
    # aa_mass_tensor: (V,)
    # new_remaining_mass for each AA candidate: target - (current + aa_mass)
    # Shape: (B, V)
    rem_mass = target_residue_mass.unsqueeze(1) - (current_residue_mass.unsqueeze(1) + aa_mass_tensor.unsqueeze(0))
    k_tensor = torch.full_like(rem_mass, fill_value=remaining_k, dtype=torch.long)

    valid_mask = reachability_dp.is_reachable(k_tensor, rem_mass)
    masked_logits = logits.masked_fill(~valid_mask, -1e9)
    return masked_logits
