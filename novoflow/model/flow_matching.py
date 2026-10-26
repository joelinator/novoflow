"""
Continuous-Time Discrete Flow Matching Dynamics.
Implements the probability path κ(t), noise schedules, and the reverse Euler ODE integrator.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from novoflow.constants import VOCAB2IDX, PAD_IDX, MASK_IDX


def cosine_scheduler(time: torch.Tensor, s: float = 0.008) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Smooth cosine noise schedule: κ(t) = sin²(θ(t)), where θ(t) = (t + s) / (1 + s) · π/2.
    Returns (κ(t), dκ/dt).
    """
    theta = (time + s) / (1 + s) * (torch.pi / 2)
    kt = torch.sin(theta).pow(2)
    kt_derivative = (torch.sin(2 * theta) * (torch.pi / 2) / (1 + s)).clamp(min=0.0)
    return kt, kt_derivative


def linear_scheduler(time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear noise schedule: κ(t) = t."""
    return time, torch.ones_like(time)


def sample_reverse_step(
    x_t: torch.Tensor,
    logits: torch.Tensor,
    t_curr: float,
    t_next: float,
    temperature: float = 0.0,
    mask_token_id: int = MASK_IDX,
    scheduler_fn=cosine_scheduler,
) -> torch.Tensor:
    """
    Take one reverse Euler flow matching step from t_curr to t_next (0 -> 1).
    Probability of transition from masked/noisy state to predicted clean token:
        P(transition) = (κ(t_next) - κ(t_curr)) / (1 - κ(t_curr))
    """
    device = x_t.device
    t_c = torch.tensor(t_curr, device=device, dtype=torch.float32)
    t_n = torch.tensor(t_next, device=device, dtype=torch.float32)

    k_c, _ = scheduler_fn(t_c)
    k_n, _ = scheduler_fn(t_n)

    # Transition probability for unmasking
    denom = (1.0 - k_c).clamp(min=1e-5)
    p_trans = ((k_n - k_c) / denom).clamp(0.0, 1.0)

    # Convert logits to probabilities
    if temperature <= 1e-4:
        # Greedy MAP
        pred_tokens = logits.argmax(dim=-1)
    else:
        # Stochastic sampling with temperature
        probs = F.softmax(logits / temperature, dim=-1)
        b, l, v = probs.shape
        pred_tokens = torch.multinomial(probs.view(-1, v), num_samples=1).view(b, l)

    # Apply transition mask
    is_masked = (x_t == mask_token_id)
    rand_mask = torch.rand_like(x_t, dtype=torch.float32) < p_trans

    update_mask = is_masked & rand_mask
    x_next = torch.where(update_mask, pred_tokens, x_t)
    return x_next
