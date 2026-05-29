r"""Gauge transport for VFE_3.0 (Regime I, Gaussian / location-scale specific).

Two parameterizations of the flat (Regime I) transport:
  phi (exp):    Omega_ij = exp(phi_i . G) exp(-phi_j . G) in GL+(K) (det>0).
  omega_direct: Omega_ij = Omega_i Omega_j^{-1} for general GL(K) (det may be <0).
Belief action: mu -> Omega @ mu, Sigma -> Omega @ Sigma @ Omega^T (sandwich;
diagonal approximation for speed). Regime II, retractions, RoPE are later phases.
"""

from typing import Dict, List, Optional, Tuple

import torch

from vfe3.geometry.groups import GaugeGroup

TransportDict = Dict[str, torch.Tensor]


def stable_matrix_exp_pair(
    matrix:         torch.Tensor,             # (..., d, d) Lie-algebra matrices

    *,
    max_norm:       float = 15.0,
    dim_threshold:  int   = 20,
    skew_symmetric: bool  = False,
    only_forward:   bool  = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""exp(M) and optionally exp(-M) with Frobenius-norm clamp + float64 upcast.

    Ported from VFE_2.0 stable_matrix_exp_pair (gauge_utils.py:53-131).
    """
    mat_norm = matrix.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    scale = (max_norm / mat_norm).clamp(max=1.0)
    matrix = matrix * scale

    d = matrix.shape[-1]
    orig_dtype = matrix.dtype
    with torch.amp.autocast('cuda', enabled=False):
        if d >= dim_threshold:
            matrix_up = matrix.double().contiguous()
        else:
            matrix_up = matrix.float().contiguous()
        exp_pos = torch.linalg.matrix_exp(matrix_up).to(orig_dtype)
        if only_forward:
            exp_neg = None
        elif skew_symmetric:
            exp_neg = exp_pos.transpose(-1, -2)
        else:
            exp_neg = torch.linalg.matrix_exp(-matrix_up).to(orig_dtype)
    return exp_pos, exp_neg


def compute_transport_operators(
    phi:        torch.Tensor,             # (B, N, n_gen) gauge frames
    group:      GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode: str = "learned",          # 'learned' (Regime I flat) or 'trivial'
) -> TransportDict:
    r"""phi/exp transport Omega_ij = exp(phi_i) @ exp(-phi_j) in GL+(K).

    Ported from VFE_2.0 compute_transport_operators (transport_ops.py:285-433),
    flat path. 'trivial' returns Omega = I. Returns 'exp_phi' (B,N,K,K),
    'exp_neg_phi' (B,N,K,K), 'Omega' (B,N,N,K,K).
    """
    B, N, _ = phi.shape
    generators = group.generators
    K = generators.shape[-1]
    dtype = phi.dtype
    device = phi.device

    if gauge_mode == "trivial":
        eye_K = torch.eye(K, device=device, dtype=dtype)
        return {
            "exp_phi":     eye_K.expand(B, N, K, K).contiguous(),
            "exp_neg_phi": eye_K.expand(B, N, K, K).contiguous(),
            "Omega":       eye_K.expand(B, N, N, K, K).contiguous(),
        }
    if gauge_mode != "learned":
        raise ValueError(f"gauge_mode must be 'learned' or 'trivial', got {gauge_mode!r}")

    phi_matrix = torch.einsum("bna,aij->bnij", phi, generators)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(
        phi_matrix, skew_symmetric=group.skew_symmetric
    )
    omega = torch.einsum("bikl,bjlm->bijkm", exp_phi, exp_neg_phi)
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}


def compute_transport_operators_direct(
    omega:      torch.Tensor,             # (B, N, K, K) per-token group elements Omega_i

    *,
    gauge_mode: str   = "learned",        # 'learned' (flat cocycle) or 'trivial'
    eps:        float = 1e-6,
) -> TransportDict:
    r"""Direct-Omega transport Omega_ij = Omega_i @ Omega_j^{-1} (general GL(K)).

    Ported from VFE_2.0 compute_transport_operators_direct (transport_ops.py:440),
    flat path. Reaches all of GL(K) (det may be < 0; needs an external det
    penalty to stay invertible). Inverse via LU solve (exact cocycle), with a
    ridge then pinv fallback for near-singular Omega. 'trivial' returns Omega=I.
    Returns 'omega_i' (B,N,K,K), 'omega_j_inv' (B,N,K,K), 'Omega' (B,N,N,K,K).
    """
    B, N, K, _ = omega.shape
    dtype = omega.dtype
    device = omega.device

    if gauge_mode == "trivial":
        eye_K = torch.eye(K, device=device, dtype=dtype)
        return {
            "omega_i":     eye_K.expand(B, N, K, K).contiguous(),
            "omega_j_inv": eye_K.expand(B, N, K, K).contiguous(),
            "Omega":       eye_K.expand(B, N, N, K, K).contiguous(),
        }
    if gauge_mode != "learned":
        raise ValueError(f"gauge_mode must be 'learned' or 'trivial', got {gauge_mode!r}")

    eye_K = torch.eye(K, device=device, dtype=dtype)
    try:
        omega_j_inv = torch.linalg.solve(omega, eye_K.expand_as(omega))
    except (torch.linalg.LinAlgError, RuntimeError):
        try:
            omega_j_inv = torch.linalg.solve(omega + eps * eye_K, eye_K.expand_as(omega))
        except (torch.linalg.LinAlgError, RuntimeError):
            omega_j_inv = torch.linalg.pinv(omega)

    omega_ij = torch.einsum("bikl,bjlm->bijkm", omega, omega_j_inv)
    return {"omega_i": omega, "omega_j_inv": omega_j_inv, "Omega": omega_ij}
