"""Pure coordinate transforms for projected canonical-content beliefs."""

from dataclasses import dataclass

import torch
from torch import Tensor


def _validate_moments_and_frame(
    mean: Tensor,
    variance: Tensor,
    frame: Tensor,
    *,
    frame_name: str,
) -> None:
    if mean.shape != variance.shape:
        raise ValueError(
            "mean and variance must have identical shapes, got "
            f"{tuple(mean.shape)} and {tuple(variance.shape)}")
    if frame.dim() < 2 or frame.shape[-2] != frame.shape[-1]:
        raise ValueError(
            f"{frame_name} must have square trailing matrix axes, got {tuple(frame.shape)}")
    if mean.dim() < 1 or mean.shape[-1] != frame.shape[-1]:
        raise ValueError(
            f"moment width must match {frame_name} matrix width, got "
            f"moments={tuple(mean.shape)}, {frame_name}={tuple(frame.shape)}")
    if mean.shape[:-1] != frame.shape[:-2]:
        raise ValueError(
            f"moment and {frame_name} leading shape must match exactly, got "
            f"{tuple(mean.shape[:-1])} and {tuple(frame.shape[:-2])}")
    if not (mean.dtype == variance.dtype == frame.dtype):
        raise ValueError(
            f"mean, variance, and {frame_name} must have the same dtype, got "
            f"{mean.dtype}, {variance.dtype}, and {frame.dtype}")
    if not (mean.device == variance.device == frame.device):
        raise ValueError(
            f"mean, variance, and {frame_name} must be on the same device, got "
            f"{mean.device}, {variance.device}, and {frame.device}")


@dataclass(frozen=True)
class CanonicalFrameContext:
    """The realized per-token vertex frame and its inverse from one transport build."""

    forward: Tensor
    inverse: Tensor

    def __post_init__(self) -> None:
        if self.forward.shape != self.inverse.shape:
            raise ValueError(
                "canonical forward and inverse factors must have identical shapes, got "
                f"{tuple(self.forward.shape)} and {tuple(self.inverse.shape)}")
        if self.forward.dim() < 2 or self.forward.shape[-2] != self.forward.shape[-1]:
            raise ValueError(
                "canonical frame factors must have square trailing matrix axes, got "
                f"{tuple(self.forward.shape)}")
        if self.forward.dtype != self.inverse.dtype:
            raise ValueError(
                "canonical forward and inverse factors must have the same dtype, got "
                f"{self.forward.dtype} and {self.inverse.dtype}")
        if self.forward.device != self.inverse.device:
            raise ValueError(
                "canonical forward and inverse factors must be on the same device, got "
                f"{self.forward.device} and {self.inverse.device}")


def project_canonical_diagonal(
    mu_c: Tensor,
    var_c: Tensor,
    frame: Tensor,
) -> tuple[Tensor, Tensor]:
    """Push canonical diagonal moments through ``frame`` and retain the output diagonal."""
    _validate_moments_and_frame(mu_c, var_c, frame, frame_name="frame")
    # These transforms are a public coordinate boundary: a surrounding model autocast region must
    # not silently change float32 canonical tables / realized frames into bf16 moments. Disabling
    # autocast preserves the already-validated common input dtype without introducing any cast or
    # detaching the canonical/frame graph.
    with torch.amp.autocast(mu_c.device.type, enabled=False):
        return (
            torch.einsum("...ij,...j->...i", frame, mu_c),
            torch.einsum("...ij,...j->...i", frame.square(), var_c),
        )


def pullback_diagonal_query(
    mu_q: Tensor,
    var_q: Tensor,
    frame_inv: Tensor,
) -> tuple[Tensor, Tensor]:
    """Pull a realized diagonal query into canonical full-covariance coordinates."""
    _validate_moments_and_frame(mu_q, var_q, frame_inv, frame_name="frame_inv")
    with torch.amp.autocast(mu_q.device.type, enabled=False):
        mu_c = torch.einsum("...ij,...j->...i", frame_inv, mu_q)
        cov_c = frame_inv @ torch.diag_embed(var_q) @ frame_inv.transpose(-1, -2)
    return mu_c, cov_c
