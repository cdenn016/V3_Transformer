"""Configuration for VFE_3.0. Single dataclass, single validation block.

No CLI parsing (project policy: click-to-run). Edit fields directly.
"""

from dataclasses import dataclass

_VALID_DIVERGENCE_FAMILIES = ("gaussian_diagonal", "gaussian_full")


@dataclass
class VFE3Config:
    """Phase 0/1 configuration surface (divergence layer only).

    Attributes:
        eps:               Regularization floor for variances / covariances.
        kl_max:            Upper clamp on divergence values.
        divergence_family: Registry key selecting the divergence kernel.
        alpha_div:         Renyi order; 1.0 recovers standard KL.
    """

    eps:               float = 1e-6
    kl_max:            float = 100.0
    divergence_family: str   = "gaussian_diagonal"
    alpha_div:         float = 1.0

    def __post_init__(self) -> None:
        if self.divergence_family not in _VALID_DIVERGENCE_FAMILIES:
            raise ValueError(
                f"divergence_family must be one of {_VALID_DIVERGENCE_FAMILIES}, "
                f"got {self.divergence_family!r}"
            )
        if self.alpha_div <= 0.0:
            raise ValueError(f"alpha_div must be positive, got {self.alpha_div}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.kl_max <= 0.0:
            raise ValueError(f"kl_max must be positive, got {self.kl_max}")
