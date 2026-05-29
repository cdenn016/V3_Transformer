# V3_Transformer (VFE_3.0)

Clean-room rebuild of the gauge-theoretic VFE transformer. No neural networks:
all capacity comes from iterative VFE minimization over Gaussian belief tuples
`(mu, Sigma, phi)`. Built bottom-up, every layer numerically pinned to VFE_2.0
by golden tests. See `docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md`.

## Hard constraints
- NO neural networks (no nn.Linear, no MLP, no activations).
- NO CLI arg parsing; entry points are click-to-run (edit config dicts, then run).
- float32 throughout; CUDA where applicable (user has an RTX 5090).
- High modularity: a config-selected registry behind every seam (divergence,
  alpha_i, family, transport/gauge, retraction, decode). Add a variant by
  writing-and-registering it, never by editing call sites.
- Always preserve a theoretically pure path under appropriate toggles.

## Function signature convention (MANDATORY)
Argument order: all torch.Tensor first, then 'float | torch.Tensor', then
undefined floats, undefined ints, undefined bools, then defined floats,
defined ints, defined bools, then Optional, then **kwargs last.

Vertical alignment: names, type annotations, `=` signs, and trailing `#`
comments are each aligned to a common column. Blank lines separate type
groups. Tensor shape comments at critical points. Type hints on every
signature. Docstrings carry the LaTeX/math form for non-trivial formulas.
Variable names match paper notation (mu_q, sigma_q, alpha, kappa).

Example:

    def kernel(
        mu_q:    torch.Tensor,             # (..., K) query means
        sigma_q: torch.Tensor,             # (..., K) query variances

        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
    ) -> torch.Tensor:

## Testing
Golden equivalence vs a pinned VFE_2.0 checkout for every ported kernel;
finite-difference gradient checks against the autograd-of-F oracle (later
phases); property tests (non-negativity, self-divergence zero, gauge
equivariance). Tests are device-agnostic (default CPU; set
VFE3_TEST_DEVICE=cuda for the GPU).
