r"""PriorBank for VFE_3.0: learnable Gaussian vocab priors + the KL decode boundary.

Holds the per-vocabulary prior pi_v = N(mu_v, Sigma_v) with gauge frame phi_v as
PARAMETER TABLES (nn.Parameter -- priors, not neural maps; the no-NN rule bans
nn.Linear/MLP/activations, not learnable parameters). encode(token_ids) looks them
up into the initial belief (q = p); decode(mu_q, sigma_q) scores the posterior
against every prior as logits = -KL(q || pi_v)/tau_eff (the divergence seam),
replacing a linear output projection.

Modularity:
    encode_mode registry -- ``per_token`` (table lookup, default); ``gauge_fixed`` a
        named stub (gauge orbit from a shared base belief).
    decode_mode registry -- ``diagonal`` (fused closed form, default); ``full`` a named
        stub (exact Cholesky for full covariances).

Divergence-agnostic, scope clarified: ``reference_decode`` is the literal seam path --
it calls ``divergence.kl`` and so tracks whatever divergence family/alpha the seam is
configured for. The default fused ``diagonal`` kernel is a hand-specialized alpha=1
``gaussian_diagonal`` shortcut (one matmul, no per-V ``kl`` call) that is pinned EXACTLY
to that seam (and under ``log_softmax``); it does not re-derive itself for a different
family. The registry seam is therefore honored at the family granularity: a new
COVARIANCE STRUCTURE (e.g. full-covariance) is added by writing-and-registering a new
decode kernel (the ``full`` stub), never by editing a call site -- and ``reference_decode``
already covers any registered divergence for verification.
"""

from typing import Callable, Dict, Optional

import torch
from torch import nn

from vfe3.belief import BeliefState
from vfe3.divergence import get_family, kl


# ---------------------------------------------------------------------------
# Registries: mode name -> callable. Variants swap by config; add a variant by
# writing-and-registering it, never by editing call sites.
#   encode: fn(pb, token_ids) -> BeliefState
#   decode: fn(pb, mu_q, sigma_q, tau_eff) -> logits (B, N, V)
# ---------------------------------------------------------------------------
_ENCODERS: Dict[str, Callable] = {}
_DECODERS: Dict[str, Callable] = {}


def register_encode(name: str) -> Callable:
    """Decorator registering an encode kernel under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        _ENCODERS[name] = fn
        return fn
    return _wrap


def get_encode(name: str) -> Callable:
    """Return the registered encode kernel for ``name`` (KeyError if absent)."""
    if name not in _ENCODERS:
        raise KeyError(
            f"no encode mode registered under {name!r}; available: {sorted(_ENCODERS)}"
        )
    return _ENCODERS[name]


def register_decode(name: str) -> Callable:
    """Decorator registering a decode kernel under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        _DECODERS[name] = fn
        return fn
    return _wrap


def get_decode(name: str) -> Callable:
    """Return the registered decode kernel for ``name`` (KeyError if absent)."""
    if name not in _DECODERS:
        raise KeyError(
            f"no decode mode registered under {name!r}; available: {sorted(_DECODERS)}"
        )
    return _DECODERS[name]


class PriorBank(nn.Module):
    r"""Learnable Gaussian vocab priors; encode (lookup) and decode (-KL/tau_eff).

    The tables ``mu_embed`` (V, K), ``sigma_log_embed`` (V, K), ``phi_embed`` (V, n_gen)
    parameterize the priors pi_v = N(mu_v, exp(sigma_log_v)) with gauge frame phi_v.
    They are PRIORS (nn.Parameter), not a neural map: there is no nn.Linear/MLP/activation
    anywhere in this module. The learnable scalar ``decode_log_scale`` tunes the decode
    temperature.
    """

    output_proj_weight: Optional[nn.Parameter]   # (V, K) linear-decode weight; None unless use_prior_bank=False

    def __init__(
        self,
        vocab_size:   int,
        K:            int,
        n_gen:        int,

        *,
        mu_init_std:         float = 0.02,
        sigma_init:          float = 1.0,
        phi_scale:           float = 0.01,
        decode_tau:          float = 1.0,
        eps:                 float = 1e-6,
        diagonal_covariance: bool  = True,
        use_prior_bank:      bool  = True,
        encode_mode:         str   = "per_token",
        decode_mode:         str   = "diagonal",
        lambda_h:            float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.K = K
        self.n_gen = n_gen
        self.decode_tau = decode_tau
        self.eps = eps
        self.diagonal_covariance = diagonal_covariance
        self.use_prior_bank = use_prior_bank
        self.encode_mode = encode_mode
        self.decode_mode = decode_mode

        sigma_log_init = float(torch.log(torch.tensor(sigma_init)))
        self.mu_embed         = nn.Parameter(mu_init_std * torch.randn(vocab_size, K))
        self.sigma_log_embed  = nn.Parameter(torch.full((vocab_size, K), sigma_log_init))
        self.phi_embed        = nn.Parameter(phi_scale * torch.randn(vocab_size, n_gen))
        self.decode_log_scale = nn.Parameter(torch.zeros(1))

        # use_prior_bank=False (VFE_2.0-parity ablation): decode is a plain linear projection
        # logits = mu_q @ W^T through a learned (V, K) weight, the single authorized neural
        # exception (a lone linear output readout; see CLAUDE.md). Realized as a raw nn.Parameter
        # matmul -- NOT an nn.Linear/MLP -- so no neural-layer class enters the module. Created
        # only on the ablation path so the pure path (use_prior_bank=True) carries no extra weight.
        # Xavier-uniform init (matches VFE_2.0's nn.Linear default), no bias (a constant shift in
        # V that softmax/cross-entropy absorbs). Encode stays the prior-bank lookup either way.
        if use_prior_bank:
            self.output_proj_weight = None
        else:
            self.output_proj_weight = nn.Parameter(torch.empty(vocab_size, K))
            nn.init.xavier_uniform_(self.output_proj_weight)

        # HYPER-PRIOR CHANNEL (manuscript eq:pointwise_free_energy), FIRST INCREMENT, default-OFF.
        # When lambda_h > 0, create the model-channel belief tables s_mu_embed/s_sigma_log_embed
        # (V, K) -- a per-token DIAGONAL Gaussian s_i looked up like the belief tables -- and the
        # global hyper-prior r_mu/r_sigma_log (K,), a single diagonal Gaussian the s_i are
        # regularized toward (the manuscript centroid). These are PRIORS (nn.Parameter), not a
        # neural map. They are created LAST and ONLY on the lambda_h>0 path: the default (lambda_h=0)
        # path draws zero new RNG, so the belief tables above are byte-unchanged and the pure path
        # is param-free (no s_mu_embed attribute at all). s init mirrors the belief tables (small mu,
        # sigma matching sigma_init); r init: mu=0, sigma matching sigma_init -- so s != r at init
        # (KL(s||r) > 0, the channel has a gradient). NOTE (first increment): s_i is NOT yet coupled
        # to the belief q; the gamma model-coupling block and the s-channel E-step update are
        # DEFERRED to increment 2.
        if lambda_h > 0.0:
            self.s_mu_embed        = nn.Parameter(mu_init_std * torch.randn(vocab_size, K))
            self.s_sigma_log_embed = nn.Parameter(torch.full((vocab_size, K), sigma_log_init))
            self.r_mu              = nn.Parameter(torch.zeros(K))
            self.r_sigma_log       = nn.Parameter(torch.full((K,), sigma_log_init))

    def encode(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
    ) -> BeliefState:
        r"""Look up the per-token Gaussian prior as the initial belief (q = p)."""
        return get_encode(self.encode_mode)(self, token_ids)

    def encode_s(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
    ) -> 'tuple[torch.Tensor, torch.Tensor]':
        r"""Look up the per-token model-channel belief s_i = N(s_mu, s_sigma) (diagonal).

        Returns (s_mu, s_sigma) with s_mu (B, N, K) and s_sigma (B, N, K) the positive
        variances exp(s_sigma_log).clamp(min=eps). Available only on the hyper-prior path
        (lambda_h>0, where the s tables are created); the manuscript hyper-prior term consumes
        this as DiagonalGaussian(s_mu, s_sigma). FIRST INCREMENT: this is consumed only by the
        lambda_h * mean_i KL(s_i||r) loss term; s_i is not yet coupled into the belief q.
        """
        s_mu = self.s_mu_embed[token_ids]                                       # (B, N, K)
        s_sigma = torch.exp(self.s_sigma_log_embed[token_ids]).clamp(min=self.eps)  # (B, N, K)
        return s_mu, s_sigma

    def _tau_eff(
        self,
        tau: Optional[float] = None,     # override decode_tau; None -> self.decode_tau
    ) -> torch.Tensor:
        r"""Effective decode temperature tau_eff = tau * exp(-clamp(decode_log_scale, -3, 3))."""
        base_tau = self.decode_tau if tau is None else tau
        return base_tau * torch.exp(-self.decode_log_scale.clamp(-3.0, 3.0))

    def decode(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances

        *,
        tau:     Optional[float] = None,  # override decode_tau; None -> self.decode_tau
    ) -> torch.Tensor:                   # (B, N, V) logits
        r"""Decode logits via the selected kernel; ``use_prior_bank`` is the single gate.

        True (default, pure path): the KL-to-prior readout -KL(q_i || pi_v)/tau_eff with the
        covariance structure given by ``decode_mode`` (diagonal | full). False (ablation): the
        ``linear`` kernel logits = mu_q @ W^T (sigma_q and tau_eff ignored). Routing here -- not
        through a second config value -- keeps ``decode_mode`` and ``use_prior_bank`` from ever
        silently disagreeing (the linear path simply does not consult ``decode_mode``)."""
        mode = self.decode_mode if self.use_prior_bank else "linear"
        return get_decode(mode)(self, mu_q, sigma_q, self._tau_eff(tau))

    def reference_decode(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances

        *,
        tau:     Optional[float] = None,  # override decode_tau; None -> self.decode_tau
    ) -> torch.Tensor:                   # (B, N, V) logits = -KL(q || pi_v)/tau_eff
        r"""Divergence-agnostic reference decode: -KL(q_i || pi_v)/tau_eff via the seam.

        Broadcasts the ``divergence.kl`` seam over the vocabulary V (general but slow,
        O(B*N*V*K)). The fused ``diagonal`` kernel is pinned to this exactly and under
        log-softmax; a new divergence family needs no decode edit (only the seam call).

        The seam is invoked with ``kl_max=inf``: a DECODE must preserve the full KL
        ranking over the vocabulary, so the divergence saturation policy (default
        ``kl_max=100``, which flattens every distant prior to a single -100 logit and
        destroys the argmax) is disabled here. The fused kernel computes the unclamped
        -KL/tau_eff, so both decode paths agree across the whole input domain, not only
        where KL < 100. (``nan_to_num`` inside ``safe_kl_clamp`` still maps NaN/+inf
        from degenerate pairs to +inf -> -inf logits.)
        """
        tau_eff = self._tau_eff(tau)
        mu_v = self.mu_embed                                             # (V, K)
        sigma_v = torch.exp(self.sigma_log_embed).clamp(min=self.eps)    # (V, K)
        mu_q_b = mu_q.unsqueeze(-2)                                      # (B, N, 1, K)
        sigma_q_b = sigma_q.unsqueeze(-2)                               # (B, N, 1, K)
        diag = get_family("gaussian_diagonal")
        kl_v = kl(diag(mu_q_b, sigma_q_b), diag(mu_v, sigma_v), kl_max=float("inf"))  # (B, N, V), unclamped
        return -kl_v / tau_eff


@register_encode("per_token")
def _encode_per_token(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""Per-token table lookup: token_ids -> (mu_v, sigma_v, phi_v) as the belief q = p.

    Diagonal family: sigma is the (B, N, K) variance vector. Full family
    (``diagonal_covariance=False``): the same per-token variances are embedded as a
    DIAGONAL full covariance (B, N, K, K) -- the SPD starting point the full-covariance
    E-step (full sandwich transport + affine-invariant SPD retraction) then evolves
    off-diagonal mass into. The mean / gauge tables are shared across families.
    """
    mu = pb.mu_embed[token_ids]                                              # (B, N, K)
    sigma_diag = torch.exp(pb.sigma_log_embed[token_ids]).clamp(min=pb.eps)  # (B, N, K), sigma > 0
    phi = pb.phi_embed[token_ids]                                            # (B, N, n_gen)
    sigma = sigma_diag if pb.diagonal_covariance else torch.diag_embed(sigma_diag)
    return BeliefState(mu=mu, sigma=sigma, phi=phi)


@register_encode("gauge_fixed")
def _encode_gauge_fixed(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""NAMED STUB: gauge-fixed encode (gauge orbit from a shared base belief).

    Deferred: would realize every prior as a gauge transform of one shared base
    belief, so the vocabulary varies only along the gauge orbit. Not yet implemented.
    """
    raise NotImplementedError(
        "encode_mode='gauge_fixed' is a named stub (gauge orbit from a shared base); "
        "use 'per_token'."
    )


@register_decode("diagonal")
def _decode_diagonal(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Exact diagonal -KL/tau_eff in closed form via a single fused matmul.

        KL = 0.5[ sum_k(sigma_q/sigma_v + (mu_q-mu_v)^2/sigma_v) - K + sum_k log(sigma_v/sigma_q) ]
    The v-dependent part A_v expands the Mahalanobis/trace terms into one matmul:
        lhs = [sigma_q + mc_q^2, -2 mc_q]            (B, N, 2K)
        rhs = [1/sigma_v,        mc_v/sigma_v]       (V, 2K)
        A_v = lhs @ rhs^T + sum_k(mc_v^2/sigma_v + log sigma_v)
            == sum_k(sigma_q/sigma_v + (mc_q-mc_v)^2/sigma_v) + sum_k log sigma_v
            == 2 KL + K + sum_k log sigma_q.
    The per-position (-K - sum_k log sigma_q) is v-INDEPENDENT (drops under softmax) but
    is KEPT so logits == -KL/tau_eff EXACTLY.

    NUMERICS: the Mahalanobis term ``(mu_q - mu_v)^2`` is reconstructed by the matmul as
    ``mc_q^2 - 2 mc_q mc_v + mc_v^2``, a subtraction of large near-equal quantities that
    catastrophically cancels in float32 once the means carry a large common offset (the
    error grows like eps * mu^2 / sigma_v and breaks the atol-1e-3 seam pin at modest
    |mu| / tight sigma_v). We remove the common offset BEFORE the matmul by subtracting
    the v-independent shift ``c = mean_v(mu_v)`` (per dim) from both means; since
    ``(mu_q - c) - (mu_v - c) == mu_q - mu_v`` the closed form is unchanged exactly while
    the cancelled magnitude collapses to the residual spread of the means.
    """
    sigma_v = torch.exp(pb.sigma_log_embed).clamp(min=pb.eps)            # (V, K)
    mu_v = pb.mu_embed                                                  # (V, K)
    inv_v = 1.0 / sigma_v                                               # (V, K) = 1/sigma_v

    c = mu_v.mean(dim=0, keepdim=True)                                  # (1, K) v-independent shift
    mc_v = mu_v - c                                                     # (V, K) centered prior means
    mc_q = mu_q - c                                                     # (B, N, K) centered query means

    lhs = torch.cat([sigma_q + mc_q ** 2, -2.0 * mc_q], dim=-1)          # (B, N, 2K)
    rhs = torch.cat([inv_v, mc_v * inv_v], dim=-1)                       # (V, 2K)
    a_v = lhs @ rhs.transpose(-1, -2)                                    # (B, N, V): sum_k[(sigma_q+mc_q^2-2 mc_q mc_v)/sigma_v]
    a_v = a_v + (mc_v ** 2 * inv_v).sum(-1) + torch.log(sigma_v).sum(-1)  # + sum_k(mc_v^2/sigma_v + log sigma_v)
    # a_v == sum_k(sigma_q/sigma_v + (mc_q-mc_v)^2/sigma_v) + sum_k log sigma_v
    #     == sum_k(sigma_q/sigma_v + (mu_q-mu_v)^2/sigma_v) + sum_k log sigma_v = 2 KL + K + sum_k log sigma_q
    per_pos = pb.K + torch.log(sigma_q.clamp(min=pb.eps)).sum(-1, keepdim=True)   # (B, N, 1) = K + sum_k log sigma_q
    kl_v = 0.5 * (a_v - per_pos)                                         # (B, N, V)
    return -kl_v / tau_eff


@register_decode("full")
def _decode_full(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K, K) posterior covariances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Exact full-covariance decode logits_{i,v} = -KL(q_i || pi_v)/tau_eff via Cholesky.

    Scores the full-covariance posterior q_i = N(mu_q, Sigma_q) against every vocab prior
    pi_v through the ``gaussian_full`` divergence seam (Cholesky KL). The prior table is
    diagonal (sigma_log_embed), embedded as a diagonal full covariance diag(exp(sigma_log_v))
    so a full q is scored against it. As in ``reference_decode`` the seam is invoked with
    ``kl_max=inf`` so the full KL ranking over the vocabulary is preserved (decode must not
    saturate distant priors to a single logit). General but O(B*N*V*K^3) (per-pair Cholesky):
    the theoretically pure full-covariance path, not the fast diagonal kernel.
    """
    mu_v = pb.mu_embed                                                   # (V, K)
    sigma_v = torch.diag_embed(torch.exp(pb.sigma_log_embed).clamp(min=pb.eps))  # (V, K, K) diagonal-as-full
    mu_q_b = mu_q.unsqueeze(-2)                                          # (B, N, 1, K)
    sigma_q_b = sigma_q.unsqueeze(-3)                                    # (B, N, 1, K, K)
    full = get_family("gaussian_full")
    kl_v = kl(full(mu_q_b, sigma_q_b), full(mu_v, sigma_v), kl_max=float("inf"))  # (B, N, V)
    return -kl_v / tau_eff


@register_decode("linear")
def _decode_linear(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances (DISCARDED)
    tau_eff: torch.Tensor,               # () effective temperature (DISCARDED)
) -> torch.Tensor:                       # (B, N, V) logits = mu_q @ W^T
    r"""Linear-projection decode (use_prior_bank=False, VFE_2.0 parity): logits = mu_q @ W^T.

    The one authorized neural exception: a single learned (V, K) output weight applied to the
    converged mean, with NO KL geometry at the decode boundary (sigma and the decode temperature
    are discarded; only encode + the E-step remain gauge-aware). Realized as a raw nn.Parameter
    matmul, not an nn.Linear module. The pure KL-readout path is always available under
    use_prior_bank=True; this is the opt-in ablation the user uses to compare with/without the
    prior-bank decode.
    """
    return mu_q @ pb.output_proj_weight.transpose(-1, -2)               # (B, N, V)
