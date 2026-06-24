r"""Geometrically-correct M-step optimizer for the gauge frame (opt-in).

The gauge-frame prior tables ``phi_embed`` / ``pos_phi_free`` are Lie-algebra COORDINATES;
their loss surface lives on the gauge group, not in Euclidean space, so the geometrically
correct steepest-descent step is the NATURAL gradient under the metric the exponential map
induces on the chart, ``G_ab(phi) = <d exp_phi(T_a), d exp_phi(T_b)>_F`` (the
``pullback_per_block`` preconditioner).

A position-dependent metric cannot be realized by preconditioning the gradient and then
handing it to AdamW: Adam divides by the per-coordinate second moment, which re-flattens any
metric scaling. (The Killing metric is conformal -- in the Frobenius-orthonormal E_ij basis the
global ``killing`` inverse is exactly c*I on sl(K), so its natural gradient grad@(c*I) = c*grad
is a uniform scalar rescale: direction-preserving, cos(nat,grad)=1. ``killing_per_block`` applies
one such conformal factor c_h = 1/(2 d_h) per irrep block, so it is block-wise direction-preserving
and globally direction-preserving exactly when the blocks share that factor -- which V3's equal-size
irrep blocks guarantee; unequal blocks would reweight blocks against each other.) Because this
natural-grad M-step steps the gauge group manually (``p.add_(buf, alpha=-lr)``, then sets
``p.grad=None`` so AdamW never touches it), the conformal factor is NOT normalized away: here
killing/killing_per_block are a direction-preserving effective-LR rescale by that conformal factor,
NOT a no-op. They are an exact no-op only under Adam's per-coordinate scale-invariance, which this
branch deliberately bypasses. Only the non-conformal ``pullback`` metric reshapes the step DIRECTION.
So the gauge frame is stepped by natural-gradient descent with heavy-ball
momentum, while every non-gauge parameter (mu / sigma / decode / ...) keeps standard AdamW --
those carry diagonal Gaussian Fisher metrics that AdamW's per-coordinate adaptivity already
realizes, so explicit preconditioning there is the documented no-op.

``GaugeNaturalGradAdamW`` subclasses ``AdamW``: a param group flagged ``gauge=True`` is updated
by the natural-gradient rule in ``step``; its gradient is then consumed (set to ``None``) so the
trailing ``super().step()`` (standard AdamW) is a no-op on it and runs normally on every other
group. Only ACTIVE rows (nonzero gradient -- the batch's tokens) are preconditioned, so a step's
per-token metric solves touch the batch, not the whole vocabulary.
"""

from typing import List

import torch

from vfe3.geometry.phi_preconditioner import precondition_phi_gradient, pullback_metric_per_block


class GaugeNaturalGradAdamW(torch.optim.AdamW):
    r"""AdamW everywhere, natural-gradient + momentum on the gauge-frame coordinate groups.

    For a group flagged ``gauge=True`` holding coordinates ``phi`` (shape ``(..., n_gen)``), with
    ``nat = G(phi)^{-1} grad`` the natural gradient (active rows only), the step depends on
    ``gauge_update_rule``::

        # 'heavy_ball' (default):
        buf  = momentum * buf + nat
        phi -= lr * buf

        # 'adam':                          # Adam moments ON the natural gradient (betas/eps from the group)
        m    = beta1 * m + (1 - beta1) * nat
        v    = beta2 * v + (1 - beta2) * nat^2
        phi -= lr * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)

    ``'heavy_ball'`` passes the natural gradient through unnormalized: the metric DIRECTION survives
    but a tiny/badly-scaled phi gradient (and the metric inverse's shrink) barely moves the frame.
    ``'adam'`` restores per-coordinate ``1/sqrt(v)`` magnitude normalization so phi actually trains,
    while keeping the metric direction; for the conformal ``killing`` metric it collapses to plain
    AdamW on phi (the conformal factor cancels in ``m/sqrt(v)``), for ``pullback_per_block`` it is a
    hybrid that keeps part of the metric direction AND moves phi.

    ``G(phi)`` is the metric named by ``precond_mode`` (``pullback_per_block`` for the exact
    exp-map geometry). No weight decay is applied to the gauge frame (decoupled L2 in Euclidean
    coordinates would be non-geometric; the ``mass_phi`` penalty handles frame-norm shrinkage in
    the loss). The reduced-coordinate chart (``pos_phi_project_slk``) is NOT routed here -- the
    full-width pullback metric is shape-incompatible with a reduced gradient -- so those params
    stay on AdamW (the caller gates on width).
    """

    # Signature-convention exception: torch.optim.Optimizer's contract REQUIRES params as the
    # first positional argument (super().__init__(params, ...)), so the tensor `generators`
    # cannot lead here as the project convention would otherwise mandate.
    def __init__(
        self,
        params,
        generators:     torch.Tensor,         # (n_gen, K, K) gauge generator basis
        irrep_dims:     List[int],            # block sizes (sum == K); used by *_per_block metrics

        *,
        precond_mode:      str   = "pullback_per_block",
        gauge_momentum:    float = 0.9,
        gauge_update_rule: str   = "heavy_ball",
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self._generators     = generators
        self._irrep_dims     = irrep_dims
        self._precond_mode   = precond_mode
        self._gauge_momentum = float(gauge_momentum)
        # Moment rule for the natural-gradient gauge step: 'heavy_ball' (default; momentum only, no
        # per-coordinate normalization) or 'adam' (Adam m/v/bias-correction ON the natural gradient,
        # restoring 1/sqrt(v) normalization while keeping the metric direction).
        if gauge_update_rule not in ("heavy_ball", "adam"):
            raise ValueError(
                f"gauge_update_rule must be 'heavy_ball' or 'adam', got {gauge_update_rule!r}"
            )
        self._gauge_update_rule = gauge_update_rule
        # D1/EXP-8 training-time diagnostics, GATED: the caller (train.py) sets _collect_gauge_diag
        # True only on a log/eval step, so the silent hot path computes NOTHING extra. When set, step()
        # stashes cos(nat, grad) (1.0 for the conformal killing rescale; <1 when pullback reshapes the
        # direction) and -- on the pullback modes -- the per-token metric condition number into
        # _gauge_diag, which train.py reads into metrics.csv.
        self._collect_gauge_diag = False
        self._gauge_diag: dict   = {}

    def __setstate__(self, state) -> None:                             # type: ignore[override]
        r"""Restore generically, running Adam's step migration only where ``"step"`` exists.

        ``Optimizer.load_state_dict`` dispatches to ``__setstate__``; ``Adam.__setstate__`` migrates
        a legacy float ``"step"`` to a tensor and assumes EVERY non-empty per-parameter state carries
        ``"step"``. A gauge param's state holds only ``"gauge_mom"`` ('heavy_ball') or
        ``"gauge_m"``/``"gauge_v"``/``"gauge_step"`` ('adam') -- never ``"step"`` -- because
        :meth:`step` consumes its grad to ``None`` so base AdamW skips it. Adam's assumption would
        raise ``KeyError: 'step'`` and checkpoint RESUME would crash on the geometric M-step. Restore
        via the base ``Optimizer`` (which carries the current-format param-group hyperparameters from
        our own checkpoints) and run the float->tensor step migration only on states that actually
        have ``"step"`` (the non-gauge AdamW groups).
        """
        torch.optim.Optimizer.__setstate__(self, state)
        for s in self.state.values():
            step = s.get("step")
            if step is not None and not torch.is_tensor(step):
                s["step"] = torch.tensor(float(step))

    @torch.no_grad()
    def step(self, closure=None):                                      # type: ignore[override]
        collect = self._collect_gauge_diag                             # gated: True only on log/eval steps
        cos_acc: List[float] = []
        cond_acc: List[torch.Tensor] = []
        for group in self.param_groups:
            if not group.get("gauge", False):
                continue
            lr  = group["lr"]
            mom = self._gauge_momentum
            for p in group["params"]:
                if p.grad is None:
                    continue
                g   = p.grad
                Gd  = self._generators.to(device=g.device, dtype=g.dtype)
                # Flatten any leading axes to (rows, n_gen); precondition ACTIVE rows only.
                flat_g   = g.reshape(-1, g.shape[-1])
                flat_phi = p.data.reshape(-1, p.data.shape[-1])
                active   = flat_g.abs().sum(dim=-1) > 0                 # rows that received gradient
                nat = torch.zeros_like(flat_g)
                if bool(active.any()):
                    nat[active] = precondition_phi_gradient(
                        flat_g[active], flat_phi[active], Gd,
                        mode=self._precond_mode, irrep_dims=self._irrep_dims,
                    )
                    if collect:                                        # D1/EXP-8 diagnostics (sparse)
                        cos = torch.nn.functional.cosine_similarity(
                            nat[active], flat_g[active], dim=-1)        # (active,)
                        cos_acc.append(float(cos.mean()))
                        if self._precond_mode in ("pullback", "pullback_per_block"):
                            from vfe3.numerics import condition_number
                            Gm = pullback_metric_per_block(flat_phi[active], Gd, self._irrep_dims)
                            eye = torch.eye(Gm.shape[-1], dtype=Gm.dtype, device=Gm.device)
                            cond_acc.append(condition_number(Gm + 1e-6 * eye).reshape(-1))
                nat = nat.reshape_as(g)

                state = self.state[p]
                if self._gauge_update_rule == "adam":
                    # Adam moments (m, v, bias-correction) ON the natural gradient nat=G^-1 grad.
                    # betas/eps come from the AdamW param group (torch fills the group defaults), so
                    # this is literally AdamW applied to the preconditioned gradient: it restores the
                    # per-coordinate 1/sqrt(v) normalization that heavy-ball lacks while keeping the
                    # metric direction. Dense m/v over the whole table with a global step count mirror
                    # plain AdamW on phi_embed (inactive rows have nat=0: m/v just decay), so under the
                    # conformal killing metric this reproduces the AdamW arm; under pullback it keeps
                    # part of the metric direction. State key 'gauge_step' (int, not 'step') so it is
                    # untouched by Adam's float->tensor 'step' migration in __setstate__.
                    b1, b2 = group["betas"]
                    eps    = group["eps"]
                    m = state.get("gauge_m")
                    if m is None:
                        m = torch.zeros_like(nat); state["gauge_m"] = m
                        state["gauge_v"]    = torch.zeros_like(nat)
                        state["gauge_step"] = 0
                    v = state["gauge_v"]
                    state["gauge_step"] += 1
                    t = state["gauge_step"]
                    m.mul_(b1).add_(nat, alpha=1 - b1)                  # m <- b1*m + (1-b1)*nat
                    v.mul_(b2).addcmul_(nat, nat, value=1 - b2)         # v <- b2*v + (1-b2)*nat^2
                    mhat = m / (1 - b1 ** t)
                    vhat = v / (1 - b2 ** t)
                    p.add_(mhat / (vhat.sqrt() + eps), alpha=-lr)       # phi <- phi - lr*mhat/(sqrt(vhat)+eps)
                else:
                    buf = state.get("gauge_mom")
                    if buf is None:
                        buf = torch.zeros_like(nat)
                        state["gauge_mom"] = buf
                    buf.mul_(mom).add_(nat)                             # heavy-ball: m <- mom*m + nat
                    p.add_(buf, alpha=-lr)                             # phi <- phi - lr*m
                p.grad = None                                          # consumed: AdamW no-ops on it
        if collect:
            self._gauge_diag = {}
            if cos_acc:
                self._gauge_diag["cos_nat_phi"] = sum(cos_acc) / len(cos_acc)
            if cond_acc:
                allc = torch.cat(cond_acc)
                self._gauge_diag["pullback_cond_median"] = float(allc.median())
                self._gauge_diag["pullback_cond_max"]    = float(allc.max())
        return super().step(closure)
