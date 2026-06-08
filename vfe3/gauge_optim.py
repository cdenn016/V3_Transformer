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

from vfe3.geometry.phi_preconditioner import precondition_phi_gradient


class GaugeNaturalGradAdamW(torch.optim.AdamW):
    r"""AdamW everywhere, natural-gradient + momentum on the gauge-frame coordinate groups.

    For a group flagged ``gauge=True`` holding coordinates ``phi`` (shape ``(..., n_gen)``)::

        nat  = G(phi)^{-1} grad          # natural gradient (active rows only)
        buf  = momentum * buf + nat      # heavy-ball momentum
        phi -= lr * buf

    ``G(phi)`` is the metric named by ``precond_mode`` (``pullback_per_block`` for the exact
    exp-map geometry). No weight decay is applied to the gauge frame (decoupled L2 in Euclidean
    coordinates would be non-geometric; the ``mass_phi`` penalty handles frame-norm shrinkage in
    the loss). The reduced-coordinate chart (``pos_phi_project_slk``) is NOT routed here -- the
    full-width pullback metric is shape-incompatible with a reduced gradient -- so those params
    stay on AdamW (the caller gates on width).
    """

    def __init__(
        self,
        params,
        generators:     torch.Tensor,         # (n_gen, K, K) gauge generator basis
        irrep_dims:     List[int],            # block sizes (sum == K); used by *_per_block metrics

        *,
        precond_mode:   str   = "pullback_per_block",
        gauge_momentum: float = 0.9,
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self._generators     = generators
        self._irrep_dims     = irrep_dims
        self._precond_mode   = precond_mode
        self._gauge_momentum = float(gauge_momentum)

    def __setstate__(self, state) -> None:                             # type: ignore[override]
        r"""Restore generically, running Adam's step migration only where ``"step"`` exists.

        ``Optimizer.load_state_dict`` dispatches to ``__setstate__``; ``Adam.__setstate__`` migrates
        a legacy float ``"step"`` to a tensor and assumes EVERY non-empty per-parameter state carries
        ``"step"``. A gauge param's state holds ONLY ``"gauge_mom"`` -- AdamW skips it because
        :meth:`step` consumes its grad to ``None`` -- so Adam's assumption raises ``KeyError: 'step'``
        and checkpoint RESUME would crash on the geometric M-step. Restore via the base ``Optimizer``
        (which carries the current-format param-group hyperparameters from our own checkpoints) and
        run the float->tensor step migration only on states that actually have ``"step"``.
        """
        torch.optim.Optimizer.__setstate__(self, state)
        for s in self.state.values():
            step = s.get("step")
            if step is not None and not torch.is_tensor(step):
                s["step"] = torch.tensor(float(step))

    @torch.no_grad()
    def step(self, closure=None):                                      # type: ignore[override]
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
                nat = nat.reshape_as(g)

                state = self.state[p]
                buf = state.get("gauge_mom")
                if buf is None:
                    buf = torch.zeros_like(nat)
                    state["gauge_mom"] = buf
                buf.mul_(mom).add_(nat)                                 # heavy-ball: m <- mom*m + nat
                p.add_(buf, alpha=-lr)                                  # phi <- phi - lr*m
                p.grad = None                                           # consumed: AdamW no-ops on it
        return super().step(closure)
