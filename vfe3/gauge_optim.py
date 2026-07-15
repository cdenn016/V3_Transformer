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

from typing import Dict, List, Optional

import torch

from vfe3.geometry.phi_preconditioner import precondition_phi_gradient, pullback_metric_per_block


@torch.no_grad()
def project_phi_parameter_rows_(
    model:           torch.nn.Module,
    max_matrix_norm: float,

    *,
    chunk_rows:      int = 64,
) -> Dict[str, float]:
    r"""Project every trainable phi-table row to an embedded Frobenius-norm ball.

    Projection rescales algebra coordinates along their current ray and deliberately leaves
    optimizer moments unchanged. It covers belief token frames, independent model-channel frames,
    and both learned positional tables when present.
    """
    if not torch.isfinite(torch.tensor(max_matrix_norm)) or max_matrix_norm <= 0.0:
        raise ValueError(
            f"max_matrix_norm must be finite and positive, got {max_matrix_norm}"
        )
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError(f"chunk_rows must be a positive int, got {chunk_rows!r}")
    tables = [
        getattr(model.prior_bank, "phi_embed", None),
        getattr(model.prior_bank, "s_phi_embed", None),
        getattr(model, "pos_phi_free", None),
        getattr(model, "s_pos_phi_free", None),
    ]
    unique = []
    seen = set()
    for table in tables:
        if table is not None and id(table) not in seen:
            unique.append(table)
            seen.add(id(table))

    generators = model.group.generators
    total_rows = 0
    projected_rows = 0
    norm_max_before = 0.0
    minimum_scale = 1.0
    for table in unique:
        rows = table.reshape(-1, table.shape[-1])
        if rows.shape[-1] != generators.shape[0]:
            raise ValueError(
                "phi chart projection requires full generator coordinates; got table width "
                f"{rows.shape[-1]} and {generators.shape[0]} generators"
            )
        basis = generators.to(device=rows.device, dtype=rows.dtype)
        total_rows += rows.shape[0]
        for start in range(0, rows.shape[0], chunk_rows):
            chunk = rows[start : start + chunk_rows]
            embedded = torch.einsum("ra,aij->rij", chunk, basis)
            norm = torch.linalg.matrix_norm(embedded, ord="fro", dim=(-2, -1))
            scale = (max_matrix_norm / norm.clamp(min=1e-12)).clamp(max=1.0)
            projected_rows += int((scale < 1.0).sum())
            norm_max_before = max(norm_max_before, float(norm.max()))
            minimum_scale = min(minimum_scale, float(scale.min()))
            chunk.mul_(scale.unsqueeze(-1))
    return {
        "phi_chart_projected_rows":     float(projected_rows),
        "phi_chart_total_rows":         float(total_rows),
        "phi_chart_projected_fraction": projected_rows / max(total_rows, 1),
        "phi_chart_preproject_max":     norm_max_before,
        "phi_chart_projection_scale_min": minimum_scale,
    }


def _polar_orthogonalize(
    U: torch.Tensor,                      # (..., K, K) possibly drifted-off-O(K) element
) -> torch.Tensor:                        # (..., K, K) nearest orthogonal matrix
    r"""Nearest orthogonal matrix to ``U`` via the polar decomposition :math:`Q = U (U^T U)^{-1/2}`.

    Uses the SVD polar factor :math:`Q = W V^T` (:math:`U = W S V^T`), the exact Frobenius-norm
    minimizer of :math:`\lVert U - Q \rVert_F` over :math:`O(K)`. Runs in a float64 island
    (autocast disabled) so the drift correction itself does not introduce fresh fp32 rounding;
    keeps a drifted skew-group frame exactly on :math:`O(K)` so ``U^T`` stays the exact inverse
    (the transpose fast path :func:`build_transport_from_element` relies on for skew groups).
    """
    with torch.amp.autocast(U.device.type, enabled=False):
        W, _, Vh = torch.linalg.svd(U.double(), full_matrices=False)
        return (W @ Vh).to(U.dtype)


def _require_finite_nonsingular_omega(
    U: torch.Tensor,                      # (..., d, d) updated full elements or compact blocks
) -> None:
    r"""Fail immediately when an omega retraction produces a nonfinite or singular element."""
    with torch.no_grad(), torch.amp.autocast(U.device.type, enabled=False):
        if not bool(torch.isfinite(U).all()):
            raise FloatingPointError("omega retraction produced a nonfinite group element")
        sign, logabsdet = torch.linalg.slogdet(U.reshape(-1, U.shape[-1], U.shape[-1]).double())
        if bool((sign == 0).any()):
            raise ValueError("omega retraction produced a singular group element")
        if not bool(torch.isfinite(logabsdet).all()):
            raise FloatingPointError("omega retraction produced a nonfinite log-determinant")


def _omega_condition_values(
    U: torch.Tensor,                      # (..., d, d) updated full elements or compact blocks
) -> torch.Tensor:                        # (...) spectral condition number per element/block
    r"""Float64 condition of each represented element for log-cadence diagnostics.

    An untied compact row has shape ``(A, H, d, d)`` and represents the block diagonal
    ``diag(U_1, ..., U_H)``. Its largest singular value is the largest singular value over every
    block and its smallest is the smallest over every block; taking ``cond`` block-by-block would
    miss cross-block scale separation. Dense and tied ``(A, d, d)`` rows keep their ordinary matrix
    condition.
    """
    with torch.no_grad(), torch.amp.autocast(U.device.type, enabled=False):
        if U.dim() == 4:                                      # (A,H,d,d) untied compact elements
            singular = torch.linalg.svdvals(U.double())       # (A,H,d), descending within each block
            return singular[..., 0].amax(dim=-1) / singular[..., -1].amin(dim=-1)
        return torch.linalg.cond(U.reshape(-1, U.shape[-1], U.shape[-1]).double())


def _symplectic_membership_residual(
    U: torch.Tensor,                      # (..., K, K), K even, defining Sp(K,R) representation
) -> torch.Tensor:                        # (...) relative ||U^T J U - J||_F
    r"""Relative defining-representation symplectic residual ``||U^T J U-J||_F/||J||_F``."""
    K = U.shape[-1]
    if K % 2 != 0:
        raise ValueError(f"symplectic omega diagnostic requires even K, got K={K}")
    with torch.no_grad(), torch.amp.autocast(U.device.type, enabled=False):
        m = K // 2
        J = torch.zeros(K, K, device=U.device, dtype=torch.float64)
        eye = torch.eye(m, device=U.device, dtype=torch.float64)
        J[:m, m:] = eye
        J[m:, :m] = -eye
        U64 = U.double()
        error = U64.transpose(-1, -2) @ J @ U64 - J
        return error.norm(dim=(-2, -1)) / J.norm()


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
        precond_mode:       str           = "pullback_per_block",
        gauge_momentum:     float         = 0.9,
        gauge_update_rule:  str           = "heavy_ball",
        omega_retract_mode: str           = "lie_exp",
        skew_symmetric:     bool          = False,
        omega_reorth_every: int           = 0,
        group_name:         Optional[str] = None,
        **kwargs,
    ) -> None:
        if type(omega_reorth_every) is not int or omega_reorth_every < 0:
            raise ValueError(
                "omega_reorth_every must be a nonnegative int, got "
                f"{type(omega_reorth_every).__name__}: {omega_reorth_every!r}"
            )
        if group_name is not None and (not isinstance(group_name, str) or not group_name):
            raise ValueError(f"group_name must be a nonempty str or None, got {group_name!r}")
        super().__init__(params, **kwargs)
        self._generators        = generators
        self._irrep_dims        = irrep_dims
        self._precond_mode      = precond_mode
        self._gauge_momentum    = float(gauge_momentum)
        # Group-manifold retraction for the omega_direct group (params flagged omega=True): 'lie_exp'
        # (matrix_exp; follows the one-parameter subgroup) or 'cayley' (exp-free (I-A/2)^{-1}(I+A/2)).
        self._omega_retract_mode = omega_retract_mode
        # Orthogonality-drift control for a SKEW (orthogonal, e.g. so_k/so_n) omega_direct group:
        # fp32 accumulation of exp(skew) retraction products walks the stored U off O(K) over many
        # M-steps, after which U^T stops being the exact inverse (the transpose fast path
        # build_transport_from_element relies on for skew groups). skew_symmetric mirrors
        # model.group.skew_symmetric (threaded from build_optimizer); omega_reorth_every>0 turns on a
        # periodic polar re-orthogonalization every that many M-steps. Default 0 = off = byte-identical.
        self._skew_symmetric     = bool(skew_symmetric)
        self._omega_reorth_every = omega_reorth_every
        self._omega_step         = 0                     # M-step counter for the reorth cadence
        self._group_name         = group_name
        self._has_omega_group    = any(group.get("omega", False) for group in self.param_groups)
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
        # _gauge_diag. Omega-direct groups additionally report active-row element condition, or the
        # defining Sp(K,R) membership residual for ``sp``. train.py reads the fixed keys into metrics.csv.
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

    def state_dict(self) -> Dict[str, object]:                         # type: ignore[override]
        r"""Serialize AdamW state plus the M-step count that drives omega reorthogonalization."""
        state = super().state_dict()
        state["optimizer_extra"] = {
            "omega_step":         int(self._omega_step),
            "omega_dirty_format": 1,
        }
        return state

    def load_state_dict(
        self,
        state_dict: Dict[str, object],
    ) -> None:                                                        # type: ignore[override]
        r"""Restore AdamW state, omega cadence, and versioned dirty-row masks.

        Pre-O5 checkpoints may carry ``omega_step`` but no dirty-mask format marker. If a live
        single-block skew cadence resumes from that format, every row is conservatively marked dirty:
        the missing mask cannot prove which rows accumulated drift before the checkpoint. Current
        checkpoints carry ``omega_dirty_format=1``; a missing mask in that format means clean.
        """
        core_state = dict(state_dict)
        extra = core_state.pop("optimizer_extra", None)
        super().load_state_dict(core_state)
        if isinstance(extra, dict) and "omega_step" in extra:
            self._omega_step = int(extra["omega_step"])
        else:
            import warnings
            self._omega_step = 0
            warnings.warn(
                "GaugeNaturalGradAdamW checkpoint has no optimizer_extra.omega_step; the omega "
                "reorthogonalization cadence restarts at zero (non-exact resume when "
                "omega_reorth_every > 0).",
                UserWarning,
                stacklevel=2,
            )

        current_dirty_format = isinstance(extra, dict) and extra.get("omega_dirty_format") == 1
        legacy_skew_cadence = (
            not current_dirty_format
            and self._has_omega_group
            and self._skew_symmetric
            and self._omega_reorth_every > 0
            and len(self._irrep_dims) == 1
        )
        for group in self.param_groups:
            if not group.get("omega", False):
                continue
            for p in group["params"]:
                dirty = self.state[p].get("omega_dirty")
                if dirty is None:
                    dirty = torch.full(
                        (p.shape[0],), legacy_skew_cadence,
                        dtype=torch.bool, device=p.device)
                else:
                    if dirty.shape != (p.shape[0],):
                        raise ValueError(
                            f"optimizer omega_dirty has shape {tuple(dirty.shape)}, expected "
                            f"({p.shape[0]},)")
                    dirty = dirty.to(device=p.device, dtype=torch.bool)
                    if legacy_skew_cadence:
                        dirty.fill_(True)
                self.state[p]["omega_dirty"] = dirty
        if legacy_skew_cadence:
            import warnings
            warnings.warn(
                "GaugeNaturalGradAdamW checkpoint predates optimizer_extra.omega_dirty_format; "
                "all omega rows were marked dirty so the next scheduled orthogonal projection "
                "cannot silently skip pre-checkpoint drift.",
                UserWarning,
                stacklevel=2,
            )

    def _compact_gld_basis(
        self,
        d:      int,                          # block dimension
        device: torch.device,
        dtype:  torch.dtype,
    ) -> 'tuple[torch.Tensor, torch.Tensor]':  # (d*d, d, d) gl(d) basis, (d*d, d*d) its gram_pinv
        r"""The reduced ``gl(d)`` generator basis + Gram pseudo-inverse for the per-block compact
        retraction, built and cached ONCE (per d, device, dtype).

        A compact ``omega_direct`` table is (V, H, d, d) / (V, d, d) blocks; each block steps on GL(d)
        under the full ``gl(d)`` elementary basis E_ij (``generate_glk(d)``, an orthonormal (d*d, d, d)
        set). This is the block-restriction of the full block-diagonal ``gl(K)`` basis, so the per-block
        step equals the full K x K step on the blocks. The Gram pseudo-inverse is cached alongside so
        ``extract_phi`` never recomputes it per step or per param (FIX 3).
        """
        cache = getattr(self, "_gld_cache", None)
        if cache is None:
            cache = self._gld_cache = {}
        key = (d, device, dtype)
        entry = cache.get(key)
        if entry is None:
            from vfe3.geometry.generators import generate_glk
            from vfe3.geometry.lie_ops import gram_pinv
            G = generate_glk(d, device=device, dtype=dtype)
            entry = cache[key] = (G, gram_pinv(G))
        return entry

    @torch.no_grad()
    def step(self, closure=None):                                      # type: ignore[override]
        if closure is not None:
            # The gauge groups are stepped and their grads consumed (set to None) BELOW, before the
            # trailing super().step(). A closure that re-evaluates the loss would repopulate those grads
            # and let base AdamW step the frame a SECOND time. No caller passes a closure (GradScaler.step
            # and a bare optimizer.step() both call with closure=None), so reject it rather than risk the
            # double-step.
            raise NotImplementedError(
                "GaugeNaturalGradAdamW does not support closure-based steps: the gauge gradient is "
                "consumed before super().step(), so a closure that re-backpropagates would double-step "
                "the frame. Call step() with no closure."
            )
        collect = self._collect_gauge_diag                             # gated: True only on log/eval steps
        if collect:
            self._gauge_diag = {}                                      # never expose a prior attempted step
        cos_acc: List[float] = []
        cond_acc: List[torch.Tensor] = []
        omega_cond_acc: List[torch.Tensor] = []
        omega_sp_acc: List[torch.Tensor] = []
        for group in self.param_groups:
            if group.get("omega", False):
                # omega_direct group: the params ARE stored GL(K) group elements U (shape (V, K, K)),
                # not algebra coordinates. Step by a group-manifold retraction of the natural-gradient
                # tangent xi = Gram^{-1} proj_g(U^T E) (extract_phi computes exactly this): U <- U retr(-lr xi),
                # active (nonzero-grad) rows only, then consume the grad so base AdamW no-ops on it.
                # Compact storage (omega_compact_storage=True): the table is per-block stacks -- untied
                # (V, H, d, d) has dim 4, tied (V, d, d) has dim 3 with block dim d < K (a full table is
                # (V, K, K), dim 3 with last dim == K). Both step each block by the reduced gl(d) basis
                # (a K x K solve/exp per block collapses to small d x d ones), which equals the full
                # block-diagonal gl(K) step restricted to the blocks (the gl(K) generators have disjoint
                # block support). Detecting tied by dim ALONE mis-routes (V,d,d) to the full path and
                # crashes extract_phi with an einsum size mismatch -- hence the d < K test.
                lr     = group["lr"]
                mode   = getattr(self, "_omega_retract_mode", "lie_exp")
                K_full = self._generators.shape[-1]                    # full frame dim K
                from vfe3.geometry.lie_ops import extract_phi, gram_pinv, retract_omega
                full_gp = None                                         # gram_pinv(full basis), built once per step
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    E   = p.grad                                       # (V,K,K) full or (V,H,d,d)/(V,d,d) compact
                    U   = p.data                                       # matching stored group elements
                    untied_compact = U.dim() == 4                      # (V,H,d,d)
                    tied_compact   = U.dim() == 3 and U.shape[-1] < K_full  # (V,d,d), d < K
                    act = E.reshape(E.shape[0], -1).abs().sum(dim=-1) > 0   # (V,) rows with gradient
                    if not bool(act.any()):
                        p.grad = None
                        continue
                    Ua, Ea = U[act], E[act]                            # (A,K,K) / (A,H,d,d) / (A,d,d)
                    _require_finite_nonsingular_omega(Ua)
                    if untied_compact or tied_compact:
                        d      = Ua.shape[-1]
                        Gd, gp = self._compact_gld_basis(d, U.device, U.dtype)   # gl(d) basis + cached gram_pinv
                        # Untied: flatten the H block axis into the row axis so each of the A*H blocks
                        # retracts independently. Tied: the active rows ARE the single (A,d,d) shared
                        # blocks (encode's broadcast adjoint already SUMMED the H per-slot grads onto
                        # them), so reshape(-1,d,d) is a no-op and each shared block retracts directly.
                        Ua_r = Ua.reshape(-1, d, d)                    # (A*H,d,d) untied / (A,d,d) tied
                        Ea_r = Ea.reshape(-1, d, d)
                        # per-block natural-gradient tangent xi = Gram^{-1} proj_gl(d)(U_h^T E_h)
                        xi = extract_phi(torch.einsum("...lk,...lm->...km", Ua_r, Ea_r), Gd, gram_pinv_=gp)
                        if tied_compact:
                            # The full tied generators kron(I_H, E_ij) are nonzero in ALL H blocks, so
                            # their Frobenius Gram is H*I and the full-tied natural gradient carries a
                            # 1/H factor. The reduced gl(d) basis has Gram = I (no 1/H), and encode's
                            # broadcast adjoint already SUMMED the H per-slot grads onto the shared block,
                            # so WITHOUT this rescale the compact-tied step is H x too large. Divide by
                            # H = K/d so the compact-tied retraction MATCHES the full-tied step exactly.
                            # (The untied gl(d)^H generators are Gram = I, already correct -- do NOT scale.)
                            xi = xi / (K_full // d)
                        Ur = retract_omega(Ua_r, -lr * xi, Gd, mode=mode)
                        Ur = Ur.reshape(Ua.shape)                      # back to (A,H,d,d) / (A,d,d)
                        _require_finite_nonsingular_omega(Ur)
                        U[act] = Ur
                    else:
                        Gd = self._generators.to(device=U.device, dtype=U.dtype)
                        if full_gp is None:
                            full_gp = gram_pinv(Gd)                    # (n_gen, n_gen) cached per step
                        # natural-gradient tangent xi = Gram^{-1} proj_g(U^T E); (U^T E)_{km} = sum_l U_{lk} E_{lm}
                        xi = extract_phi(torch.einsum("...lk,...lm->...km", Ua, Ea), Gd, gram_pinv_=full_gp)
                        Ur = retract_omega(Ua, -lr * xi, Gd, mode=mode)
                        _require_finite_nonsingular_omega(Ur)
                        U[act] = Ur                                    # (A, K, K) U <- U retr(-lr xi)
                    state = self.state[p]
                    dirty = state.get("omega_dirty")
                    if dirty is None:
                        dirty = torch.zeros(U.shape[0], dtype=torch.bool, device=U.device)
                        state["omega_dirty"] = dirty
                    elif dirty.shape != (U.shape[0],):
                        raise ValueError(
                            f"optimizer omega_dirty has shape {tuple(dirty.shape)}, expected "
                            f"({U.shape[0]},)")
                    elif dirty.device != U.device or dirty.dtype != torch.bool:
                        dirty = dirty.to(device=U.device, dtype=torch.bool)
                        state["omega_dirty"] = dirty
                    dirty.logical_or_(act)
                    if collect:
                        updated = U[act]
                        if self._group_name == "sp" and not (untied_compact or tied_compact):
                            omega_sp_acc.append(_symplectic_membership_residual(updated).reshape(-1))
                        else:
                            omega_cond_acc.append(_omega_condition_values(updated).reshape(-1))
                    p.grad = None                                      # consumed: AdamW no-ops on it
                # Orthogonality-drift control (default OFF: omega_reorth_every=0 -> byte-identical).
                # Polar reorth guarantees O(K) membership, which equals the structure group ONLY for
                # the single-block defining rep (so_k: rho(SO(K)) = SO(K), so the nearest-O(K) matrix
                # _polar_orthogonalize returns is exactly the correct projection). For an irrep TOWER
                # (so_n/sp_n, len(irrep_dims) > 1) the stored omega_embed is a faithful rho(SO(N))
                # IMAGE -- a proper submanifold of O(K) -- so the nearest-O(K) matrix is NOT guaranteed
                # to stay in that image; firing reorth there would silently relax the structure group.
                # Towers are therefore excluded and left a no-op: their bounded drift is left to the
                # retraction's own second-order stability (a faithful rho-image projection is
                # deferred). Cadence is counted in M-steps (this branch runs once per optimizer.step()
                # call), not in active-row updates, so drift accumulated over past steps is corrected
                # on schedule even on a step with no gradient. The compact block groups
                # (block_glk/tied_block_glk) are non-skew, so this never fires for a (V,H,d,d)/(V,d,d)
                # table regardless of the block-count gate below.
                if (self._skew_symmetric and self._omega_reorth_every > 0
                        and len(self._irrep_dims) == 1):
                    self._omega_step += 1
                    if self._omega_step % self._omega_reorth_every == 0:
                        for p in group["params"]:
                            dirty = self.state[p].get("omega_dirty")
                            if dirty is None:
                                continue
                            if dirty.shape != (p.shape[0],):
                                raise ValueError(
                                    f"optimizer omega_dirty has shape {tuple(dirty.shape)}, expected "
                                    f"({p.shape[0]},)")
                            if dirty.device != p.device or dirty.dtype != torch.bool:
                                dirty = dirty.to(device=p.device, dtype=torch.bool)
                                self.state[p]["omega_dirty"] = dirty
                            if not bool(dirty.any()):
                                continue
                            projected = _polar_orthogonalize(p.data[dirty])
                            _require_finite_nonsingular_omega(projected)
                            p.data[dirty] = projected
                            dirty.zero_()
                continue
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
                            cond_acc.append(
                                condition_number(Gm + 1e-6 * eye, kind="full").reshape(-1)
                            )
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
            if cos_acc:
                self._gauge_diag["cos_nat_phi"] = sum(cos_acc) / len(cos_acc)
            if cond_acc:
                allc = torch.cat(cond_acc)
                self._gauge_diag["pullback_cond_median"] = float(allc.median())
                self._gauge_diag["pullback_cond_max"]    = float(allc.max())
            if omega_cond_acc:
                allc = torch.cat(omega_cond_acc)
                self._gauge_diag["omega_condition_median"] = float(allc.median())
                self._gauge_diag["omega_condition_max"]    = float(allc.max())
            if omega_sp_acc:
                allr = torch.cat(omega_sp_acc)
                self._gauge_diag["omega_symplectic_residual_median"] = float(allr.median())
                self._gauge_diag["omega_symplectic_residual_max"]    = float(allr.max())
        return super().step()                                          # closure already rejected above
