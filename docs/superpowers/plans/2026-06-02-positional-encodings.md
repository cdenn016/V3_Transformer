# Positional Encodings (gauge-RoPE + BCH-PE) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two default-off positional-encoding seams to VFE_3.0 — BCH positional encoding (a per-position Lie-algebra element composed into the gauge frame) and gauge-RoPE (a block-diagonal positional rotation folded into the transport operator).

**Architecture:** BCH-PE composes `pos_phi` into `belief.phi` after encode via the existing `compose_phi(mode="bch")`; it is a tiny forward-side change plus a `pos_phi` registry. gauge-RoPE introduces a `RopeTransport` container that wraps the built transport; `transport_mean`/`transport_covariance` dispatch on it, so the gradient kernel and autograd oracle (which consume the built `omega` opaquely) need no changes. The rotation `R(theta)` threads from `forward`/`diagnostics`/`attention_maps` down to the transport build sites.

**Tech Stack:** Python, PyTorch (float32, CUDA-capable), pytest. Reuses `vfe3/geometry/lie_ops.py::compose_phi`, `project_phi_to_slk`, and `vfe3/geometry/transport.py`.

**Spec:** `docs/superpowers/specs/2026-06-02-positional-encodings-design.md`. Tests are property-based, not VFE_2.0 byte-parity.

**Ordering note:** Part 1 (BCH-PE) ships first — smaller surface, no gradient-path changes. Part 2 (gauge-RoPE) builds on a clean tree. Each task ends green and is committed.

**Conventions for the executor:**
- Run the full suite with `python -m pytest` (do NOT add `-q`; `pyproject.toml` already sets it and a second makes `-qq`, which hides the pass count). For a single test: `python -m pytest tests/test_x.py::test_name`.
- The function-signature convention (CLAUDE.md): tensors first, then `float|Tensor`, then plain floats/ints/bools, then defined-default floats/ints/bools, then Optional, then `**kwargs`; vertically aligned names/types/`=`/comments. Match it.
- Branch is `vfe3-positional-encodings-2026-06-02` (already created from origin/main).

## Part 1: BCH positional encoding

### Task 1.1: The `pos_phi` registry module

**Files:**
- Create: `vfe3/model/positional_phi.py`
- Test: `tests/test_positional_phi.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_positional_phi.py
import torch

from vfe3.geometry.groups import get_group
from vfe3.model.positional_phi import (
    get_pos_phi, positional_phi_coords, apply_positional_phi,
)


def _glk_group(k=4):
    return get_group("glk")(k)


def test_none_returns_none_coords():
    coords = positional_phi_coords("none", 5, 3, device=torch.device("cpu"), dtype=torch.float32)
    assert coords is None


def test_frozen_coords_are_position_times_scale_on_one_axis():
    coords = positional_phi_coords("frozen", 4, 3, scale=0.1, frozen_axis=0,
                                   device=torch.device("cpu"), dtype=torch.float32)
    assert coords.shape == (4, 3)
    assert torch.allclose(coords[:, 0], torch.tensor([0.0, 0.1, 0.2, 0.3]))
    assert torch.allclose(coords[:, 1:], torch.zeros(4, 2))


def test_learned_coords_slice_the_table():
    table = torch.randn(8, 3)
    coords = positional_phi_coords("learned", 4, 3, pos_phi_free=table,
                                   device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(coords, table[:4])


def test_apply_none_is_identity():
    g = _glk_group()
    phi = torch.randn(2, 5, g.generators.shape[0])
    out = apply_positional_phi(phi, g, mode="none")
    assert torch.equal(out, phi)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_positional_phi.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'vfe3.model.positional_phi'`.

- [ ] **Step 3: Write the module**

```python
# vfe3/model/positional_phi.py
r"""Per-position gauge-frame positional encoding (BCH-PE) for VFE_3.0.

A registry of per-position Lie-algebra coordinate builders ``pos_phi_i in R^{n_gen}``
that are composed into the token gauge frame via :func:`vfe3.geometry.lie_ops.compose_phi`
BEFORE transport, so position enters through the gauge transport ``Omega_ij`` (the
self-transport ``Omega_ii = I`` is unaffected). Default-off: ``"none"`` returns no
coordinates and the frame is unchanged. ``"learned"`` slices a model-owned parameter
table; ``"frozen"`` is the parameter-free ``i * scale`` on one generator axis (a
Lie-algebra ALiBi).
"""

from typing import Callable, Dict, Optional

import torch

from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.lie_ops import compose_phi, project_phi_to_slk

_POS_PHI: Dict[str, Callable[..., Optional[torch.Tensor]]] = {}


def register_pos_phi(name: str) -> Callable:
    """Decorator registering a pos-phi coordinate builder -> (N, n_gen) coords or None."""
    def _wrap(fn: Callable[..., Optional[torch.Tensor]]) -> Callable[..., Optional[torch.Tensor]]:
        _POS_PHI[name] = fn
        return fn
    return _wrap


def get_pos_phi(name: str) -> Callable[..., Optional[torch.Tensor]]:
    """Return the registered pos-phi builder (KeyError-with-available-list if absent)."""
    if name not in _POS_PHI:
        raise KeyError(f"no pos_phi {name!r}; available: {sorted(_POS_PHI)}")
    return _POS_PHI[name]


@register_pos_phi("none")
def _pos_phi_none(
    n:     int,
    n_gen: int,

    *,
    device: torch.device,
    dtype:  torch.dtype = torch.float32,
    **kwargs,
) -> Optional[torch.Tensor]:
    r"""No positional element: returns None (the frame is left unchanged)."""
    return None


@register_pos_phi("learned")
def _pos_phi_learned(
    n:     int,
    n_gen: int,

    *,
    pos_phi_free: Optional[torch.Tensor] = None,   # (max_seq_len, n_gen) model-owned table
    device:       torch.device,
    dtype:        torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Learned absolute positional coords: the first ``n`` rows of the model's table."""
    if pos_phi_free is None:
        raise ValueError("pos_phi='learned' requires the model-owned pos_phi_free table")
    return pos_phi_free[:n]


@register_pos_phi("frozen")
def _pos_phi_frozen(
    n:     int,
    n_gen: int,

    *,
    scale:       float = 0.02,
    frozen_axis: int   = 0,
    device:      torch.device,
    dtype:       torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Parameter-free Lie-algebra ALiBi: pos_phi_i = (i * scale) on one generator axis."""
    coords = torch.zeros(n, n_gen, device=device, dtype=dtype)
    coords[:, frozen_axis] = torch.arange(n, device=device, dtype=dtype) * scale
    return coords


def positional_phi_coords(
    mode:  str,
    n:     int,
    n_gen: int,

    *,
    scale:        float = 0.02,
    frozen_axis:  int   = 0,
    pos_phi_free: Optional[torch.Tensor] = None,
    device:       torch.device,
    dtype:        torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    r"""Dispatch to the registered pos-phi builder ``mode``; returns (N, n_gen) coords or None."""
    return get_pos_phi(mode)(
        n, n_gen, scale=scale, frozen_axis=frozen_axis,
        pos_phi_free=pos_phi_free, device=device, dtype=dtype,
    )


def apply_positional_phi(
    phi:   torch.Tensor,                  # (..., N, n_gen) token gauge frame
    group: GaugeGroup,

    *,
    mode:         str   = "none",
    compose_mode: str   = "bch",
    order:        int   = 4,
    scale:        float = 0.02,
    frozen_axis:  int   = 0,
    project_slk:  bool  = False,
    pos_phi_free: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Compose the per-position element into ``phi`` via ``compose_phi`` (BCH by default).

    ``"none"`` returns ``phi`` unchanged (byte-identical pure path). Otherwise the (N, n_gen)
    coords broadcast over any leading batch axis through ``compose_phi``. ``project_slk`` removes
    the per-block trace from the positional element so ``det(Omega_h) = 1`` is preserved.
    """
    n, n_gen = phi.shape[-2], phi.shape[-1]
    coords = positional_phi_coords(
        mode, n, n_gen, scale=scale, frozen_axis=frozen_axis,
        pos_phi_free=pos_phi_free, device=phi.device, dtype=phi.dtype,
    )
    if coords is None:
        return phi
    if project_slk:
        coords = project_phi_to_slk(coords, group.generators, group.irrep_dims)
    return compose_phi(phi, coords, group.generators, order=order, mode=compose_mode)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_positional_phi.py`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add vfe3/model/positional_phi.py tests/test_positional_phi.py
git commit -m "feat(positional): pos_phi registry (none/learned/frozen) + apply_positional_phi (BCH-PE core)"
```

### Task 1.2: Config fields + validation for BCH-PE

**Files:**
- Modify: `vfe3/config.py` (add fields near the gauge seam; add `_VALID_POS_PHI_COMPOSE`; add validation)
- Test: `tests/test_config.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py  (append; imports VFE3Config already present in this file)
import pytest
from vfe3.config import VFE3Config


def test_pos_phi_defaults_off_and_validates():
    cfg = VFE3Config()
    assert cfg.pos_phi == "none"
    assert cfg.pos_phi_compose == "bch"
    assert cfg.bch_pe_order == 4

def test_pos_phi_rejects_unknown_mode():
    with pytest.raises(ValueError):
        VFE3Config(pos_phi="banana")

def test_pos_phi_compose_rejects_unknown():
    with pytest.raises(ValueError):
        VFE3Config(pos_phi_compose="quaternion")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_config.py::test_pos_phi_defaults_off_and_validates`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'pos_phi'` (or AttributeError on `cfg.pos_phi`).

- [ ] **Step 3: Add fields and validation**

In `vfe3/config.py`, add the compose-validator tuple near the other `_VALID_*` tuples (after `_VALID_PHI_RETRACT_MODES`):

```python
_VALID_POS_PHI_COMPOSE     = ("bch", "euclidean")
```

Add fields in the gauge-seam group (after `cross_couplings`, before `gauge` ends / near `diagonal_covariance`):

```python
    # BCH positional encoding (default-off): a per-position Lie-algebra element pos_phi_i composed
    # into the token gauge frame via compose_phi BEFORE transport. "learned" owns a model parameter
    # table (max_seq_len, n_gen); "frozen" is the parameter-free i*pos_phi_scale on one axis. The
    # pure path is "none" (no composition). Validated against the pos_phi registry.
    pos_phi:                   str   = "none"      # "none" | "learned" | "frozen"
    pos_phi_compose:           str   = "bch"       # composition chart: bch (default) | euclidean
    bch_pe_order:              int   = 4           # BCH Dynkin truncation order (compose_phi order)
    pos_phi_scale:             float = 0.02        # learned-table init scale AND frozen per-position step
    pos_phi_project_slk:       bool  = False       # per-block trace projection (det Omega = 1)
```

Add validation in `__post_init__` (after the `phi_retract_mode` `_require`, around line 392). The `pos_phi` mode validates against the registry (import locally to avoid a module cycle, mirroring how `_TRANSPORTS` is used):

```python
        from vfe3.model.positional_phi import _POS_PHI
        _require(self.pos_phi, tuple(sorted(_POS_PHI)), "pos_phi")
        _require(self.pos_phi_compose, _VALID_POS_PHI_COMPOSE, "pos_phi_compose")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_config.py -k pos_phi`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add vfe3/config.py tests/test_config.py
git commit -m "feat(config): BCH-PE fields (pos_phi/compose/order/scale/project_slk) + validation"
```

### Task 1.3: Wire BCH-PE into the model (forward, diagnostics, attention_maps) + the learned parameter

**Files:**
- Modify: `vfe3/model/model.py` (create `pos_phi_free` in `__init__`; apply in `forward`, `diagnostics`, `attention_maps`)
- Test: `tests/test_positional_phi.py` (append model-level tests)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_positional_phi.py  (append)
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _cfg(**kw):
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                warmup_steps=1, max_steps=4)
    base.update(kw)
    return VFE3Config(**base)


def test_pos_phi_none_logits_byte_identical_to_no_field():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    m = VFEModel(_cfg(pos_phi="none"))
    logits_a = m(x)
    logits_b = m(x)
    assert torch.equal(logits_a, logits_b)              # determinism guard
    assert not hasattr(m, "pos_phi_free")               # no parameter created on the pure path


def test_pos_phi_learned_creates_parameter_and_changes_logits():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    base = VFEModel(_cfg(pos_phi="none"))
    learned = VFEModel(_cfg(pos_phi="learned", pos_phi_scale=0.3))
    learned.load_state_dict(base.state_dict(), strict=False)   # share priors; pos_phi_free is extra
    assert hasattr(learned, "pos_phi_free")
    assert learned.pos_phi_free.shape == (8, base.group.generators.shape[0])
    # A non-zero pos_phi_free perturbs the frame, hence the logits.
    with torch.no_grad():
        learned.pos_phi_free.add_(0.2)
    assert not torch.allclose(base(x), learned(x), atol=1e-5)


def test_pos_phi_learned_receives_gradient():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    y = torch.randint(0, 6, (2, 8))
    m = VFEModel(_cfg(pos_phi="learned", pos_phi_scale=0.3))
    with torch.no_grad():
        m.pos_phi_free.add_(0.1)                          # off the zero init so the grad is non-trivial
    _, loss, _ = m(x, y)
    loss.backward()
    assert m.pos_phi_free.grad is not None
    assert m.pos_phi_free.grad.abs().sum() > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_positional_phi.py -k "pos_phi_learned_creates or pos_phi_none_logits"`
Expected: FAIL (`pos_phi_free` not created; logits unaffected).

- [ ] **Step 3: Implement the model wiring**

In `vfe3/model/model.py`:

(a) Add the import near the other model imports (top of file):

```python
from vfe3.model.positional_phi import apply_positional_phi
```

(b) In `__init__`, after the `_log_prior_cache` line (around line 152), create the learned table only when selected:

```python
        # BCH positional encoding (default-off): a learned per-position Lie-algebra element table
        # composed into the gauge frame before transport. Created ONLY for pos_phi='learned' (a raw
        # nn.Parameter like log_alpha/connection_W, not a network); the "none"/"frozen" paths add no
        # parameter, so the pure path stays param-free. Init scaled by pos_phi_scale.
        if cfg.pos_phi == "learned":
            self.pos_phi_free = nn.Parameter(
                torch.randn(cfg.max_seq_len, n_gen) * cfg.pos_phi_scale)
            if cfg.detach_e_step:
                # Footgun (mirrors log_alpha / connection_W): pos_phi_free enters the loss ONLY
                # through the E-step belief transport, which detach_e_step wraps in no_grad, so the
                # positional table receives no gradient and stays frozen at init. Set
                # detach_e_step=False to learn it.
                import warnings
                warnings.warn(
                    "pos_phi='learned' with detach_e_step=True freezes pos_phi_free: the positional "
                    "gauge element enters the loss only through the E-step transport, which the "
                    "detached (no_grad) E-step severs. Set detach_e_step=False to train it.",
                    stacklevel=2,
                )
```

(c) Add a small private helper (method on VFEModel) used by forward/diagnostics/attention_maps so the call is identical in all three:

```python
    def _apply_pos_phi(self, phi: torch.Tensor) -> torch.Tensor:
        r"""Compose the configured BCH positional element into the gauge frame (no-op for 'none')."""
        if self.cfg.pos_phi == "none":
            return phi
        return apply_positional_phi(
            phi, self.group,
            mode=self.cfg.pos_phi, compose_mode=self.cfg.pos_phi_compose,
            order=self.cfg.bch_pe_order, scale=self.cfg.pos_phi_scale,
            project_slk=self.cfg.pos_phi_project_slk,
            pos_phi_free=getattr(self, "pos_phi_free", None),
        )
```

(d) In `forward`, immediately after `beliefs = self.prior_bank.encode(token_ids)` (line 222), compose the positional element into the frame:

```python
        beliefs = beliefs._replace(phi=self._apply_pos_phi(beliefs.phi))
```

(e) In `diagnostics` and `attention_maps`, find where the belief frame is first obtained from the encode (`belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=enc.phi[0])` in `attention_maps`, and the analogous encode in `diagnostics`) and apply the same map to its `phi` so the diagnostics/plots see the positionally-composed frame. For `attention_maps` (around line 602):

```python
        belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=self._apply_pos_phi(enc.phi[0]))
```

Apply the identical `_apply_pos_phi(...)` to the frame the `diagnostics` method builds from its encode (locate its `BeliefState(...)`/`.phi` construction and wrap the `phi`).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_positional_phi.py`
Expected: PASS (all BCH-PE tests).

- [ ] **Step 5: Commit**

```bash
git add vfe3/model/model.py tests/test_positional_phi.py
git commit -m "feat(model): apply BCH-PE pos_phi to the frame in forward/diagnostics/attention_maps"
```

### Task 1.4: Group `pos_phi_free` in the optimizer

**Files:**
- Modify: `vfe3/train.py` (`build_optimizer`)
- Test: `tests/test_train.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py  (append; VFE3Config / VFEModel / build_optimizer already imported there)
def test_build_optimizer_groups_pos_phi_free():
    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     pos_phi="learned", m_phi_lr=0.009)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)                      # must NOT raise the coverage AssertionError
    grouped = {p for g in opt.param_groups for p in g["params"]}
    assert model.pos_phi_free in grouped
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_train.py::test_build_optimizer_groups_pos_phi_free`
Expected: FAIL with the `build_optimizer left 1 model parameter(s) ungrouped` AssertionError (the coverage guard catches the un-grouped `pos_phi_free`).

- [ ] **Step 3: Add the param group**

In `vfe3/train.py::build_optimizer`, after the head-mixer group block (around line 64), add:

```python
    if getattr(model, "pos_phi_free", None) is not None:        # pos_phi='learned' positional table
        groups.append({"params": [model.pos_phi_free], "lr": cfg.m_phi_lr})  # a gauge-frame scale
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_train.py::test_build_optimizer_groups_pos_phi_free`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vfe3/train.py tests/test_train.py
git commit -m "feat(train): group learned pos_phi_free at m_phi_lr in build_optimizer"
```

### Task 1.5: BCH-PE property tests (structure preservation + isolation)

**Files:**
- Test: `tests/test_positional_phi.py` (append)

- [ ] **Step 1: Write the tests**

```python
# tests/test_positional_phi.py  (append)
from vfe3.geometry.lie_ops import embed_phi


def test_bch_differs_from_euclidean_when_bracket_nonzero():
    g = _glk_group(4)                                       # gl(4): non-abelian -> [X,Y] != 0 generically
    torch.manual_seed(0)
    phi = torch.randn(3, g.generators.shape[0])
    coords = positional_phi_coords("frozen", 3, g.generators.shape[0], scale=0.5,
                                   device=torch.device("cpu"), dtype=torch.float32)
    from vfe3.geometry.lie_ops import compose_phi
    bch = compose_phi(phi, coords, g.generators, order=4, mode="bch")
    euc = compose_phi(phi, coords, g.generators, order=4, mode="euclidean")
    assert not torch.allclose(bch, euc, atol=1e-4)         # they agree only when [phi, pos]=0


def test_project_slk_makes_blocks_traceless():
    g = get_group("block_glk")(4, 2)                       # gl(2)^2 blocks
    coords = positional_phi_coords("frozen", 5, g.generators.shape[0], scale=0.3,
                                   device=torch.device("cpu"), dtype=torch.float32)
    out = apply_positional_phi(torch.zeros(5, g.generators.shape[0]), g,
                               mode="frozen", scale=0.3, project_slk=True)
    M = embed_phi(out, g.generators)                       # (5, 4, 4) composed algebra element
    # det(Omega_h)=1  <=>  block-trace of the algebra element = 0
    assert torch.allclose(M[:, 0:2, 0:2].diagonal(dim1=-2, dim2=-1).sum(-1), torch.zeros(5), atol=1e-5)
    assert torch.allclose(M[:, 2:4, 2:4].diagonal(dim1=-2, dim2=-1).sum(-1), torch.zeros(5), atol=1e-5)


def test_pos_phi_does_not_change_self_coupling_diagnostic():
    # BCH-PE modifies belief.phi only; the self-coupling KL(q_i||p_i) reads the prior p_i (encode
    # mu/sigma), which pos_phi never touches -> the diagnostic's self_coupling term is unchanged.
    torch.manual_seed(0)
    x = torch.randint(0, 6, (1, 8))
    base = VFEModel(_cfg(pos_phi="none"))
    learned = VFEModel(_cfg(pos_phi="learned", pos_phi_scale=0.3))
    learned.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        learned.pos_phi_free.add_(0.2)
    # n_e_steps small + same priors: the prior p_i is identical, so the self_coupling read is too.
    assert abs(base.diagnostics(x)["self_coupling"] - learned.diagnostics(x)["self_coupling"]) < 1e-6
```

- [ ] **Step 2: Run to verify they pass**

Run: `python -m pytest tests/test_positional_phi.py`
Expected: PASS (all). If `test_bch_differs...` is flaky on the seed, the brackets are non-zero for `gl(4)` with these inputs; do not weaken the assertion. NOTE on `Omega_ii = I`: this holds by construction for ANY frame (`exp(phi_i)exp(-phi_i)=I`), so it is not the place BCH-PE acts; no separate test is needed beyond the existing transport tests. If the `self_coupling` test reveals the diagnostic also folds `phi` into the prior on your build, treat that as a real finding and report it rather than weakening the assertion.

- [ ] **Step 3: Run the full suite (BCH-PE complete)**

Run: `python -m pytest`
Expected: all pass, 0 failures, 0 errors. Read the `N passed` line.

- [ ] **Step 4: Commit**

```bash
git add tests/test_positional_phi.py
git commit -m "test(positional): BCH vs euclidean divergence + sl(K) trace-projection property tests"
```

## Part 2: gauge-RoPE

### Task 2.1: The rotation builder + `pos_rotation` registry

**Files:**
- Create: `vfe3/geometry/rope.py`
- Test: `tests/test_rope.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rope.py
import torch

from vfe3.geometry.rope import build_rope_rotation, get_pos_rotation


def test_rope_rotation_is_orthogonal_and_block_diagonal():
    irrep_dims = [4, 4]                                     # two head-blocks of size 4
    R = build_rope_rotation(torch.arange(6), irrep_dims, base=100.0,
                            device=torch.device("cpu"), dtype=torch.float32)
    assert R.shape == (6, 8, 8)
    eye = torch.eye(8).expand(6, 8, 8)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)   # orthogonal
    # off-block entries are exactly zero (block-diagonal on irrep_dims)
    assert torch.count_nonzero(R[:, 0:4, 4:8]) == 0
    assert torch.count_nonzero(R[:, 4:8, 0:4]) == 0


def test_rope_position_zero_is_identity():
    R = build_rope_rotation(torch.arange(3), [4], base=100.0,
                            device=torch.device("cpu"), dtype=torch.float32)
    assert torch.allclose(R[0], torch.eye(4), atol=1e-6)   # position 0 -> angle 0 -> I


def test_pos_rotation_none_registered():
    assert get_pos_rotation("none")(torch.arange(3), [4], base=100.0,
                                    device=torch.device("cpu"), dtype=torch.float32) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_rope.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'vfe3.geometry.rope'`.

- [ ] **Step 3: Write the module**

```python
# vfe3/geometry/rope.py
r"""Gauge-RoPE: a block-diagonal positional rotation R(theta) for VFE_3.0 transport.

Realizes the manuscript's identification of the per-token frame U_i with a rotary positional
rotation (GL(K)_attention.tex, "Identification with rotary positional structure"): within each
irrep block of size d, coordinate pairs (2k, 2k+1) rotate by theta_{n,k} = n * base^{-2k/d}, so
the combined frame is U_i = R(theta_i) exp(phi_i) and Omega_ij = R(theta_i) Omega_ij^learned
R(theta_j)^T. Block-diagonal on irrep_dims so R is orthogonal and preserves the block-diagonal
fast path. Parameter-free; default-off via the ``pos_rotation`` registry ("none").
"""

from typing import Callable, Dict, List, Optional

import torch

_POS_ROTATIONS: Dict[str, Callable[..., Optional[torch.Tensor]]] = {}


def register_pos_rotation(name: str) -> Callable:
    """Decorator registering a positional-rotation builder -> (N, K, K) rotation or None."""
    def _wrap(fn: Callable[..., Optional[torch.Tensor]]) -> Callable[..., Optional[torch.Tensor]]:
        _POS_ROTATIONS[name] = fn
        return fn
    return _wrap


def get_pos_rotation(name: str) -> Callable[..., Optional[torch.Tensor]]:
    """Return the registered positional-rotation builder (KeyError if absent)."""
    if name not in _POS_ROTATIONS:
        raise KeyError(f"no pos_rotation {name!r}; available: {sorted(_POS_ROTATIONS)}")
    return _POS_ROTATIONS[name]


@register_pos_rotation("none")
def _pos_rotation_none(
    positions:  torch.Tensor,
    irrep_dims: List[int],

    *,
    base:   float = 100.0,
    device: torch.device = None,
    dtype:  torch.dtype  = torch.float32,
) -> Optional[torch.Tensor]:
    r"""No rotation: returns None (the transport is left un-rotated)."""
    return None


@register_pos_rotation("rope")
def build_rope_rotation(
    positions:  torch.Tensor,             # (N,) integer token positions
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    base:   float = 100.0,
    device: torch.device = None,
    dtype:  torch.dtype  = torch.float32,
) -> torch.Tensor:                        # (N, K, K) block-diagonal orthogonal rotation
    r"""Per-position block-diagonal rotation R(theta) on ``irrep_dims``.

    Within a block of size d at offset s, pairs (s+2k, s+2k+1) rotate by
    theta_{n,k} = n * base^{-2k/d}; an odd leftover coordinate stays identity. The result is
    orthogonal and block-diagonal, so it preserves the block-diagonal exp fast path.
    """
    pos = positions.to(device=device, dtype=dtype)                 # (N,)
    K = int(sum(irrep_dims))
    N = pos.shape[0]
    R = torch.eye(K, device=device, dtype=dtype).expand(N, K, K).clone()
    start = 0
    for d in irrep_dims:
        n_pairs = d // 2
        for k in range(n_pairs):
            freq = base ** (-2.0 * k / d)
            theta = pos * freq                                     # (N,)
            c, s = torch.cos(theta), torch.sin(theta)
            a, b = start + 2 * k, start + 2 * k + 1
            R[:, a, a] = c;  R[:, a, b] = -s
            R[:, b, a] = s;  R[:, b, b] = c
        start += d
    return R
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_rope.py`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add vfe3/geometry/rope.py tests/test_rope.py
git commit -m "feat(rope): block-diagonal R(theta) builder + pos_rotation registry (none/rope)"
```

### Task 2.2: `RopeTransport` container + transport_mean/transport_covariance dispatch

**Files:**
- Modify: `vfe3/geometry/transport.py` (add `RopeTransport`; handle it in `transport_mean`/`transport_covariance`)
- Test: `tests/test_rope.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rope.py  (append)
from vfe3.geometry.transport import RopeTransport, transport_mean, transport_covariance


def test_rope_mean_at_identity_omega_is_relative():
    # Omega = I -> Omega^RoPE = R(theta_i) R(theta_j)^T = R(theta_i - theta_j): relative position.
    N, K = 5, 4
    R = build_rope_rotation(torch.arange(N), [K], base=100.0,
                            device=torch.device("cpu"), dtype=torch.float32)
    omega_I = torch.eye(K).expand(N, N, K, K).contiguous()
    mu = torch.randn(N, K)
    rt = RopeTransport(base=omega_I, rope=R, on_cov=False)
    mu_t = transport_mean(rt, mu)                            # (N, N, K)
    # mu_t[i, j] should equal R(theta_i - theta_j) mu_j; check it depends only on (i-j) for a
    # constant key mean (Toeplitz structure in the transported norm is automatic for orthogonal R).
    mu_const = torch.ones(N, K)
    rt_c = RopeTransport(base=omega_I, rope=R, on_cov=False)
    t = transport_mean(rt_c, mu_const)                      # (N, N, K)
    # rows of equal (i-j) give equal transported vectors
    assert torch.allclose(t[2, 1], t[3, 2], atol=1e-5)      # both are (i-j)=1
    assert torch.allclose(t[3, 1], t[4, 2], atol=1e-5)      # both are (i-j)=2


def test_rope_mean_only_leaves_covariance_unrotated():
    N, K = 4, 4
    R = build_rope_rotation(torch.arange(N), [K], base=10.0,
                            device=torch.device("cpu"), dtype=torch.float32)
    omega_I = torch.eye(K).expand(N, N, K, K).contiguous()
    sigma = torch.rand(N, K) + 0.5
    rt = RopeTransport(base=omega_I, rope=R, on_cov=False)
    plain = transport_covariance(omega_I, sigma)            # un-rotated diagonal sandwich
    roped = transport_covariance(rt, sigma)                 # mu-only -> ignores rope
    assert torch.allclose(plain, roped, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_rope.py -k "rope_mean"`
Expected: FAIL with `ImportError: cannot import name 'RopeTransport'`.

- [ ] **Step 3: Implement the container and dispatch**

In `vfe3/geometry/transport.py`, add the dataclass after `FactoredTransport` (around line 49):

```python
@dataclass
class RopeTransport:
    r"""A built transport wrapped with a gauge-RoPE positional rotation R(theta).

    ``base`` is the un-rotated transport (a dense (N,N,K,K) Omega OR a FactoredTransport). The
    effective operator is Omega^RoPE_ij = R(theta_i) Omega_ij R(theta_j)^T. ``transport_mean``
    always applies the rotation; ``transport_covariance`` applies it only when ``on_cov`` (the
    means+covariance "full-gauge" regime, which the config gates to full covariance). Means-only
    (``on_cov=False``) leaves the covariance sandwich on the un-rotated ``base`` -- numerically
    identical to no RoPE for the covariance, so the diagonal-covariance path stays valid.
    """

    base:   'torch.Tensor | FactoredTransport'
    rope:   torch.Tensor                  # (N, K, K) block-diagonal orthogonal rotation
    on_cov: bool = False


def _rope_dense_omega(base, rope: torch.Tensor) -> torch.Tensor:
    r"""Effective dense Omega^RoPE_ij = R(theta_i) Omega_ij R(theta_j)^T (full-gauge / dense path)."""
    omega = base.to_dense_omega() if isinstance(base, FactoredTransport) else base   # (...,N,N,K,K)
    # R_i Omega_ij R_j^T: contract R on the left of the i-axis output and the right (transposed) of j.
    rot = torch.einsum("...ikl,...ijlm,...jnm->...ijkn", rope, omega, rope)
    return rot
```

Then, at the TOP of `transport_mean`, before the `FactoredTransport` branch, add:

```python
    if isinstance(omega, RopeTransport):
        # mu_t[i,j] = R_i Omega_ij R_j^T mu_j: pre-rotate the key mean by R_j^T, transport on the
        # un-rotated base, post-rotate the result by R_i. R_j^T mu_j = sum_l R[j,l,k] mu[j,l].
        m = torch.einsum("...jlk,...jl->...jk", omega.rope, mu)        # (..., N, K)
        t = transport_mean(omega.base, m)                             # (..., N, N, K)
        return torch.einsum("...ikl,...ijl->...ijk", omega.rope, t)   # post-rotate by R_i
```

And at the TOP of `transport_covariance`, before the `FactoredTransport` branch, add:

```python
    if isinstance(omega, RopeTransport):
        if not omega.on_cov:
            return transport_covariance(omega.base, sigma, diagonal_out=diagonal_out)   # mu-only
        # full-gauge: sandwich with the rotated dense operator (requires full covariance).
        return transport_covariance(_rope_dense_omega(omega.base, omega.rope), sigma,
                                    diagonal_out=diagonal_out)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_rope.py`
Expected: PASS (all rope tests so far).

- [ ] **Step 5: Commit**

```bash
git add vfe3/geometry/transport.py tests/test_rope.py
git commit -m "feat(transport): RopeTransport container + R(theta) dispatch in transport_mean/covariance"
```

### Task 2.3: Build-site wrapping + thread `rope` to the transport builders

**Files:**
- Modify: `vfe3/inference/e_step.py` (`build_belief_transport`, `e_step_iteration`, `e_step` add `rope`/`rope_on_cov`, wrap in `RopeTransport`). NOTE: `_transport` itself is NOT modified — diagnostics/attention_maps wrap its output at the model call site (Task 2.5), and `_transport_qk` is the flat-only trajectory diagnostic (intentionally rope-unaware, like `free_energy_value`).
- Test: `tests/test_rope.py` (append a build-site test)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rope.py  (append)
from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import build_belief_transport
from vfe3.geometry.transport import RopeTransport


def test_build_belief_transport_wraps_in_ropetransport_when_rope_set():
    g = get_group("block_glk")(8, 2)
    phi = torch.randn(1, 6, g.generators.shape[0])
    R = build_rope_rotation(torch.arange(6), g.irrep_dims, base=100.0,
                            device=phi.device, dtype=phi.dtype)
    out = build_belief_transport(phi, g, transport_mode="flat", rope=R, rope_on_cov=False)
    assert isinstance(out, RopeTransport)
    # rope=None reproduces the plain build (no wrapper).
    plain = build_belief_transport(phi, g, transport_mode="flat")
    assert not isinstance(plain, RopeTransport)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_rope.py::test_build_belief_transport_wraps_in_ropetransport_when_rope_set`
Expected: FAIL with `TypeError: build_belief_transport() got an unexpected keyword argument 'rope'`.

- [ ] **Step 3: Thread `rope` and wrap**

In `vfe3/inference/e_step.py`:

(a) Import `RopeTransport` (extend the existing transport import block at the top, lines ~24-28):

```python
    RopeTransport,
```

(b) `build_belief_transport` (line 83): add two keyword params at the end of its signature (after `cocycle_relaxation`):

```python
    rope:               Optional[torch.Tensor] = None,      # (N, K, K) gauge-RoPE rotation (None -> off)
    rope_on_cov:        bool                   = False,     # rotate the covariance too (full-gauge)
```

At its `return` points, wrap. The function currently returns `build_factored_transport(...)` (fused path, line 106) or `_transport(...)` (dense path). Change the body tail to capture the built transport then wrap:

```python
    if _can_fuse_flat(transport_mode, group):
        built = build_factored_transport(phi, group)
    else:
        transport_kw = (
            dict(mu=mu, connection_W=connection_W, cocycle_relaxation=cocycle_relaxation)
            if transport_mode == "regime_ii" else {}
        )
        built = _transport(phi, group, transport_mode=transport_mode, **transport_kw)
    if rope is None:
        return built
    return RopeTransport(base=built, rope=rope, on_cov=rope_on_cov)
```

(Verify against the existing lines 105-111 you read; preserve whatever the current non-fused branch passes.)

(c) `e_step_iteration` (line 237): add the same two keyword params (after `cocycle_relaxation`, before `log_prior`):

```python
    rope:                      Optional[torch.Tensor] = None,   # (N, K, K) gauge-RoPE rotation
    rope_on_cov:               bool                   = False,  # full-gauge: rotate covariance too
```

Forward them in the `build_belief_transport(...)` call (line 291):

```python
    omega = build_belief_transport(
        belief.phi, group, transport_mode=transport_mode,
        mu=belief.mu, connection_W=connection_W, cocycle_relaxation=cocycle_relaxation,
        rope=rope, rope_on_cov=rope_on_cov,
    )
```

(d) `e_step` (line 348): add `rope`/`rope_on_cov` as EXPLICIT keywords (NOT via `**kwargs`, so they do not reach the kwargs-rejecting `free_energy_value` diagnostic), after `e_step_gradient`:

```python
    rope:              Optional[torch.Tensor] = None,
    rope_on_cov:       bool                   = False,
```

and forward them explicitly in the `e_step_iteration(...)` call (line 387):

```python
        belief = e_step_iteration(
            belief, mu_p, sigma_p, group, tau=tau,
            e_mu_lr=e_mu_lr, e_sigma_lr=e_sigma_lr, e_phi_lr=e_phi_lr,
            e_step_gradient=e_step_gradient, log_prior=log_prior,
            rope=rope, rope_on_cov=rope_on_cov, **kwargs,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_rope.py::test_build_belief_transport_wraps_in_ropetransport_when_rope_set`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vfe3/inference/e_step.py tests/test_rope.py
git commit -m "feat(e_step): thread rope/rope_on_cov to build_belief_transport; wrap in RopeTransport"
```

### Task 2.4: Thread `rope` through the block/stack and model forward + config

**Files:**
- Modify: `vfe3/config.py` (add `pos_rotation`/`rope_base`/`rope_full_gauge` + validation)
- Modify: `vfe3/model/stack.py`, `vfe3/model/block.py` (thread `rope`/`rope_on_cov`)
- Modify: `vfe3/model/model.py` (precompute + cache `R`; pass to `vfe_stack`)
- Test: `tests/test_config.py`, `tests/test_rope.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py  (append)
def test_rope_defaults_off_and_full_gauge_requires_full_cov():
    cfg = VFE3Config()
    assert cfg.pos_rotation == "none" and cfg.rope_full_gauge is False
    with pytest.raises(ValueError):
        VFE3Config(pos_rotation="rope", rope_full_gauge=True, diagonal_covariance=True)
    # full-gauge with full covariance is allowed
    VFE3Config(pos_rotation="rope", rope_full_gauge=True, diagonal_covariance=False)
```

```python
# tests/test_rope.py  (append)
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _rope_cfg(**kw):
    base = dict(vocab_size=6, embed_dim=8, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0, gauge_group="block_glk",
                warmup_steps=1, max_steps=4)
    base.update(kw)
    return VFE3Config(**base)


def test_rope_changes_logits_vs_no_rope():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    base = VFEModel(_rope_cfg(pos_rotation="none"))
    roped = VFEModel(_rope_cfg(pos_rotation="rope"))
    roped.load_state_dict(base.state_dict())
    assert not torch.allclose(base(x), roped(x), atol=1e-5)   # RoPE perturbs attention -> logits
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_config.py::test_rope_defaults_off_and_full_gauge_requires_full_cov tests/test_rope.py::test_rope_changes_logits_vs_no_rope`
Expected: FAIL (`pos_rotation` unknown kwarg; logits identical).

- [ ] **Step 3a: Config fields + validation**

In `vfe3/config.py`, add fields in the gauge seam (near the BCH-PE fields from Task 1.2):

```python
    # gauge-RoPE (default-off): a block-diagonal positional rotation R(theta) folded into the
    # transport (Omega^RoPE_ij = R(theta_i) Omega_ij R(theta_j)^T). Means-only by default;
    # rope_full_gauge=True also rotates the covariance sandwich and REQUIRES full covariance.
    pos_rotation:              str   = "none"      # "none" | "rope" (the positional-rotation registry)
    rope_base:                 float = 100.0       # rotary frequency base
    rope_full_gauge:           bool  = False       # rotate covariance too (needs diagonal_covariance=False)
```

Add validation in `__post_init__` (next to the `pos_phi` validation from Task 1.2):

```python
        from vfe3.geometry.rope import _POS_ROTATIONS
        _require(self.pos_rotation, tuple(sorted(_POS_ROTATIONS)), "pos_rotation")
        if self.rope_full_gauge and self.diagonal_covariance:
            raise ValueError(
                "rope_full_gauge=True rotates the covariance sandwich (R Sigma R^T), which the "
                "diagonal-covariance approximation cannot carry; set diagonal_covariance=False."
            )
```

- [ ] **Step 3b: Thread `rope` through stack/block**

In `vfe3/model/stack.py::vfe_stack`, add two keyword params (after `e_step_gradient`):

```python
    rope:            Optional[torch.Tensor]    = None,   # (N, K, K) gauge-RoPE rotation (None -> off)
    rope_on_cov:     bool                      = False,  # full-gauge: rotate covariance too
```

and forward them in the `vfe_block(...)` call (line 45):

```python
        belief = vfe_block(belief, mu_p, sigma_p, group, cfg, log_prior=log_prior,
                           block_norm=block_norm, log_alpha=log_alpha, connection_W=connection_W,
                           e_step_gradient=e_step_gradient, rope=rope, rope_on_cov=rope_on_cov)
```

In `vfe3/model/block.py::vfe_block`, add the same two keyword params (after `e_step_gradient`) and forward them in the `e_step(...)` call (line 39):

```python
        e_step_gradient=e_step_gradient,
        rope=rope, rope_on_cov=rope_on_cov,
        log_prior=log_prior,
```

- [ ] **Step 3c: Precompute + cache `R` in the model and pass it to `vfe_stack`**

In `vfe3/model/model.py`:

(a) Add the import:

```python
from vfe3.geometry.rope import get_pos_rotation
```

(b) Add a cache dict in `__init__` next to `self._log_prior_cache = {}`:

```python
        self._rope_cache: dict = {}
```

and clear it in `_apply` next to `self._log_prior_cache.clear()`:

```python
        self._rope_cache.clear()
```

(c) Add a cached builder method (mirrors `_attention_log_prior`):

```python
    def _rope_rotation(self, n: int, device: torch.device) -> Optional[torch.Tensor]:
        r"""Cached gauge-RoPE rotation R(theta) for length n (None when pos_rotation='none')."""
        if self.cfg.pos_rotation == "none":
            return None
        dtype = self.prior_bank.mu_embed.dtype
        key = (n, device, dtype)
        cached = self._rope_cache.get(key)
        if cached is None:
            cached = get_pos_rotation(self.cfg.pos_rotation)(
                torch.arange(n, device=device), self.group.irrep_dims,
                base=self.cfg.rope_base, device=device, dtype=dtype)
            self._rope_cache[key] = cached
        return cached
```

(d) In `forward`, build it next to `log_prior` (after line 223) and pass to `vfe_stack` (line 257):

```python
        rope = self._rope_rotation(N, token_ids.device)
```

```python
            out = vfe_stack(beliefs, beliefs.mu, beliefs.sigma, self.group, self.cfg,
                            log_prior=log_prior, block_norm=self.block_norm, log_alpha=log_alpha,
                            connection_W=connection_W, e_step_gradient=e_step_gradient,
                            rope=rope, rope_on_cov=self.cfg.rope_full_gauge)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_config.py -k rope tests/test_rope.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vfe3/config.py vfe3/model/stack.py vfe3/model/block.py vfe3/model/model.py tests/test_config.py tests/test_rope.py
git commit -m "feat(rope): config + thread R(theta) through stack/block/forward (means-only default)"
```

### Task 2.5: RoPE in diagnostics + attention_maps

**Files:**
- Modify: `vfe3/model/model.py` (`diagnostics`, `attention_maps` build transport via `_transport`; pass `rope`)
- Test: `tests/test_rope.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rope.py  (append)
def test_attention_maps_reflect_rope():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (1, 8))
    base = VFEModel(_rope_cfg(pos_rotation="none"))
    roped = VFEModel(_rope_cfg(pos_rotation="rope"))
    roped.load_state_dict(base.state_dict())
    a = base.attention_maps(x)
    b = roped.attention_maps(x)
    assert a.shape == b.shape
    assert not torch.allclose(a, b, atol=1e-5)             # RoPE changes the per-head attention
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_rope.py::test_attention_maps_reflect_rope`
Expected: FAIL (maps identical — `attention_maps`/`diagnostics` still build the un-rotated transport).

- [ ] **Step 3: Pass `rope` into the diagnostics/attention_maps transport build**

In `vfe3/model/model.py::attention_maps`, the loop builds `omega = _transport(belief.phi, self.group, transport_mode=..., ...)` (around line 620). Compute the rotation once before the loop and wrap the built transport. Because `attention_maps` calls `_transport` directly (returning a dense Omega) and then `transport_mean`/`transport_covariance`, wrap that Omega in `RopeTransport`:

```python
        from vfe3.geometry.transport import RopeTransport
        rope = self._rope_rotation(n, token_ids.device)
```

and where it currently does `omega = _transport(...)`, follow with:

```python
            if rope is not None:
                omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge)
```

Apply the identical pattern in `diagnostics` at its `_transport(...)` build site (locate the `omega = _transport(...)` line in `diagnostics` and wrap it the same way, using `n = belief.mu.shape[0]` for the rope length).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_rope.py::test_attention_maps_reflect_rope`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vfe3/model/model.py tests/test_rope.py
git commit -m "feat(rope): wrap diagnostics/attention_maps transport in RopeTransport"
```

### Task 2.6: Gradient correctness (kernel-vs-oracle, covariance sandwich) + full suite

**Files:**
- Test: `tests/test_rope.py` (append)

**Why not whole-model finite differences:** FD on the model loss through the unrolled E-step is the WRONG gate — the E-step has kinks (trust-region clamps, `sigma_max`, the Frobenius clamp in `matrix_exp`, `kl_max` NaN-cap) where FD != autograd for reasons unrelated to RoPE, so such a test fails spuriously. The codebase's actual discipline is "analytic kernel vs autograd-of-F **oracle**". Both `belief_gradients` (hand kernel) and `belief_gradients_autograd` (oracle) consume the built `omega` identically, so feeding the SAME `RopeTransport` to both isolates RoPE exactly.

- [ ] **Step 1: Write the gradient-correctness tests**

```python
# tests/test_rope.py  (append)
from vfe3.gradients.kernels import belief_gradients
from vfe3.gradients.oracle import belief_gradients_autograd
from vfe3.inference.e_step import build_belief_transport
from vfe3.geometry.transport import transport_covariance


def test_rope_means_only_kernel_matches_oracle():
    # The analytic belief-gradient kernel must still agree with autograd-of-F when the transport is
    # rope-rotated (means-only). Both consume the RopeTransport opaquely; agreement isolates RoPE.
    torch.manual_seed(0)
    g = get_group("block_glk")(8, 2)
    N, K, n_gen = 5, 8, g.generators.shape[0]
    phi = torch.randn(1, N, n_gen) * 0.1
    R = build_rope_rotation(torch.arange(N), g.irrep_dims, base=100.0,
                            device=phi.device, dtype=phi.dtype)
    omega = build_belief_transport(phi, g, transport_mode="flat", rope=R, rope_on_cov=False)
    mu   = torch.randn(1, N, K); sigma   = torch.rand(1, N, K) + 0.5
    mu_p = torch.randn(1, N, K); sigma_p = torch.rand(1, N, K) + 0.5
    kw = dict(tau=1.0, alpha_div=1.0, kl_max=100.0, eps=1e-6, b0=1.0, c0=1.0, value=1.0,
              include_attention_entropy=True, gradient_mode="filtering", family="gaussian_diagonal",
              divergence_family="renyi", alpha_mode="constant", irrep_dims=g.irrep_dims, log_prior=None)
    gk = belief_gradients(mu, sigma, mu_p, sigma_p, omega, **kw)          # hand kernel
    go = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega, **kw)  # autograd-of-F oracle
    assert torch.allclose(gk[0], go[0], atol=1e-5)
    assert torch.allclose(gk[1], go[1], atol=1e-5)


def test_rope_full_gauge_covariance_equals_manual_sandwich():
    # Full-gauge covariance: transport_covariance(RopeTransport(on_cov=True)) must equal the manual
    # sandwich with the rotated operator Omega'_ij = R_i Omega_ij R_j^T. Pure property; no model.
    from vfe3.geometry.transport import RopeTransport
    torch.manual_seed(0)
    N, K = 4, 4
    R = build_rope_rotation(torch.arange(N), [K], base=10.0,
                            device=torch.device("cpu"), dtype=torch.float64)
    omega = torch.randn(N, N, K, K, dtype=torch.float64)
    A = torch.randn(N, K, K, dtype=torch.float64)
    sigma = A @ A.transpose(-1, -2) + K * torch.eye(K, dtype=torch.float64)   # SPD full cov
    got = transport_covariance(RopeTransport(base=omega, rope=R, on_cov=True), sigma)
    Op = torch.einsum("ikl,ijlm,jnm->ijkn", R, omega, R)                      # R_i Omega_ij R_j^T
    manual = torch.einsum("ijkl,jlm,ijnm->ijkn", Op, sigma, Op)
    assert torch.allclose(got, manual, atol=1e-9)


def test_full_gauge_model_runs_forward_backward():
    # Reachability: a full-covariance rope_full_gauge model trains (finite gradients) end to end.
    # Executor: set the full-covariance family/decode here -- see the procedure note below.
    cfg = _full_cov_cfg(pos_rotation="rope", rope_full_gauge=True)   # defined per the note
    torch.manual_seed(0)
    m = VFEModel(cfg)
    x = torch.randint(0, 6, (1, 6)); y = torch.randint(0, 6, (1, 6))
    _, loss, _ = m(x, y); loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(m.prior_bank.mu_embed.grad).all()
```

Concrete procedure for `_full_cov_cfg` (the full-gauge path needs a working full-covariance model): grep the existing full-cov tests for the working combination — `rg "diagonal_covariance=False" tests/` — and copy their `family` / `decode_mode` / `spd_retract_mode`. Define `_full_cov_cfg(**kw)` = `_rope_cfg` with `diagonal_covariance=False` plus those values. BEFORE wiring RoPE in, confirm `VFEModel(_full_cov_cfg(pos_rotation="none"))` runs forward+backward; if it does not, that is a pre-existing full-cov issue to report, not a RoPE bug.

- [ ] **Step 2: Run to verify they pass**

Run: `python -m pytest tests/test_rope.py -k "kernel_matches_oracle or covariance_equals_manual or full_gauge_model"`
Expected: PASS. The kernel-vs-oracle test is the load-bearing RoPE-means correctness gate; the covariance-sandwich test is the full-gauge correctness gate.

- [ ] **Step 3: Run the full suite (RoPE complete)**

Run: `python -m pytest`
Expected: all pass, 0 failures, 0 errors. Read the `N passed` line (or `--junitxml=out.xml` and read `testsuite tests=/failures=/errors=`).

- [ ] **Step 4: Commit**

```bash
git add tests/test_rope.py
git commit -m "test(rope): FD gradient checks for means-only and full-gauge transport rotation"
```

### Task 2.7: Post-edit doc + 2x2 ablation smoke

**Files:**
- Create/Modify: `docs/edits/2026-06-02-positional-encodings.md` (post-edit policy)
- Test: `tests/test_rope.py` (append the 2x2 toggle smoke)

- [ ] **Step 1: 2x2 ablation smoke test**

```python
# tests/test_rope.py  (append)
def test_2x2_positional_ablation_runs():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    for pr in ("none", "rope"):
        for pp in ("none", "learned"):
            m = VFEModel(_rope_cfg(pos_rotation=pr, pos_phi=pp))
            out = m(x)
            assert out.shape[0] == 2 and torch.isfinite(out).all()
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_rope.py::test_2x2_positional_ablation_runs`
Expected: PASS.

- [ ] **Step 3: Write the post-edit doc**

Create `docs/edits/2026-06-02-positional-encodings.md` summarizing both seams: the two registries (`pos_phi`, `pos_rotation`), the `RopeTransport` mechanism and why the gradient kernel/oracle needed no change, the means-only vs full-gauge split and its config gate, the no-NN status of `pos_phi_free`, and the property-test coverage. Follow the project's prose style (no `---` rules, no banned words).

- [ ] **Step 4: Final full suite + commit**

Run: `python -m pytest`
Expected: all pass, 0 failures, 0 errors.

```bash
git add tests/test_rope.py docs/edits/2026-06-02-positional-encodings.md
git commit -m "test(positional): 2x2 RoPE/BCH-PE ablation smoke + post-edit doc"
```

## Self-review notes (for the executor)

- The means-only RoPE path (default) keeps `transport_covariance` on the un-rotated base, so it is valid under the user's `diagonal_covariance=True`. Do not let RoPE touch the covariance unless `rope_full_gauge=True` (which the config forces to full covariance).
- `belief_gradients` / `belief_gradients_autograd` signatures must NOT change — they consume the `RopeTransport` opaquely through `transport_mean`/`transport_covariance`. If you find yourself adding `rope` to those, stop: the container already carries it.
- `rope` must be an EXPLICIT keyword in `e_step` (not via `**kwargs`), or the diagnostic `free_energy_value` will reject it.
- `_transport_qk` (the untied query/key path) and the `e_step` trajectory diagnostic (`free_energy_value`) are flat-only by existing design and are intentionally left rope-unaware. They are NOT on the training forward path (which builds transport via `build_belief_transport`). The spec lists `_transport_qk` in the integration surface for completeness, but threading RoPE there is out of scope for this plan; do not add it unless a later need arises.
- After Part 1, `build_optimizer`'s exact-coverage guard will RAISE if `pos_phi_free` is not grouped — Task 1.4 is mandatory, not optional.
- Both features default off; with `pos_rotation="none"` and `pos_phi="none"` the forward must be byte-identical to the pre-feature build (the `test_pos_phi_none_logits...` determinism guard plus the existing suite are the gate).
```
