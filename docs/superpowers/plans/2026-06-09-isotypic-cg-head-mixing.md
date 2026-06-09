# Isotypic + Clebsch-Gordan Head Mixing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mixing for non-equal gauge-irrep heads: a per-type Schur mixer (the full linear commutant of a mixed irrep tower) plus an exactly-equivariant Clebsch-Gordan bilinear coupling across inequivalent types.

**Architecture:** `GaugeGroup` gains optional per-block `irrep_labels`; `HeadMixer` generalizes to label-grouped isotypic components (legacy groups byte-identical); a new numerical CG solver (`vfe3/geometry/cg.py`) feeds a new `CGCoupling` module applied at the existing between-block seam (`vfe_block`, after the mixer, before the norm). Both parameter sets are zero-initialized (step 0 byte-identical) and join the documented NN-exception family. Spec: `docs/superpowers/specs/2026-06-09-isotypic-cg-head-mixing-design.md`.

**Tech Stack:** PyTorch (float32 runtime, float64 construction), pytest. No new dependencies.

Conventions that bind every task: pytest pass counts come from the summary line (never add `-q`; pyproject already sets it). Tensors-first signature ordering with aligned columns. Tests are device-agnostic CPU. Run commands from the repo root `C:\Users\chris and christine\Desktop\V3_Transformer`.

### Task 1: `GaugeGroup.irrep_labels`

**Files:**
- Modify: `vfe3/geometry/groups.py` (dataclass ~line 31-51; `_build_so_n` / `_build_sp_n` ~lines 247-340)
- Test: `tests/test_son_irreps.py` (append)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_son_irreps.py`:

```python
def test_groups_expose_irrep_labels():
    grp = get_group("so_n")(14, group_n=3,
                            irrep_spec=[("l0", 1), ("l1", 2), ("l3", 1)])
    assert grp.irrep_dims == [1, 3, 3, 7]
    assert grp.irrep_labels == ["l0", "l1", "l1", "l3"]
    grp2 = get_group("sp_n")(5, group_n=4, irrep_spec=[("sym0", 1), ("sym1", 1)])
    assert grp2.irrep_labels == ["sym0", "sym1"]
    # legacy groups carry no labels
    assert get_group("glk")(4).irrep_labels is None
    assert get_group("block_glk")(6, 3).irrep_labels is None


def test_irrep_labels_length_validated():
    from vfe3.geometry.groups import GaugeGroup
    with pytest.raises(ValueError, match="irrep_labels"):
        GaugeGroup(name="x", generators=torch.zeros(1, 4, 4), irrep_dims=[2, 2],
                   skew_symmetric=True, irrep_labels=["a"])
```

- [ ] **Step 2: Run, verify failure**

Run: `python -m pytest tests/test_son_irreps.py -k labels`
Expected: FAIL (`irrep_labels` is not a field / unexpected keyword).

- [ ] **Step 3: Implement** — in `vfe3/geometry/groups.py`:

(a) Add the field at the END of the `GaugeGroup` dataclass (after `invariant_families`, so every existing keyword construction stays valid):

```python
    irrep_labels:       Optional[List[str]] = None   # per-block irrep label ('l1', 'sym2', ...);
                                                     # None for label-less groups (glk/block_glk/...)
```

(b) In `__post_init__`, after the existing sum check:

```python
        if self.irrep_labels is not None and len(self.irrep_labels) != len(self.irrep_dims):
            raise ValueError(
                f"irrep_labels has {len(self.irrep_labels)} entries but there are "
                f"{len(self.irrep_dims)} irrep blocks"
            )
```

(c) In `_build_so_n` AND `_build_sp_n`, compute the per-block labels from the spec (one entry per copy, matching `direct_sum_generators`' block order) and pass them:

```python
    labels = [lab for lab, mult in irrep_spec for _ in range(int(mult))]
```

and add `irrep_labels=labels,` to both `GaugeGroup(...)` constructions.

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest tests/test_son_irreps.py`
Expected: all pass (existing 20 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add vfe3/geometry/groups.py tests/test_son_irreps.py
git commit -m "feat(groups): per-block irrep_labels on GaugeGroup (so_n/sp_n populate)"
```

### Task 2: isotypic `HeadMixer`

**Files:**
- Modify: `vfe3/model/head_mixer.py` (whole class)
- Create: `tests/test_head_mixer_isotypic.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_head_mixer_isotypic.py`:

```python
r"""Isotypic (label-grouped) HeadMixer: the full linear commutant of a mixed irrep tower."""

import pytest
import torch

from vfe3.geometry.groups import get_group
from vfe3.model.head_mixer import HeadMixer


def test_label_runs_become_components():
    m = HeadMixer([1, 3, 3, 7], irrep_labels=["l0", "l1", "l1", "l3"])
    assert [tuple(d.shape) for d in m.mixer_deltas] == [(1, 1), (2, 2), (1, 1)]
    assert m.is_identity()


def test_unlabeled_unequal_dims_still_raise():
    with pytest.raises(ValueError, match="equal-size blocks"):
        HeadMixer([1, 3, 7, 9])


def test_legacy_single_component_keeps_mixer_delta_attribute():
    m = HeadMixer([4, 4, 4])
    assert m.mixer_delta.shape == (3, 3)        # back-compat accessor (single component)


def test_identity_init_is_exact_passthrough():
    m = HeadMixer([1, 3, 3, 7], irrep_labels=["l0", "l1", "l1", "l3"])
    mu = torch.randn(2, 5, 14)
    sig = torch.rand(2, 5, 14) + 0.5
    mu2, sig2 = m(mu, sig)
    assert torch.equal(mu2, mu) and torch.equal(sig2, sig)


def test_isotypic_mixer_exactly_equivariant_under_tower_gauge_full_cov():
    # mix(g mu, g S g^T) == (g mix_mu, g mix_S g^T) for a trained (non-identity) mixer,
    # because blockdiag_t(A_t kron I_d) is the commutant of the tower (real-type irreps).
    torch.manual_seed(0)
    grp = get_group("so_n")(14, group_n=3,
                            irrep_spec=[("l0", 1), ("l1", 2), ("l3", 1)],
                            dtype=torch.float64)
    m = HeadMixer(grp.irrep_dims, irrep_labels=grp.irrep_labels).double()
    with torch.no_grad():
        for d in m.mixer_deltas:
            d.copy_(0.3 * torch.randn(*d.shape, dtype=torch.float64))
    g = torch.linalg.matrix_exp(
        torch.einsum("a,aij->ij", 0.4 * torch.randn(3, dtype=torch.float64), grp.generators))
    mu = torch.randn(5, 14, dtype=torch.float64)
    A = torch.randn(5, 14, 14, dtype=torch.float64)
    S = A @ A.transpose(-1, -2) + torch.eye(14, dtype=torch.float64)
    mu_m, S_m = m(mu, S)
    mu_mg = torch.einsum("kl,nl->nk", g, mu_m)
    S_mg = g @ S_m @ g.T
    mu_gm, S_gm = m(torch.einsum("kl,nl->nk", g, mu), g @ S @ g.T)
    assert (mu_gm - mu_mg).abs().max() < 1e-12
    assert (S_gm - S_mg).abs().max() < 1e-11


def test_mults_one_tower_gives_scalar_gains():
    m = HeadMixer([1, 3, 5, 7], irrep_labels=["l0", "l1", "l2", "l3"])
    assert all(tuple(d.shape) == (1, 1) for d in m.mixer_deltas)
    with torch.no_grad():
        m.mixer_deltas[2].fill_(0.5)            # gain 1.5 on the l2 head
    mu = torch.randn(3, 16)
    mu2, _ = m(mu, torch.ones(3, 16))
    assert torch.allclose(mu2[:, 4:9], 1.5 * mu[:, 4:9])
    assert torch.equal(mu2[:, :4], mu[:, :4])   # other heads untouched
```

- [ ] **Step 2: Run, verify failure**

Run: `python -m pytest tests/test_head_mixer_isotypic.py`
Expected: FAIL (`irrep_labels` unexpected keyword / `mixer_deltas` missing).

- [ ] **Step 3: Rewrite `vfe3/model/head_mixer.py`** — replace the `HeadMixer` class body (keep the module docstring, extend it with two sentences noting label-grouped components and that label grouping closes the dim-collision hazard):

```python
class HeadMixer(nn.Module):
    r"""Isotypic per-component mixer: one :math:`A_t = I + \Delta_t` per maximal run of
    equal-labeled blocks, embedded as :math:`\mathrm{blockdiag}_t(A_t \otimes I_{d_t})` --
    the full linear commutant of the tower for real-type irreps. Without labels the whole
    group must be one equal-dims component (the legacy behavior, byte-identical)."""

    def __init__(
        self,
        irrep_dims:   List[int],                      # gauge block sizes
        irrep_labels: Optional[List[str]] = None,     # per-block labels; None -> legacy equal-dims
    ) -> None:
        super().__init__()
        if len(irrep_dims) < 2:
            raise ValueError(
                f"HeadMixer needs >= 2 blocks to mix, got irrep_dims={irrep_dims}; a single-block "
                f"group (glk / so_k) has nothing to mix. Use block_glk (n_heads >= 2)."
            )
        if irrep_labels is None:
            if len(set(irrep_dims)) != 1:
                raise ValueError(
                    f"HeadMixer needs equal-size blocks for kron(A, I_d), got "
                    f"irrep_dims={irrep_dims}. A labeled irrep tower (so_n/sp_n) mixes per "
                    f"isotypic component instead."
                )
            runs = [(0, len(irrep_dims))]                       # one component: all blocks
        else:
            runs, i = [], 0                                     # maximal runs of equal labels
            while i < len(irrep_dims):
                j = i
                while j < len(irrep_dims) and irrep_labels[j] == irrep_labels[i]:
                    j += 1
                runs.append((i, j))
                i = j
        # components: (coordinate start, copies m, block dim d); spec layout makes runs contiguous
        starts = [0]
        for d in irrep_dims:
            starts.append(starts[-1] + d)
        self.components = [(starts[i], j - i, irrep_dims[i]) for i, j in runs]
        self.mixer_deltas = nn.ParameterList(
            nn.Parameter(torch.zeros(m, m)) for _, m, _ in self.components
        )

    @property
    def mixer_delta(self) -> nn.Parameter:
        r"""Back-compat accessor for the single-component (legacy equal-dims) mixer."""
        if len(self.mixer_deltas) != 1:
            raise AttributeError("mixer_delta is single-component only; use mixer_deltas")
        return self.mixer_deltas[0]

    def _A(self, t: int) -> torch.Tensor:
        d = self.mixer_deltas[t]
        return torch.eye(d.shape[0], device=d.device, dtype=d.dtype) + d

    def is_identity(self) -> bool:
        return all(bool((d.detach() == 0).all().item()) for d in self.mixer_deltas)

    def forward(
        self,
        mu:    torch.Tensor,             # (..., K) belief means
        sigma: torch.Tensor,             # (..., K) diagonal variances OR (..., K, K) full covariance
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_parts, sig_parts = [], []
        for t, (s, m, d) in enumerate(self.components):
            A = self._A(t)
            blk = mu[..., s:s + m * d].reshape(*mu.shape[:-1], m, d)
            mu_parts.append(torch.einsum("mn,...nd->...md", A, blk)
                            .reshape(*mu.shape[:-1], m * d))
            if sigma.dim() == mu.dim():                          # diagonal closed form
                sblk = sigma[..., s:s + m * d].reshape(*sigma.shape[:-1], m, d)
                sig_parts.append(torch.einsum("mn,...nd->...md", A * A, sblk)
                                 .reshape(*sigma.shape[:-1], m * d))
        mu_out = torch.cat(mu_parts, dim=-1)
        if sigma.dim() == mu.dim():
            return mu_out, torch.cat(sig_parts, dim=-1)
        # full covariance: exact sandwich M Sigma M^T with the block-diagonal commutant M
        M = self._dense_m(sigma.device, sigma.dtype)             # (K, K)
        return mu_out, M @ sigma @ M.transpose(-1, -2)

    def _dense_m(self, device, dtype) -> torch.Tensor:
        r"""blockdiag_t(A_t kron I_d) materialized once per call (K x K, full-cov path only)."""
        K = sum(m * d for _, m, d in self.components)
        M = torch.zeros(K, K, device=device, dtype=dtype)
        for t, (s, m, d) in enumerate(self.components):
            M[s:s + m * d, s:s + m * d] = torch.kron(
                self._A(t).to(device=device, dtype=dtype), torch.eye(d, device=device, dtype=dtype))
        return M
```

Required import change at the top of the file: `from typing import List, Optional, Tuple`.

NOTE on byte-identity: the single-component mu/diagonal-sigma einsums are the same ops on the same shapes as the old code (bit-identical). The full-covariance path switches from two einsums to the materialized sandwich; `tests/test_head_mixer.py`'s full-cov assertions use `allclose`, which this passes. If any existing assertion there uses `torch.equal` on the full-cov path, relax it to `torch.allclose(..., atol=1e-6)` with a one-line comment citing this plan.

- [ ] **Step 4: Run, verify pass (new + existing mixer suites)**

Run: `python -m pytest tests/test_head_mixer_isotypic.py tests/test_head_mixer.py tests/test_head_mixer_per_block.py tests/test_model.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add vfe3/model/head_mixer.py tests/test_head_mixer_isotypic.py
git commit -m "feat(head_mixer): isotypic label-grouped components (commutant of mixed towers)"
```

### Task 3: model wires labels into the mixer

**Files:**
- Modify: `vfe3/model/model.py` (~line 125-128, the `HeadMixer` construction)
- Modify: `train_vfe3.py`, `ablation.py` (stale "needs EQUAL blocks" toggle comments)
- Test: `tests/test_head_mixer_isotypic.py` (append)

- [ ] **Step 1: Write the failing test** — append to `tests/test_head_mixer_isotypic.py`:

```python
def test_so_n_mixed_tower_model_constructs_with_mixer_and_trains():
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel
    cfg = VFE3Config(vocab_size=20, embed_dim=8, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_mu_lr=0.05, e_phi_lr=0.0,
                     gauge_group="so_n", group_n=3,
                     irrep_spec=[("l0", 1), ("l1", 1), ("l0", 1), ("l1", 1)],
                     use_head_mixer=True, phi_precond_mode="none")
    model = VFEModel(cfg)                       # pre-fix: raises (unequal dims [1,3,1,3])
    assert [tuple(d.shape) for d in model.head_mixer.mixer_deltas] == [(1, 1), (1, 1), (1, 1), (1, 1)]
    with torch.no_grad():
        model.head_mixer.mixer_deltas[1].fill_(0.1)
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    _, loss, _ = model(tok, tgt)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.head_mixer.mixer_deltas[1].grad).all()
```

(The spec entries here are deliberately non-adjacent same-label blocks: runs treat them as four separate components, the documented less-expressive-but-still-equivariant behavior.)

- [ ] **Step 2: Run, verify failure**

Run: `python -m pytest tests/test_head_mixer_isotypic.py -k constructs`
Expected: FAIL with the equal-size-blocks ValueError (labels not passed).

- [ ] **Step 3: Implement** — in `vfe3/model/model.py` replace the construction line and refresh its comment:

```python
        # Opt-in Schur-commutant head mixer (default off). Built ONCE from the gauge group's
        # irrep blocks. Label-less groups need >= 2 EQUAL blocks (block_glk/tied_block_glk);
        # labeled irrep towers (so_n/sp_n) mix per isotypic component (mults-one towers get
        # per-head scalar gains -- the entire linear commutant there). Bad pairings fail here,
        # not at forward.
        self.head_mixer = HeadMixer(self.group.irrep_dims,
                                    irrep_labels=self.group.irrep_labels) \
            if cfg.use_head_mixer else None
```

In `train_vfe3.py` and `ablation.py`, update the `use_head_mixer` comment lines (currently "needs >=2 equal blocks ...") to: `needs >=2 equal blocks (block_glk/tied_block_glk) OR a labeled irrep tower (so_n/sp_n: per-isotypic-component mixing; mults-one towers get scalar gains)`. In `ablation.py`, delete the now-stale `"use_head_mixer": False` override from the `so3_tower` sweep arm and drop the sentence about it from the arm comment.

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest tests/test_head_mixer_isotypic.py tests/test_son_irreps.py tests/test_config.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add vfe3/model/model.py train_vfe3.py ablation.py tests/test_head_mixer_isotypic.py
git commit -m "feat(model): head mixer accepts labeled irrep towers (so_n/sp_n)"
```

### Task 4: public single-irrep builder in `irreps.py`

**Files:**
- Modify: `vfe3/geometry/irreps.py` (one new public function after `irrep_dim`)
- Test: `tests/test_cg.py` (create with this first test)

- [ ] **Step 1: Write the failing test** — create `tests/test_cg.py`:

```python
r"""Numerical Clebsch-Gordan intertwiners over the irrep registry."""

import pytest
import torch

from vfe3.geometry.generators import generate_son
from vfe3.geometry.irreps import irrep_generators


def test_irrep_generators_public_builder():
    G_def = generate_son(3, dtype=torch.float64)
    rho = irrep_generators(G_def, algebra="so", label="l2")
    assert rho.shape == (3, 5, 5)
    assert (rho + rho.transpose(-1, -2)).abs().max() < 1e-12
```

- [ ] **Step 2: Run, verify failure**

Run: `python -m pytest tests/test_cg.py`
Expected: FAIL (ImportError: cannot import `irrep_generators`).

- [ ] **Step 3: Implement** — in `vfe3/geometry/irreps.py`, after `irrep_dim`:

```python
def irrep_generators(
    G_def:   torch.Tensor,                 # (n_gen, N, N) defining-rep algebra basis (float64)

    *,
    algebra: str,                          # 'so' | 'sp'
    label:   str,                          # e.g. 'l2' (so), 'sym3' (sp)
) -> torch.Tensor:                         # (n_gen, d, d) generator images on the irrep
    """Build one irrep's generator images (the registry's public single-label entry point)."""
    key, p = _parse_label(algebra, label)
    return _IRREPS[key][1](G_def, p)
```

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest tests/test_cg.py tests/test_son_irreps.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add vfe3/geometry/irreps.py tests/test_cg.py
git commit -m "feat(irreps): public irrep_generators(label) builder"
```

### Task 5: CG solver (`vfe3/geometry/cg.py`)

**Files:**
- Create: `vfe3/geometry/cg.py`
- Test: `tests/test_cg.py` (append)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_cg.py`:

```python
def test_so3_selection_rules():
    from vfe3.geometry.cg import cg_intertwiners
    # l1 (x) l1 = l0 (+) l1 (+) l2 : each target multiplicity 1
    for c, n in (("l0", 1), ("l1", 1), ("l2", 1), ("l3", 0)):
        C = cg_intertwiners(3, algebra="so", label_a="l1", label_b="l1", label_c=c)
        assert C.shape[0] == n, (c, C.shape)
    # l1 (x) l2 = l1 (+) l2 (+) l3 : no l0
    assert cg_intertwiners(3, algebra="so", label_a="l1", label_b="l2", label_c="l0").shape[0] == 0
    assert cg_intertwiners(3, algebra="so", label_a="l1", label_b="l2", label_c="l3").shape[0] == 1


def test_cg_intertwiner_is_equivariant():
    from vfe3.geometry.cg import cg_intertwiners
    from vfe3.geometry.irreps import irrep_generators
    G_def = generate_son(3, dtype=torch.float64)
    ra = irrep_generators(G_def, algebra="so", label="l1")
    rb = irrep_generators(G_def, algebra="so", label="l2")
    rc = irrep_generators(G_def, algebra="so", label="l2")
    C = cg_intertwiners(3, algebra="so", label_a="l1", label_b="l2", label_c="l2")[0]  # (5, 15)
    gen = torch.Generator().manual_seed(0)
    coeff = 0.4 * torch.randn(3, generator=gen, dtype=torch.float64)
    ga = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, ra))
    gb = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, rb))
    gc = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, rc))
    x = torch.randn(3, generator=gen, dtype=torch.float64)
    y = torch.randn(5, generator=gen, dtype=torch.float64)
    lhs = C @ torch.kron(ga @ x, gb @ y)                 # C(g x (x) g y)
    rhs = gc @ (C @ torch.kron(x, y))                    # g C(x (x) y)
    assert (lhs - rhs).abs().max() < 1e-10


def test_cg_selection_enumerates_admissible_triples():
    from vfe3.geometry.cg import cg_selection
    sel = {(a, b, c) for a, b, c, _ in cg_selection(3, algebra="so",
                                                    labels=["l0", "l1", "l2"])}
    assert ("l1", "l1", "l2") in sel
    assert ("l1", "l2", "l1") in sel
    assert ("l0", "l1", "l1") in sel                     # l0 source acts as a learned gate
    assert ("l1", "l1", "l3") not in sel                 # target not in the spec
    # unordered source pairs: (l2, l1) never appears (canonical order a <= b)
    assert all(a <= b for a, b, _c in sel)


def test_cg_cost_guard():
    from vfe3.geometry.cg import cg_intertwiners
    with pytest.raises(ValueError, match="construction size"):
        cg_intertwiners(8, algebra="so", label_a="l3", label_b="l3", label_c="l3")
```

- [ ] **Step 2: Run, verify failure**

Run: `python -m pytest tests/test_cg.py`
Expected: FAIL (ModuleNotFoundError: `vfe3.geometry.cg`).

- [ ] **Step 3: Create `vfe3/geometry/cg.py`** — complete module:

```python
r"""Numerical Clebsch-Gordan intertwiners for VFE_3.0 irrep towers.

For irrep labels a, b, c of one structure algebra, an intertwiner is a map
C: V_a (x) V_b -> V_c with C rho_{a(x)b}(X) = rho_c(X) C for every algebra basis
element X, where rho_{a(x)b}(X) = rho_a(X) (x) I + I (x) rho_b(X) (the Leibniz action
on the tensor product). The solution space is computed NUMERICALLY: accumulate the
Gram matrix of the stacked Sylvester operators over the basis and take its null space
(eigh), so no symbol tables and no per-family sign conventions exist. Each null vector
is one independent intertwiner (multiplicity slot), orthonormal in Frobenius norm, and
is verified by an equivariance-residual assert (raise, not warn) before caching.

Row-major vec convention (torch.reshape): vec(C rho) = kron(I_dc, rho^T) vec(C) and
vec(rho_c C) = kron(rho_c, I_D) vec(C).
"""

from typing import Dict, List, Tuple

import torch

from vfe3.geometry.generators import generate_son, generate_sp
from vfe3.geometry.irreps import irrep_dim, irrep_generators

_CG_CACHE: Dict[Tuple[str, int, str, str, str], torch.Tensor] = {}


def _defining(N: int, algebra: str) -> torch.Tensor:
    if algebra == "so":
        return generate_son(N, dtype=torch.float64)
    if algebra == "sp":
        return generate_sp(N, dtype=torch.float64)
    raise ValueError(f"unknown algebra {algebra!r}; registered: 'so', 'sp'")


def cg_intertwiners(
    N:       int,                          # defining-rep dimension (N of SO(N); 2m of Sp(2m))

    *,
    algebra: str,                          # 'so' | 'sp'
    label_a: str,                          # first source irrep label
    label_b: str,                          # second source irrep label
    label_c: str,                          # target irrep label
    atol:    float = 1e-8,
) -> torch.Tensor:                         # (n_mult, d_c, d_a * d_b) float64; n_mult may be 0
    """All independent intertwiners V_a (x) V_b -> V_c (empty leading axis if none)."""
    key = (algebra, N, label_a, label_b, label_c)
    if key in _CG_CACHE:
        return _CG_CACHE[key]
    da = irrep_dim(N, algebra=algebra, label=label_a)
    db = irrep_dim(N, algebra=algebra, label=label_b)
    dc = irrep_dim(N, algebra=algebra, label=label_c)
    D = da * db
    if dc * D > 5000:
        raise ValueError(
            f"CG solve for ({label_a}, {label_b}) -> {label_c} over R^{N} exceeds the supported "
            f"construction size (d_c * d_a * d_b = {dc * D} > 5000); larger products await a "
            f"matrix-free solver."
        )
    G_def = _defining(N, algebra)
    ra = irrep_generators(G_def, algebra=algebra, label=label_a)
    rb = irrep_generators(G_def, algebra=algebra, label=label_b)
    rc = irrep_generators(G_def, algebra=algebra, label=label_c)
    I_a = torch.eye(da, dtype=torch.float64)
    I_b = torch.eye(db, dtype=torch.float64)
    I_c = torch.eye(dc, dtype=torch.float64)
    I_D = torch.eye(D, dtype=torch.float64)
    gram = torch.zeros(dc * D, dc * D, dtype=torch.float64)
    rho_ab = []
    for a in range(G_def.shape[0]):
        r = torch.kron(ra[a], I_b) + torch.kron(I_a, rb[a])        # (D, D) Leibniz action
        rho_ab.append(r)
        op = torch.kron(I_c, r.T.contiguous()) - torch.kron(rc[a], I_D)
        gram += op.T @ op
    evals, evecs = torch.linalg.eigh(gram)
    null = evecs[:, evals < atol]                                  # (dc*D, n_mult), orthonormal
    C = null.T.reshape(-1, dc, D).contiguous()
    for a in range(G_def.shape[0]):                                # build-time verification
        res = (C @ rho_ab[a] - torch.einsum("ij,mjk->mik", rc[a], C)).abs().max() \
            if C.shape[0] else torch.tensor(0.0)
        if float(res) > 1e-7:
            raise RuntimeError(
                f"CG intertwiner ({label_a}, {label_b}) -> {label_c} equivariance residual "
                f"{float(res):.3e} exceeds 1e-7 at generator {a}"
            )
    _CG_CACHE[key] = C
    return C


def cg_selection(
    N:       int,                          # defining-rep dimension

    *,
    algebra: str,                          # 'so' | 'sp'
    labels:  List[str],                    # the spec's irrep labels (duplicates allowed)
) -> List[Tuple[str, str, str, int]]:      # admissible (a, b, c, n_mult), a <= b, n_mult > 0
    """Enumerate admissible CG triples among the spec's labels (unordered source pairs:
    swapped duplicates are not independent bilinear maps, so a <= b canonically)."""
    uniq = sorted(set(labels))
    out: List[Tuple[str, str, str, int]] = []
    for i, a in enumerate(uniq):
        for b in uniq[i:]:
            for c in uniq:
                n = cg_intertwiners(N, algebra=algebra, label_a=a, label_b=b,
                                    label_c=c).shape[0]
                if n > 0:
                    out.append((a, b, c, n))
    return out
```

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest tests/test_cg.py`
Expected: all pass. (If `test_so3_selection_rules` fails on counts, the vec convention is wrong — the equivariance assert inside the solver will already have raised; fix the kron order there, not the test.)

- [ ] **Step 5: Commit**

```bash
git add vfe3/geometry/cg.py tests/test_cg.py
git commit -m "feat(cg): numerical Clebsch-Gordan intertwiner solver over the irrep registry"
```

### Task 6: `CGCoupling` module

**Files:**
- Create: `vfe3/model/cg_coupling.py`
- Test: `tests/test_cg.py` (append)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_cg.py`:

```python
def _tower_group():
    from vfe3.geometry.groups import get_group
    return get_group("so_n")(9, group_n=3,
                             irrep_spec=[("l0", 1), ("l1", 1), ("l2", 1)],
                             dtype=torch.float64)


def test_cg_coupling_zero_init_is_exact_passthrough():
    from vfe3.model.cg_coupling import CGCoupling
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    assert cpl.path_weights.shape[0] > 0
    mu = torch.randn(2, 4, 9, dtype=torch.float64)
    sig = torch.rand(2, 4, 9, dtype=torch.float64)
    mu2, sig2 = cpl(mu, sig)
    assert torch.equal(mu2, mu) and torch.equal(sig2, sig)


def test_cg_coupling_means_update_is_exactly_equivariant():
    from vfe3.model.cg_coupling import CGCoupling
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    with torch.no_grad():
        cpl.path_weights.copy_(0.3 * torch.randn(cpl.path_weights.shape[0],
                                                 dtype=torch.float64))
    g = torch.linalg.matrix_exp(
        torch.einsum("a,aij->ij", 0.4 * torch.randn(3, dtype=torch.float64), grp.generators))
    mu = torch.randn(5, 9, dtype=torch.float64)
    sig = torch.rand(5, 9, dtype=torch.float64)
    out_then_g = torch.einsum("kl,nl->nk", g, cpl(mu, sig)[0])
    g_then_out = cpl(torch.einsum("kl,nl->nk", g, mu), sig)[0]
    assert (out_then_g - g_then_out).abs().max() < 1e-12


def test_cg_coupling_self_product_reaches_other_types():
    # zero everything except one l1 (x) l1 -> l2 path: the l2 head must move, others must not.
    from vfe3.model.cg_coupling import CGCoupling
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    idx = next(p for p, (a, b, c) in enumerate(cpl.path_types)
               if (a, b, c) == ("l1", "l1", "l2"))
    with torch.no_grad():
        cpl.path_weights[idx] = 1.0
    mu = torch.randn(3, 9, dtype=torch.float64)
    mu2, _ = cpl(mu, torch.ones(3, 9, dtype=torch.float64))
    assert not torch.allclose(mu2[:, 4:9], mu[:, 4:9])   # l2 head updated
    assert torch.equal(mu2[:, 0:4], mu[:, 0:4])          # l0 and l1 heads untouched
```

- [ ] **Step 2: Run, verify failure**

Run: `python -m pytest tests/test_cg.py -k coupling`
Expected: FAIL (ModuleNotFoundError: `vfe3.model.cg_coupling`).

- [ ] **Step 3: Create `vfe3/model/cg_coupling.py`** — complete module:

```python
r"""Clebsch-Gordan between-block coupling for irrep towers (opt-in; default off).

The only exactly-equivariant cross-type information flow: linear equivariant maps between
inequivalent irreps are zero (Schur), so the coupling is BILINEAR through the numerically
solved CG intertwiners,

    mu'^(c,r) = mu^(c,r) + sum_p w_p C_p( mu^(a,i), mu^(b,j) ),

one learned scalar w_p per path (source copy pair x target copy x multiplicity slot),
zero-initialized so step 0 is byte-identical to the coupling-off path. Equivariance holds
for ANY weights because the weights multiply intertwiners. Covariance is MEANS-ONLY in
this phase: sigma passes through untouched (a bilinear map of Gaussians has no closed-form
pushforward; the honest sigma treatment belongs to the deferred F-term phase -- see the
2026-06-09 design spec). NEURAL-NETWORK EXCEPTION (sanctioned, default-off), the
use_head_mixer family.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn

from vfe3.geometry.cg import cg_intertwiners, cg_selection


class CGCoupling(nn.Module):
    r"""Bilinear CG coupling over the blocks of a labeled irrep tower."""

    def __init__(
        self,
        group_n:      int,                       # N of SO(N) / 2m of Sp(2m)
        algebra:      str,                       # 'so' | 'sp'
        irrep_dims:   List[int],                 # per-block dims
        irrep_labels: Optional[List[str]],       # per-block labels (REQUIRED non-None)
    ) -> None:
        super().__init__()
        if irrep_labels is None:
            raise ValueError(
                "CGCoupling requires a labeled irrep tower (gauge_group 'so_n'/'sp_n')"
            )
        starts = [0]
        for d in irrep_dims:
            starts.append(starts[-1] + d)
        blocks = list(zip(irrep_labels, starts[:-1], irrep_dims))     # (label, start, d)

        # one stacked intertwiner buffer per admissible type triple
        triples = cg_selection(group_n, algebra=algebra, labels=irrep_labels)
        self._triple_index = {}
        for t, (a, b, c, _n) in enumerate(triples):
            C = cg_intertwiners(group_n, algebra=algebra,
                                label_a=a, label_b=b, label_c=c)      # (n_mult, dc, da*db)
            self.register_buffer(f"cg_{t}", C.to(torch.float32))
            self._triple_index[(a, b, c)] = t

        # paths: source copy pair (i <= j for equal labels) x target copy x multiplicity slot
        self.paths: List[Tuple[int, int, int, int, int, int, int, int]] = []
        self.path_types: List[Tuple[str, str, str]] = []
        for (a, b, c, n_mult) in triples:
            t = self._triple_index[(a, b, c)]
            srcs_a = [(s, d) for lab, s, d in blocks if lab == a]
            srcs_b = [(s, d) for lab, s, d in blocks if lab == b]
            tgts_c = [(s, d) for lab, s, d in blocks if lab == c]
            for ia, (sa, da) in enumerate(srcs_a):
                for jb, (sb, db) in enumerate(srcs_b):
                    if a == b and jb < ia:                            # unordered copies
                        continue
                    for (sc, dc) in tgts_c:
                        for m in range(n_mult):
                            self.paths.append((sa, da, sb, db, sc, dc, t, m))
                            self.path_types.append((a, b, c))
        if not self.paths:
            raise ValueError(
                f"CGCoupling found no admissible CG paths for labels {irrep_labels} "
                f"(algebra {algebra!r}, N={group_n}); disable use_cg_coupling"
            )
        self.path_weights = nn.Parameter(torch.zeros(len(self.paths)))

    def is_identity(self) -> bool:
        return bool((self.path_weights.detach() == 0).all().item())

    def forward(
        self,
        mu:    torch.Tensor,             # (..., K) belief means
        sigma: torch.Tensor,             # (..., K) or (..., K, K); passes through UNTOUCHED
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = torch.zeros_like(mu)
        for p, (sa, da, sb, db, sc, dc, t, m) in enumerate(self.paths):
            x = mu[..., sa:sa + da]
            y = mu[..., sb:sb + db]
            xy = (x.unsqueeze(-1) * y.unsqueeze(-2)).reshape(*x.shape[:-1], da * db)
            C = getattr(self, f"cg_{t}")[m].to(dtype=mu.dtype)        # (dc, da*db)
            delta[..., sc:sc + dc] = delta[..., sc:sc + dc] \
                + self.path_weights[p].to(mu.dtype) * torch.einsum("cd,...d->...c", C, xy)
        return mu + delta, sigma
```

NOTE: `delta` is a fresh `zeros_like(mu)` — the slice-accumulate into it is autograd-correct as written; do not add `.clone()` aliasing guards.

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest tests/test_cg.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add vfe3/model/cg_coupling.py tests/test_cg.py
git commit -m "feat(cg): CGCoupling between-block module (means-only, exactly equivariant)"
```

### Task 7: threading + config + optimizer + docs

**Files:**
- Modify: `vfe3/model/block.py` (param + application), `vfe3/model/stack.py` (param + forward), `vfe3/model/model.py` (construction + three call sites), `vfe3/config.py` (field + validation), `vfe3/train.py` (optimizer group), `CLAUDE.md` (exception note), `train_vfe3.py` + `ablation.py` (toggle key)
- Test: `tests/test_cg.py` (append)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_cg.py`:

```python
def _e2e_cfg(**kw):
    from vfe3.config import VFE3Config
    base = dict(vocab_size=20, embed_dim=9, n_heads=3, max_seq_len=5, n_layers=1,
                n_e_steps=1, e_mu_lr=0.05, e_phi_lr=0.0,
                gauge_group="so_n", group_n=3,
                irrep_spec=[("l0", 1), ("l1", 1), ("l2", 1)],
                phi_precond_mode="none")
    base.update(kw)
    return VFE3Config(**base)


def test_use_cg_coupling_rejected_off_towers():
    from vfe3.config import VFE3Config
    with pytest.raises(ValueError, match="use_cg_coupling"):
        VFE3Config(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                   gauge_group="block_glk", use_cg_coupling=True)


def test_cg_model_step0_byte_identical_and_trains():
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    torch.manual_seed(0)
    base = VFEModel(_e2e_cfg())
    torch.manual_seed(0)
    cg = VFEModel(_e2e_cfg(use_cg_coupling=True))
    lg_base, _, _ = base(tok); lg_cg, _, _ = cg(tok)
    assert torch.equal(lg_base, lg_cg)                  # zero-init: step 0 byte-identical
    build_optimizer(cg, cg.cfg)                         # exact-coverage guard must pass
    with torch.no_grad():
        cg.cg_coupling.path_weights.add_(0.05)
    _, loss, _ = cg(tok, tgt)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(cg.cg_coupling.path_weights.grad).all()
    assert cg.cg_coupling.path_weights.grad.abs().sum() > 0
```

- [ ] **Step 2: Run, verify failure**

Run: `python -m pytest tests/test_cg.py -k "rejected or step0"`
Expected: FAIL (`use_cg_coupling` is not a config field).

- [ ] **Step 3: Implement the threading**

(a) `vfe3/config.py` — field directly under `use_head_mixer`:

```python
    # CG cross-type coupling (opt-in, default off; so_n/sp_n only): bilinear Clebsch-Gordan
    # between-block update on the means, exactly equivariant for any weights; sigma untouched
    # (means-only phase; see the 2026-06-09 design spec). NEURAL-NETWORK EXCEPTION (sanctioned,
    # default-off): learned scalar path weights, zero-init (step 0 byte-identical).
    use_cg_coupling:           bool  = False
```

and validation inside the existing `if self.gauge_group in ("so_n", "sp_n"):` block has nothing to add; AFTER that block's `elif` chain add:

```python
        if self.use_cg_coupling and self.gauge_group not in ("so_n", "sp_n"):
            raise ValueError(
                f"use_cg_coupling requires an irrep-labeled tower group ('so_n'/'sp_n'); got "
                f"gauge_group={self.gauge_group!r}"
            )
```

(b) `vfe3/model/block.py` — add the parameter after `head_mixer` in `vfe_block`'s signature:

```python
    cg_coupling:     Optional[Callable[..., 'tuple']]      = None,   # opt-in CG cross-type coupling (None -> off)
```

and apply it between the mixer and the norm:

```python
    if cg_coupling is not None:              # opt-in CG cross-type coupling: after mixing, before norm
        mu_cg, sigma_cg = cg_coupling(out.mu, out.sigma)
        out = BeliefState(mu=mu_cg, sigma=sigma_cg, phi=out.phi)
```

(c) `vfe3/model/stack.py` — add the same `cg_coupling` keyword to `vfe_stack` and forward it to `vfe_block` (one line in the signature mirroring `head_mixer`, one `cg_coupling=cg_coupling,` in the call).

(d) `vfe3/model/model.py` — construction right after the head-mixer line:

```python
        # Opt-in CG cross-type coupling (default off; so_n/sp_n only). Built ONCE from the
        # group's labels; CGCoupling raises at construction when no admissible paths exist.
        if cfg.use_cg_coupling:
            from vfe3.model.cg_coupling import CGCoupling
            self.cg_coupling = CGCoupling(
                cfg.group_n, "so" if cfg.gauge_group == "so_n" else "sp",
                self.group.irrep_dims, self.group.irrep_labels)
        else:
            self.cg_coupling = None
```

and pass `cg_coupling=self.cg_coupling,` at ALL THREE replay-parity sites: the `vfe_stack` call in `forward` (~line 468), the `vfe_stack` call in `diagnostics` (~line 788), and the `vfe_block` call in `attention_maps` (~line 889). (`getattr(self, "cg_coupling", None)` is unnecessary — the attribute always exists.)

(e) `vfe3/train.py` — in `build_optimizer`, after the head-mixer group (~line 91):

```python
    if getattr(model, "cg_coupling", None) is not None:         # use_cg_coupling=True CG path weights
        groups.append({"params": [model.cg_coupling.path_weights], "lr": cfg.m_mu_lr})
```

(f) `CLAUDE.md` — in hard-constraint exception (2), after the head-mixer sentence, add: `Its irrep-tower siblings (so_n/sp_n) are the isotypic per-type mixer (exactly equivariant under the tied gauge) and, under use_cg_coupling=True, learned scalar Clebsch-Gordan path weights (exactly equivariant for any weights; means-only sigma) -- both zero-init, default OFF.`

(g) `train_vfe3.py` + `ablation.py` — add the toggle next to `use_head_mixer`:

```python
    use_cg_coupling           = False,               # so_n/sp_n only: CG cross-type coupling (bilinear, exactly
                                                     # equivariant, means-only sigma; zero-init path weights)
```

- [ ] **Step 4: Run targeted, then the FULL suite**

Run: `python -m pytest tests/test_cg.py tests/test_head_mixer_isotypic.py`
Expected: all pass.
Run: `python -m pytest`
Expected: summary line shows 0 failed (expected total: prior 784 + 1 xpassed, plus the ~14 new tests from Tasks 1-7).

- [ ] **Step 5: Commit**

```bash
git add vfe3/model/block.py vfe3/model/stack.py vfe3/model/model.py vfe3/config.py vfe3/train.py CLAUDE.md train_vfe3.py ablation.py tests/test_cg.py
git commit -m "feat(cg): wire CGCoupling through config/model/stack/optimizer (use_cg_coupling)"
```

### Task 8: docs + spec correction + push

**Files:**
- Modify: `docs/edits/2026-06-09-edits.md` (or the dated file for the implementation day), `docs/superpowers/specs/2026-06-09-isotypic-cg-head-mixing-design.md`

- [ ] **Step 1: Correct the spec's degenerate-path example.** The spec's "Guards and failure modes" section cites "a single-type spec" as a no-path example; that is wrong (self-products such as l2 (x) l2 -> l2 are admissible, so single-type specs DO get paths). Replace that parenthetical with: `(no admissible triples -- rare, since self-products l (x) l -> l are usually admissible; reachable only for towers whose products all land outside the spec's labels)`. Also amend the spec's "ordered pair" sentence to note the implemented canonicalization: unordered source pairs a <= b (swapped duplicates are not independent bilinear maps), and copies i <= j within an equal-label pair.

- [ ] **Step 2: Append a brief section to the dated edits doc** (one doc per day; create the day's file if implementing on a later date) summarizing: irrep_labels field, isotypic HeadMixer, cg.py solver, CGCoupling + use_cg_coupling, threading sites, and the full-suite pass count read from the pytest summary line.

- [ ] **Step 3: Commit and push**

```bash
git add docs/
git commit -m "docs: isotypic + CG head mixing implementation notes; spec corrections"
git push
```

## Self-review notes (already applied)

Spec coverage: irrep_labels (Task 1), isotypic mixer incl. legacy byte-identity and label grouping (Tasks 2-3), CG solver with build-time verification, cache, cost guard, selection enumerator (Tasks 4-5), CGCoupling with zero-init/means-only/no-path guard (Task 6), seam threading incl. diagnostics/attention_maps parity, config validation, optimizer group, CLAUDE.md exception, entry-point toggles (Task 7), docs + the two spec corrections discovered while planning (Task 8). Deviations from the spec, both documented in Task 8: unordered source pairs (a <= b) instead of ordered, and the corrected degenerate-path example. Type consistency: `mixer_deltas` (ParameterList) with a single-component `mixer_delta` property; `CGCoupling.path_weights`, `.paths`, `.path_types` used consistently across Tasks 6-7.
