# M-step natural gradient on phi: VFE_2.0 vs V3_Transformer

Date: 2026-06-24. Question: VFE_2.0's `riemannian_adam` got pretty good results on the
gauge-frame M-step — how does V3's M-step nat-grad on phi compare?

Scope note: in both repos the gauge frame is learned as **phi Lie-algebra coordinates**
(`phi_embed`, shape `(V, n_gen)`), not as group matrices. The K×K element `Omega = exp(phi·G)`
is rebuilt downstream and never stored. So this is an apples-to-apples coordinate-table
comparison. The E-step belief-frame refinement (`belief.phi`, gated on `e_phi_lr`) is a
*separate* object reusing the same `precondition_phi_gradient` + `retract_phi` helpers; it is
OFF by default in V3 (`e_phi_lr=0.0`) and is not what "M-step nat grad on phi" refers to.

## The metrics are identical; only the optimizer wrapper differs

`vfe3/geometry/phi_preconditioner.py` and `VFE_2.0/transformer/core/gauge_preconditioner.py`
implement the *same* two metrics: the Cartan-involution Killing form
`g̃_ab = 2K·tr(GₐᵀG_b) − 2tr(Gₐ)tr(G_b)` and the exp-map pullback
`G_ab(φ) = ⟨D_φexp[Tₐ], D_φexp[T_b]⟩_F = ΨᵀH(φ)Ψ`, with the same `Ψ(z)=(eᶻ−1)/z` series.
The difference between the codebases is entirely in **how the preconditioned gradient is
consumed by the optimizer**.

| | VFE_2.0 `RiemannianAdamW` (phi path) | V3 `GaugeNaturalGradAdamW` (opt-in) | V3 default |
|---|---|---|---|
| On by default? | **YES** (`gauge_parameterization='phi'`) | No (`m_phi_natural_grad=False`) | — plain AdamW |
| nat-grad direction | `grad ← grad @ K_inv` | `nat = G(φ)⁻¹·grad` (active rows) | none |
| moment rule on phi | **full AdamW** (m, v, bias-corr) | **heavy-ball only** `buf=0.9·buf+nat` | full AdamW |
| Adam 2nd moment (v) on phi | **applied** | **bypassed** (grad→None) | applied |
| bias correction on phi | yes | no | yes |
| default metric | `killing` (per-block, conformal) | `none` (→ bare heavy-ball SGD) | n/a |
| retraction | chart addition (`p -= lr·step`) | chart addition (`p.add_`) | chart addition |
| phi weight decay | **pinned to 0** (gauge protection) | 0 (when nat on) | **0.065** |
| phi LR | 0.0075 (live) / 0.0025 (VFE1 preset) | `m_phi_lr=0.015` | 0.015 |

Refs: VFE_2.0 `transformer/training/optimizer.py:193-210,263`, `transformer/vfe/trainer.py:346-349,445,461`,
`config.py:667-668`. V3 `vfe3/gauge_optim.py:121-145`, `vfe3/train.py:74-94,183-199`,
`vfe3/config.py:372,445,463-464,474`.

## The load-bearing finding: VFE_2.0's Killing preconditioning on phi was a near-no-op

V3's `gauge_optim.py` docstring asserts: *"a position-dependent metric cannot be realized by
preconditioning the gradient and then handing it to AdamW: Adam divides by the per-coordinate
second moment, which re-flattens any metric scaling."* This is precisely what VFE_2.0 does on
the phi path. The claim was verified numerically (`scratch` torch experiments):

1. **Conformal metric (exact cancellation).** For a scalar conformal factor `c`,
   `precondition(g → c·g)` then Adam reproduces plain Adam **byte-for-byte** — `c` cancels in
   `mhat/√vhat`. Verified to machine epsilon: max trajectory diff `1.39e-17` with `eps=0`
   (and `~1e-9` with the default `eps=1e-8`, the eps-floor being the sole residual), for
   `c ∈ {0.37, 5.0, 100.0}`.

   VFE_2.0's default phi metric is **Killing** (`omega_metric='killing'`), and on the shipped
   Frobenius-orthonormal `E_ij` generator basis the per-block Killing-Cartan form reduces to
   `c·I` on each `sl(d_h)` block after center regularization (off-diagonal generators get `2K`;
   the traceless-diagonal directions get `2K`; the center is lifted to `2K`). So it is conformal,
   and **fed to Adam it cancels exactly**. VFE_2.0's "Riemannian" preconditioning on the phi
   table contributed essentially nothing to the step *direction*.

   Corollary: **VFE_2.0's good-results phi M-step ≈ plain AdamW on phi coordinates**, with
   Adam's per-coordinate second-moment adaptivity and bias correction doing the real work.

2. **Non-conformal / pullback metric (partial re-flatten).** For a genuinely anisotropic
   off-diagonal SPD metric-inverse `M` (cond 250, off-diag mass 0.66), feeding the `M·g` stream
   to Adam yields a step at cosine **0.85** to the true natural gradient `M·g` and **0.31** to
   plain Adam — a muddled hybrid that is *neither*. Adam's per-coordinate `1/√vhat` weights on
   the `M·g` stream (`[1.27, 1.39, 0.90, 0.22, 0.30, 0.37]`) drag each coordinate back toward
   unit RMS, partially undoing the metric's anisotropy. So precondition-then-Adam cannot realize
   an off-diagonal natural gradient. This justifies V3's choice to route the pullback metric
   through heavy-ball momentum (no `v`) — only that path keeps the `M·g` direction exact.

The two are mathematically at odds: **no single optimizer simultaneously preserves an
off-diagonal metric direction AND applies Adam's diagonal adaptivity in the same coordinates.**

## What this means for the comparison

- **The "Riemannian" label oversold what VFE_2.0 actually did on phi.** Its documented results
  (~60 PPL; 71.6 test PPL GL(15)) came from plain-AdamW dynamics on the phi coordinate table —
  conformal Killing preconditioning canceled in Adam. The genuinely-Riemannian path in VFE_2.0
  (the `omega_direct` group: Adam moments in algebra coords + group-exp retraction onto GL⁺(K),
  `optimizer.py:397-558`) is **test-only**; no documented PPL run used it.

- **V3's default M-step on phi is already plain AdamW** — structurally the closest thing to what
  worked in VFE_2.0. So if V3 isn't matching VFE_2.0, the gap is almost certainly **not** "missing
  Riemannian geometry." The real, factual differences in the default configs are:
  - **phi weight decay**: V3 default `phi_weight_decay=0.065` decays `phi_embed` toward 0,
    shrinking the gauge frame toward `Omega=I` (the *ungauged* transformer). VFE_2.0 **pins phi
    weight decay to 0** (gauge protection). V3 already documents that `phi_weight_decay=0` gives
    "full gauge-frame protection" (`config.py:471-473`) — it just isn't the default.
  - **phi LR**: V3 `m_phi_lr=0.015` vs VFE_2.0 `0.0075` (live) / `0.0025` (VFE1 preset).

- **V3's opt-in geometric path trades adaptivity for metric fidelity.** When
  `m_phi_natural_grad=True`, V3 uses heavy-ball momentum with no Adam `v` and no bias correction,
  and its default `phi_precond_mode='none'` makes it *bare heavy-ball SGD on phi* unless you set
  `pullback_per_block` (config even warns: killing is conformal → effective-LR rescale only;
  `config.py:1184-1191`). This is geometrically purer than VFE_2.0 but is **not** the recipe that
  gave VFE_2.0 its results.

## Recommendations

1. **To match VFE_2.0's effective phi M-step in V3 (cheapest):** stay on the default plain-AdamW
   phi path, but set `phi_weight_decay=0.0` (let `mass_phi` handle frame-norm shrinkage in the
   loss) and drop `m_phi_lr` toward `~0.0075`. That reproduces VFE_2.0's actual behavior
   (AdamW + adaptivity + protected gauge frame), since its Killing preconditioning was a no-op.

2. **If you want genuine position-dependent geometry** (which VFE_2.0 never actually ran on phi):
   `m_phi_natural_grad=True` + `phi_precond_mode='pullback_per_block'`. Expect to retune
   `m_phi_lr`/`m_gauge_momentum` from scratch — you lose Adam's per-coordinate adaptive LR and
   bias correction, which are strong empirical stabilizers against the `exp(‖φ‖)` gradient
   blow-up in non-compact GL(K) directions.

3. **If you want adaptivity AND a real (off-diagonal) metric** — the one design that gives both —
   port VFE_2.0's `omega_direct` style: keep Adam's `m,v` in algebra/body-frame coordinates and
   retract with the group exponential. V3 currently has no omega-matrix M-step; its gauge frame
   is phi-coordinates only.

## Empirical root cause (V3 ablation, seed 6, wikitext-103): the nat-grad freezes phi

`vfe3_ablation_results/gauge_mstep_optim` (AdamW vs killing vs pullback, identical model):

| phi M-step | val PPL | `phi_norm_mean` | `weight_norm_phi` (5k→15k) | `cos_nat_phi` |
|---|---|---|---|---|
| AdamW | **144.5** | 6.2 | 908 → 622 (moves) | — |
| pullback | 252.3 | 1.2 | 190.4 → 190.5 (frozen) | 0.988 |
| killing | 271.8 | 0.90 | 190.2 → 190.3 (frozen) | 1.000 |

Cross-referenced against `gauge_transport` (same seed): Omega=I ungauged = 267–270 PPL,
frozen-random gauge = 277–281, gauge-learned-via-AdamW = 154. So **killing nat-grad (271.8)
sits exactly on the ungauged/frozen floor** — it is not learning the gauge frame at all.

Mechanism, read straight off the metrics:
- The **raw** phi gradient is tiny and well-behaved in every run: `grad_norm_phi` ≈ 0.06–0.08.
- AdamW divides by `√v` (per-coordinate RMS ≈ the gradient scale), so the effective per-coordinate
  step is ≈ `lr` ≈ 0.01 regardless of the raw magnitude. phi moves; `weight_norm_phi` climbs to
  ~900 and `phi_norm_mean` to ~6. The gauge frame trains.
- The nat-grad path applies **no gradient-magnitude normalization**, and the Killing metric
  *inverse* is `≈ 1/(2·d_h)·I` (conformal), so it **divides** the already-tiny 0.06 gradient by
  ~20. Step ≈ `lr · (1/20) · 0.06 ≈ 3e-5` per coordinate; with heavy-ball over 15k steps that is
  negligible against a table norm of 190. phi stays at init → ungauged PPL.
- `cos_nat_phi = 1.000` (killing) confirms the metric is doing **nothing** to the direction — it
  is a pure (shrinking) rescale. `pullback_cond_median ≈ 2.67` and `cos ≈ 0.988` confirm that at
  the operating point (‖φ‖ ~ 1) even the "exact" pullback metric is **near-isotropic** — there is
  essentially no geometry to exploit, so it too mostly just shrinks the step (phi crawls from
  0.90 to 1.2 instead of staying pinned).

So AdamW does not "beat the natural gradient" — at this operating point the natural metric is
near-identity (no curvature to exploit), and the nat-grad path additionally discards the one thing
phi needs to train at all: per-coordinate gradient-magnitude normalization. AdamW's `1/√v` IS a
diagonal empirical-Fisher preconditioner that tracks the real loss-gradient scale; the exp-map /
Killing metric tracks the *parameterization* geometry, which here carries no useful information and
costs phi its trainability.

### Why this is structural, not a tuning miss
- The metric is the geometry of the GL(K) *parameterization*, not the Fisher metric of the
  cross-entropy *loss*. Amari's natural-gradient speedup is a statement about the output
  distribution's Fisher; the manifold metric has no such guarantee and, at small ‖φ‖, is ≈ I.
- Cranking `m_phi_lr` to undo the ~20× Killing shrink does not rescue it: the phi coordinates have
  heterogeneous gradient scales (compact vs non-compact directions; frequent vs rare embedding
  rows), so a single global heavy-ball LR blows up the loud coordinates before the quiet ones move.
  Per-coordinate `1/√v` is exactly the fix and the conformal metric cannot supply it. (The
  `m_phi_lr_natgrad` sweep at `m_phi_lr=0.0005` gives 271.9 — even more frozen, the wrong direction.)

### If you want a geometric step that is competitive
1. Restore normalization: run Adam's `m,v` **on the natural gradient** `nat = G⁻¹·grad`. For the
   conformal Killing case this collapses to AdamW (which is why AdamW already wins); for pullback it
   is the "muddled hybrid," but at least phi moves and it would beat both frozen nat-grad runs.
2. Port VFE_2.0's `omega_direct`: Adam moments in algebra/body-frame coordinates + group-exp
   retraction — adaptivity AND a real metric AND exact retraction in one optimizer.
3. Or accept the data's verdict: AdamW (a diagonal empirical Fisher) is the right preconditioner for
   the phi block; the manifold metric is near-isotropic here and buys nothing.

## Caveats

The conformal-cancellation result is exact and metric-independent (algebraic). The "near" in
"near-no-op" covers any departure of VFE_2.0's actual generator basis from exact Frobenius
orthonormality; for the shipped `E_ij` basis it is exact. The pullback cosine numbers (0.85/0.31)
depend on `M`'s conditioning and the gradient stream's RMS profile, but the qualitative verdict
(neither pure nat-grad nor plain Adam) is robust because Adam's diagonal `v` cannot represent an
off-diagonal congruence. The claim that Adam adaptivity empirically stabilizes `exp(‖φ‖)` blow-up
is V3's stated rationale, reported here as the tradeoff, not independently training-verified.
