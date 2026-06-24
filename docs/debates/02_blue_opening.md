# Blue Opening — Defense of Critique F3 (three-preconditions / abstract undercount)

## Claim under defense (restated)

The critique asserts a real, actionable gap in the **current vault copy** of
`GL(K)_attention.tex`: the abstract advertises the reduction to standard attention as
**two** successive limits (isotropic covariances, flat gauge connection), while the
manuscript's own body states the reduction holds only under **three** preconditions —
constant gauge, isotropic covariance, **and** key-norm constancy. The abstract therefore
silently drops the one precondition the body itself flags as approximate. The proposed
fix: bring the abstract into agreement with body line 1358 by naming key-norm constancy
(or, equivalently, attaching the body's `≈` to the abstract's reduction claim), and
tighten the body's line-1241 phrasing that layer normalization "normalizes `||μ_j||`"
into a constant key-norm.

I defend this critique. It is correct on the manuscript's own text, it is internally
load-bearing (the body and supplementary disagree on *which* norm LN controls), and the
fix is surgical.

## Position

The edit is warranted. Two changes, in priority order:

1. **(Required) Abstract count fix.** Replace, at vault line 47,
   "Two successive limits (isotropic covariances, flat gauge connection) recover the
   standard rule" with a three-precondition statement that adds key-norm constancy and
   carries the approximate marker, e.g.: *"Three conditions — flat gauge connection,
   isotropic covariances, and key-norm constancy (from layer normalization or
   high-dimensional concentration) — recover the standard rule
   `β_ij ≈ softmax(Q_i K_j^T / √d_k)`,"* with `∝` replaced by `≈` to match the body's
   own honesty marker. The body already calls the third precondition "the chain's one
   approximate step, recorded with `≈`" (line 1358); the abstract must not launder that
   `≈` into the exact-looking `∝`.

2. **(Recommended) Body norm fix at line 1241.** Sharpen "layer normalization, which
   normalizes `||μ_j||` to be constant across tokens" so it does not silently claim that
   normalizing the **pre-projection embedding** `μ_j` yields constant **projected key**
   norm `||K_j|| = ||W_K^T μ_j||`. Pre-LN normalizes the embedding to a sphere; the learned
   projection `W_K` then re-introduces per-token norm spread `||W_K^T μ_j||` unless `W_K`
   is an isometry. State the bias in the variable the cancellation actually requires.

## Evidence

**Vault, abstract (line 47):**
> "Two successive limits (isotropic covariances, flat gauge connection) recover the
> standard rule $\beta_{ij} \propto \operatorname{softmax}(Q_i K_j^\top / \sqrt{d_k})$."

Two limits, exact `∝`, no key-norm condition.

**Vault, body (line 1358):**
> "Standard scaled dot-product attention is recovered under three preconditions ...:
> constant gauge ... isotropic covariance ... and key-norm constancy supplied either by
> layer normalization or by high-dimensional concentration of $\|\mu_j\|^2$. ... The third
> precondition is the chain's one approximate step, recorded with $\approx$."

The manuscript's own body says three, and explicitly grades the third as approximate.
The abstract and body are in direct numerical and modal disagreement. That is the gap.

**Vault, body (line 1224):** the residual key bias is written
$-\tfrac{1}{2\sigma^2}\|\mu_j\|^2$ — the **embedding** norm. **Vault, body (line 1241):**
LN "normalizes $\|\mu_j\|$." Self-consistent so far. But the manuscript does not stay in
that variable:

**Supplementary, flat-limit rule (line 774):**
> $\beta_{ij}^{(\text{flat})} = \text{softmax}_j\left(-\frac{\|Q_i-K_j\|^2}{\tau} - \lambda_K\|K_j\|^2\right)$

— bias in the **projected key norm** $\|K_j\|^2$, not $\|\mu_j\|^2$. **Supplementary,
empirical validation (line 785, Fig. line 791):** the key-norm bias analysis correlates
$\rho(\|K_j\|^2,\text{attention weight})$ — again the **projected** key $K_j = W_K^\top\mu_j$,
across all 144 BERT heads. The manuscript's own experiment measures the projected-key bias;
the body's LN argument cancels the embedding-norm bias. LN regulating `||μ_j||` does not by
itself null the quantity the manuscript validates against, namely `||K_j||²`. That is the
substance behind precondition fix (2).

**External canon — what the standard rule actually is.** Vaswani et al. (2017), *Attention
Is All You Need*, §3.2.1, Eq. (1): `Attention(Q,K,V) = softmax(QK^T / √d_k) V`, with **no**
additive `||K_j||²` term. For the gauge derivation to land on this exact form, the
`-λ_K ||K_j||²` term of supplementary line 774 must vanish — i.e. projected-key-norm
constancy is a genuine, named precondition, not a free byproduct of the two limits the
abstract lists. The abstract claims the canonical Vaswani form on two limits; the canonical
form requires the third.

**External canon — LN does not fix projected-key norm.** Ba, Kiros & Hinton (2016),
*Layer Normalization*, normalizes the summed inputs of a layer to zero mean and unit
variance **before** the learned affine/projection downstream; it constrains the input
statistic, not the post-projection norm `||W_K^T μ_j||`. Constant `||μ_j||` implies constant
`||W_K^T μ_j||` only if `W_K` preserves norms on the embedding sphere (an isometry / scalar
multiple of a partial isometry), which trained `W_K` is not. So the body-line-1241 claim
that LN delivers "constant key-norms" is the over-strong step the critique flags.

## Strongest attack on this critique, pre-empted

The strongest red move: *"No gap — the abstract is a permissible compression. The body
already states all three preconditions and grades the third with `≈`; abstracts routinely
omit approximate side-conditions, and 'two successive limits' refers specifically to the two
**exact** algebraic specializations (constant gauge + isotropic-with-absorption), with
key-norm constancy being an approximation rather than a 'limit.' Calling this an error is a
style nitpick, not a substantive defect."*

Rebuttal, pre-empted on three points. First, the abstract does not merely omit the third
condition — it makes a **stronger, exact** claim than the body supports: it writes `∝`
(exact proportionality) where the body writes `≈` (approximate). An abstract that upgrades
the body's approximate result to an exact one is not compression; it is a fidelity error a
reader cannot detect without reaching line 1358. Second, the "two limits are the exact ones"
defense actually concedes the critique: if precondition three is categorically different
(approximate, not a limit), the abstract should say so, because a reader counts "two limits
→ standard rule" as a complete reduction and will not supply the missing approximate step.
Third, the manuscript's own supplementary spends a figure and 144-head correlation study
(lines 785–805) demonstrating that the key-norm bias is **measurable and reproducible**,
including a residual `r̄` degradation with sequence length (line 903) attributed precisely to
"the residual bias term that the flat-bundle isotropic limit neglects." A condition whose
violation the authors quantify empirically is load-bearing, not cosmetic. The abstract
hiding it is therefore a substantive undercount.

## Falsification conditions

This critique is **not** defensible if any of the following holds.

1. **The abstract already carries the approximate marker or the third condition.** If vault
   line 47 in fact reads `≈` rather than `∝`, or names key-norm/LN constancy, the
   numerical-disagreement claim collapses. *Checked: line 47 reads `∝` and lists exactly two
   limits with no key-norm clause. Not falsified.*

2. **Constant `||μ_j||` provably forces constant `||W_K^T μ_j||`.** If `W_K` in the recovered
   regime is constrained to an isometry (or the isotropic-absorption step `W_Q W_K^T =
   σ^{-2} Ω^{-T}` independently pins `||W_K^T μ_j||` constant on the LN sphere), then LN does
   deliver constant projected key norms and body fix (2) is unwarranted. The abstract count
   fix (1) would still stand, because the body itself still labels the third precondition
   approximate at line 1358. *No such isometry constraint is imposed in the recovery
   derivation (lines 1266–1305); `W_Q, W_K` are general learned `GL(d_head)` factors. Fix (1)
   not falsified; fix (2) survives unless such a constraint is added.*

3. **The "two limits" wording is internally defined to subsume key-norm constancy.** If an
   earlier passage explicitly defines "the two limits" as a bundle that includes key-norm
   constancy, the abstract is shorthand for a defined term and there is no undercount. *No
   such definition precedes line 47; the body at 1358 introduces "three preconditions" as a
   distinct, higher count. Not falsified.*

4. **The body / supplementary agree on the norm variable.** If the supplementary's flat-limit
   bias and empirical study used `||μ_j||²` (embedding) rather than `||K_j||²` (projected),
   the body's LN argument and the supplementary's validation would be in the same variable
   and fix (2) would be moot. *Supplementary line 774 and line 785 both use `||K_j||²`
   (projected). Not falsified.*

On the verified vault text, none of the falsifiers fire. The critique stands: fix (1) is
required (abstract undercounts and over-claims exactness); fix (2) is the correct,
narrower tightening of the body's LN-to-key-norm step.
