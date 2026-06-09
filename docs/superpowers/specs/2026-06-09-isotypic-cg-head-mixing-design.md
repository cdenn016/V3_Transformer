# Isotypic and Clebsch–Gordan head mixing for irrep towers

Date: 2026-06-09. Status: approved design, pre-implementation.

## Motivation

The `so_n`/`sp_n` gauge groups carry the embedding as a direct sum of irreps, so attention
heads can be different irreps of one structure group, with unequal dimensions. The existing
`HeadMixer` (`kron(A, I_d)`) requires equal blocks and therefore refuses every mixed tower.
This refusal is mathematically correct for what the mixer is — by Schur's lemma the space of
linear equivariant maps between inequivalent irreps is zero — but it leaves two genuine
capabilities unbuilt. First, a mixed tower with repeated labels (multiplicity greater than
one) has a nontrivial commutant that the current mixer cannot express: V2 had exactly this
per-type mixer and the V3 port dropped it. Second, information flow across inequivalent
types is possible equivariantly only through at-least-bilinear maps, whose canonical form is
the Clebsch–Gordan (CG) decomposition rho_a (x) rho_b = (+)_c N^c_ab rho_c. The existing
`cross_couplings` feature is not a substitute: it enlarges the gauge algebra with off-block
generators, and between inequivalent irreps the bracket closure floods the rectangular block
(the adjoint action on Hom(V_b, V_a) has no trivial component) and dissolves the tower toward
gl(K), abandoning the representation structure the tower exists to provide.

## Scope and phases

Phase 1 implements the isotypic Schur mixer: one learned matrix per repeated label, the full
linear commutant of the tower. Phase 2a implements the CG coupling as a between-block map at
the same seam as the head mixer, exactly equivariant, means-only on the covariance. Phase 2b
— the trilinear coupling as a term in the free energy descended by the E-step — is sketched
at the end of this spec and deliberately deferred: no code, no config keys, until it is its
own project. The theoretically pure path is both toggles off, unchanged and default.

## Mathematical specification

### Phase 1: isotypic mixer

A spec `[(label_t, mult_t), ...]` lays out blocks contiguously in order, copies of one label
adjacent. Group the blocks into isotypic components, one per spec entry. The embedded group
element restricted to component t is I_{m_t} (x) rho_t(g), and for real-type irreps (the
commutant-dimension-1 property verified at irrep build time) the commutant of the full tower
is exactly

    M = blockdiag_t ( A_t (x) I_{d_t} ),    A_t = I_{m_t} + Delta_t,

with one learned Delta_t in R^{m_t x m_t} per component, zero-initialized so step 0 is
byte-identical to the mixer-off path. M commutes with every tower element, so the mixer
operation is exactly gauge-equivariant under the tied gauge on the full-covariance path; on
the diagonal family the closed form sigma'[m] = sum_n A[m,n]^2 sigma[n] applies per component
with the existing caveat (equivariant only under diagonal gauges), identical in status to the
current mixer under `tied_block_glk`. A component with m_t = 1 contributes a learned scalar
gain (1 + delta) on its head — the entire linear commutant there, stated honestly. Mean and
covariance rules are the existing `HeadMixer` rules applied per component.

For an equal-blocks single-type group the construction reduces to one component with
m = n_heads, reproducing today's `kron(A, I_d)` byte-for-byte; `block_glk` /
`tied_block_glk` therefore keep their current behavior and tests.

### Phase 2a: CG between-block coupling

For an ordered pair of source types (a, b) and a target type c with rho_c contained in
rho_a (x) rho_b, an intertwiner is a map C: V_a (x) V_b -> V_c with

    C ( rho_a(g) x (x) rho_b(g) y ) = rho_c(g) C(x (x) y)        for all g,

equivalently C rho_{a(x)b}(X) = rho_c(X) C for every algebra basis element X. The
intertwiners are computed NUMERICALLY from the existing irrep machinery: stack the Sylvester
operators of that linear condition over the n_gen basis elements and take the null space
(SVD), orthonormalize, and keep one basis intertwiner per multiplicity slot. No symbol
tables and no sign conventions; the construction works uniformly for the so(N)
symmetric-traceless family and the sp(2m) Sym^p family, in float64, with a build-time
equivariance-residual assert (raise, not warn — the same discipline as the irrep
homomorphism check) and a per-(algebra, N, a, b, c) cache.

The coupling acts on the converged belief at the block boundary, immediately after the head
mixer:

    mu'^(c,r) = mu^(c,r) + sum_p  w_p * C_p ( mu^(a,i), mu^(b,j) ),

where the path set p ranges over all admissible triples of isotypic types (a, b, c) with
N^c_ab > 0 — INCLUDING the self-products a = b, which are what give a multiplicity-one spin
tower any cross-type flow at all — over copy indices (i, j, r) within the source and target
components, and over independent intertwiners when the multiplicity N^c_ab exceeds one. As
built, source pairs are canonicalized UNORDERED (a <= b lexicographically; swapped duplicates
are not independent bilinear maps), and copies within an equal-label pair take i <= j. Each
path carries one learned scalar w_p, zero-initialized (step 0 byte-identical). Intertwiners
are normalized to unit Frobenius norm so the w_p share a common scale. Realistic specs make
the path count small (a four-type mults-one tower has on the order of tens of paths); a
build-time count is logged, and no cap is imposed.

Covariance is MEANS-ONLY in this phase: sigma passes through the coupling untouched. A
bilinear map of Gaussians has no closed-form pushforward; leaving sigma fixed preserves SPD
trivially and keeps the update exactly equivariant on the means (both arguments and the
output transform in their representations, so the update commutes with the gauge action for
any w). The approximation is documented at the seam; the honest sigma treatment belongs to
the F-term phase, where the expectation E_q handles it properly.

### Equivariance summary

Phase 1 is exactly equivariant on the full-covariance path and diagonal-approximate exactly
as the current mixer is. Phase 2a is exactly equivariant on the means for any weights; sigma
is untouched. Neither phase touches F, beta, or the E-step stationarity: both are
between-block maps, the same contract the head mixer already has.

## Architecture and components

`GaugeGroup` gains an optional field `irrep_labels: Optional[List[str]] = None`, one label
per block, populated by the `so_n`/`sp_n` builders and left None by every legacy builder —
a backward-compatible dataclass extension (all existing constructions remain valid).

`vfe3/model/head_mixer.py`: `HeadMixer` generalizes to type-grouped blocks. Construction
takes `irrep_dims` and optional `irrep_labels`; with labels, components are maximal runs of
equal labels (which the spec layout makes contiguous); without labels, the current behavior
is kept exactly — equal dims construct the single-component mixer, unequal dims raise.
Grouping on labels rather than dims also closes the latent footnote that a future irrep
family could collide dimensions between inequivalent irreps. One `nn.ParameterList` of
per-component deltas replaces the single delta; the single-component case stores the same
tensor shape as today.

`vfe3/geometry/cg.py` (new): the intertwiner solver and cache. Public surface:
`cg_intertwiners(N, algebra, label_a, label_b, label_c) -> (n_mult, d_c, d_a * d_b)` in
float64, plus a `cg_selection(N, algebra, labels) -> [(a, b, c, n_mult), ...]` enumerator
that tests containment numerically (null-space dimension). Built on `vfe3.geometry.irreps`
(the tensor-power representations and structure constants); registry-shaped so a future
irrep family participates by registration alone. Cost guard mirroring the irreps guard
(product dimension cap), with the same clear error.

`vfe3/model/cg_coupling.py` (new): `CGCoupling(nn.Module)`, built from the group (labels,
dims) at model construction; holds the cached intertwiners as buffers (cast to model dtype)
and the per-path `w` as one `(n_paths,)` parameter; forward takes `(mu, sigma)` and returns
the updated pair (sigma unchanged). Applied in `vfe3/model/stack.py` immediately after the
head mixer application, gated on its presence — model owns `self.cg_coupling` exactly as it
owns `self.head_mixer`.

Config: `use_head_mixer` becomes legal for `so_n`/`sp_n` (construction dispatches the
isotypic form; the existing equal-blocks groups are unaffected); new field
`use_cg_coupling: bool = False`, validated to require a group with `irrep_labels` (rejected
otherwise with a clear message). Both new parameter sets are documented neural-network
exceptions in the CLAUDE.md list (the family of `use_head_mixer` exception 2); the
optimizer's exact-coverage guard forces explicit param groups in `build_optimizer`
(mixer deltas at the existing mixer LR group; `w` likewise).

## Testing

Property tests, no goldens: build-time intertwiner residual asserted small in float64;
isotypic mixer commutes with random tower group elements at machine precision (the probe run
during this design becomes the test); equal-blocks groups byte-identical to the current
mixer (torch.equal on outputs); SO(3) selection rules (l1 (x) l1 reaches l0, l1, l2 and not
l3; l1 (x) l2 reaches l1, l2, l3); CG equivariance probe C(rho_a(g) x, rho_b(g) y) =
rho_c(g) C(x, y) for random g; step-0 byte-identity of the full model for both toggles;
end-to-end mixed-tower model with both enabled (finite loss, nonzero gradients on every
Delta_t and on w); model-level gauge-equivariance residual with both enabled against the
out-of-group control. Existing suites must pass unchanged.

## Guards and failure modes

`use_cg_coupling` without labels: config ValueError. Mixer on unlabeled unequal dims:
unchanged ValueError. CG solve too large: ValueError with the cost guard message. Degenerate
path set (no admissible triples -- rare, since self-products l (x) l -> l are usually admissible; reachable only for towers whose products all land outside the spec's labels): `CGCoupling` construction raises
with the explanation that there is nothing to couple (prefer a loud config error over a
silent no-op module). All asserts fire at model construction, never mid-step.

## Phase 2b sketch (deferred)

The coupling as a free-energy term: a trilinear interaction such as
F += - sum_p w_p E_q[ < mu^(c), C_p(mu^(a), mu^(b)) > ] (or a KL toward the CG prediction),
descended by the E-step, with sigma entering through the expectation. This changes the
variational problem: beta stationarity, the analytic kernel, and the oracle all gain a term,
and the manuscripts' canonical F gains an interaction block — which is why it is deferred to
its own spec, plan, and manuscript note. Nothing in phases 1/2a forecloses it: the seam
(per-type structure on the group, cached intertwiners) is exactly what that term consumes.

## Out of scope

Cross-type LINEAR mixing (zero by Schur — permanently out, not deferred); extending
`cross_couplings` to towers (dissolves the structure group, see Motivation); gauging the
commutant (adding A (x) I directions to the algebra for per-token mixing — a coherent
alternative design, not taken here); Young-tableau irrep families (future buildout; the CG
solver and label-grouped mixer are already registry-shaped for them).
