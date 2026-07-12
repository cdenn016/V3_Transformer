# UMAP figure improvement investigation — 2026-07-11

Scope: the five UMAP figures emitted per run (`belief_umap_{mu,sigma,phi}`, `model_umap_{mu,sigma}`,
all rendered by `plot_belief_umap` in `vfe3/viz/figures.py`) plus their companion
`belief_category_separation` and the supporting pipeline (`umap_embed`, `_belief_channel_features`,
`_cluster_embedding`, `_cluster_lift_labels`, `vfe3/viz/report.py`, `vfe3/viz/extract.py`).

Method: a 21-agent workflow — five parallel expert critics (visual design against the actual rendered
PNGs from recent runs, UMAP methodology, matplotlib label-placement engineering, data-flow/cost
analysis, external best-practices with sources), a merge/prioritization pass (48 raw findings
deduplicated to 15), and one adversarial feasibility verifier per finding. Verifiers checked every
claim against the actual code and the installed environment (umap-learn 0.5.12, scikit-learn 1.8.0,
scipy 1.17.1) and against the repo's hard constraints: the umap-learn subprocess isolation must stay
(Windows numba access violation), seeded determinism must be preserved, figures stay config-driven
and best-effort, and no new dependencies without justification. All 15 findings were confirmed
feasible; several recommendations were materially corrected during verification (corrections noted
inline). Nothing has been implemented — this document is the investigation deliverable.

## High impact

### 1. HDBSCAN's parameter regime produces the one-giant-blob-plus-islands failure

`_cluster_embedding` (figures.py:1630) runs `HDBSCAN(min_cluster_size=max(20, M//60), min_samples=5)`
with the default `eom` selection. At the production bank (M = 256 seqs x 128 tokens = 32,768) the
size floor is ~546 while `min_samples` stays 5, so the density estimate is far noisier than the floor
implies, and `eom` prefers the root-level blob on a diffuse-core-plus-islands cloud — exactly the
rendered failure (belief_umap_mu: one blue mass holding most points plus tiny islands). The linear
`M//60` scaling also makes the 64-sequence finalize bank (floor 136) and the 256-sequence
make_figures bank (floor 546) incomparable in granularity. Meanwhile `umap_embed` fixes
`n_neighbors=15` — 0.05% of a 32k cloud, an extremely local view.

Verified fix (all inside the existing try block so the KMeans fallback still catches failures):
`cluster_selection_method='leaf'` plus a scale-relative
`cluster_selection_epsilon = 0.02 * np.ptp(coords, axis=0).max()` (an absolute epsilon is wrong —
UMAP extents vary per channel/run); `min_cluster_size = max(20, int(np.sqrt(M)))` (181 at 32k, 90 at
8k — comparable granularity across bank sizes); `min_samples = max(10, min_cluster_size//20)`.
Optionally a depth-1 giant-cluster guard: if the largest cluster holds >50% of non-noise points,
re-run HDBSCAN once on its members and accept the split only if it yields >=2 sub-clusters each above
the size floor. In `plot_belief_umap`, scale `n_neighbors = int(np.clip(round(sqrt(M)/4), 15, 100))`
(45 at 32k); raise the subprocess timeout (figures.py:133) 600 -> 1200 in the same edit since this
roughly triples the kNN/SGD cost. sklearn 1.8.0's `cluster.HDBSCAN` accepts both new kwargs (verified
by signature); HDBSCAN has no RNG, so determinism is unaffected. Caveats: `leaf` raises the reported
noise fraction, so regenerated figures are not comparable to old ones; the causal mechanism was
verified as plausible from code, not reproduced — eyeball one production bank before/after.

### 2. Clustering is computed on the 2-D display embedding (the documented UMAP anti-pattern)

The worker hardcodes `n_components=2` with display-tuned `min_dist=0.1` (figures.py:90), and HDBSCAN
runs directly on those display coordinates (figures.py:1630). The umap-learn documentation's own
clustering guidance is to cluster a higher-dimensional `min_dist=0` embedding and reserve the 2-D run
for display: 2-D UMAP tears and compresses distances, and the figure then semantically annotates
those artifacts as "data-driven clusters."

Verified fix: add a keyword-only `n_components` (default 2) to `umap_embed` and one argv slot to the
worker; clamp it to `min(n_components, X.shape[1], X.shape[0]-1)` — required, because PCA init raises
when `n_components` exceeds the feature dim, which WILL happen on the phi channel for small algebras
and in every tiny-K CPU test; also change the collapsed-channel guard (figures.py:117-118) to
`zeros((N, n_components))`. In `plot_belief_umap`, keep the 2-D run for display and add a second
seeded run (`n_components=10`, `min_dist=0.0`, `n_neighbors=30`) of the same features for
`_cluster_embedding`, wrapped in try/except falling back to clustering the 2-D coords. When the
clamped `n_components` >= the native feature dim (phi with small n_gen, tiny tests), cluster the
native `_belief_channel_features` directly instead. Both embeddings can run in one worker invocation
to halve spawn overhead. Amend the caption to state clusters live in a separate clustering-space
embedding, so a cluster may legitimately render as multiple 2-D islands. Determinism verified:
`init='pca'` seeds both the PCA and the jitter in umap-learn 0.5.12.

### 3. Dominant-cluster lift labels are deterministic junk (ū, ō, ī) — and the tie-break is the actual culprit

For the giant cluster every common token has lift ~1, so the winners are count-2 rarities. The
verifier strengthened the diagnosis: every cluster-exclusive token ties at exactly lift = M/n_c, and
`scored.sort(reverse=True)` (figures.py:1673) breaks those ties by decoded string DESCENDING —
deterministically promoting high-codepoint accented glyphs. Any fix must change the tie-break, not
just the score. No label carries cluster size, so nothing conveys that the blue blob is ~30x larger
than the islands.

Verified fix, in `_cluster_lift_labels`: rank by smoothed log-odds
`log((ct+a)/(n_c+a*V)) - log((glob-ct+a)/((M-n_c)+a*V))` with `a=0.5`, `V` = observed token types;
replace the reverse tuple sort with an explicit key (score desc, global count desc, forward
lexicographic); in `plot_belief_umap`, when `n_c/M > 0.25` or the top raw lift < 1.5, label the
cluster "mixed core — n={n_c:,} ({share:.0%})" instead of token junk; append `n={n_c:,} ({share:.0%})`
as a second line inside every annotation box (second line, not same-line, so the fixed margin padding
does not clip widened boxes). Expose stats via a default-off kwarg so the existing return type and
its unit test stay intact.

### 4. The token bank is an unshuffled validation-split prefix

`report._build_loader` builds `shuffle=False` and `_collect_token_batches` takes the FIRST n batches
(report.py:75-97), so the 256-sequence bank is literally the first ~32,769 contiguous tokens of the
split. On wikitext-103 validation (~60 concatenated articles, ~1,914 windows) that covers roughly the
first dozen articles — every UMAP figure, lift label, and silhouette is computed on a topically
correlated prefix. The finalize auto-run path is worse (64-sequence TEST prefix).

Verified fix, deterministic with no RNG: stride the selection across the whole split. In
`_build_loader`, pass `make_dataloader`'s existing `stride` parameter computed from the cached
split's token count so ~256 windows space evenly over the split (window 0 stays first, so the
single-sequence figures stay byte-identical); in `_collect_token_batches`, when the caller supplies
its own loader (the finalize path), keep every `len(loader)//n_batches`-th batch. Flag when
implementing: ALL corpus-bank figures (UMAPs, silhouettes, vocab calibration/confusion, sigma-CE
join) change relative to previously generated runs, since `belief_ce_bank` and
`vocab_prediction_stats` consume the same selection.

### 5. Margin-callout labels: full-span linspace produces whole-plot leader spaghetti

`xs = np.linspace(xmin, xmax, len(row))` (figures.py:1774,1780) spaces labels along each margin with
no regard to cluster position, so a cluster at x~xmax can get its label at x~xmin; leaders cross ~60%
of the canvas, starburst on centrally huddled clusters (belief_umap_phi), and cross each other
(belief_umap_sigma). Association then rests on box-edge color alone, ambiguous under tab20's hue
pairs.

Verified fix — the badge+legend redesign (two critics converged on it independently): numbered
circular badges at the cluster anchors (1..N in the existing descending-size order, already
deterministic), a compact legend band outside the axes with one row per cluster (swatch, number, top
tokens, n, share), and deletion of the whole margin machinery (sides dict, `_callout` loops, the
±30%/±15% padding). Concretizations from verification: de-overlap badges with a fixed-order greedy
nudge (NOT adjustText — new dependency and nondeterministic); replace `fig.tight_layout()` with
`tight_layout(rect=...)` or `subplots_adjust(right=...)` because plain tight_layout ignores
`fig.text` and lets the axes overlap the legend band; `savefig.bbox='tight'` (already set by
`set_publication_style`) keeps `fig.text` in the PNG. One change fixes all five figures.

### 6. Cluster anchor is nearest-to-MEAN, not a medoid — stars land in empty space

`md = pts[np.argmin(((pts - pts.mean(0))**2).sum(1))]` (figures.py:1753). For crescent or
multi-island clusters the mean falls off-support and the "anchor" is a sparse straggler between
lobes (verified against rendered PNGs: several stars sit in inter-island gaps). The star is also
filled with the cluster's own color, so it vanishes inside the saturated dominant blob.

Verified fix: a density-peak member anchor — bin each cluster's members onto a fixed ~200x200 grid
over the global extent, smooth with `scipy.ndimage.gaussian_filter(sigma=2)`, score each MEMBER by
the smoothed density at its own bin, take the argmax (first-index rule = deterministic tie-break).
Do not select "the member in the densest bin" — after smoothing the argmax bin can be empty of
members. Keep the cluster-color fill + black edge (it is the visual key to the label) and add a white
halo via `path_effects=[patheffects.withStroke(linewidth=3, foreground='white')]`. Fall back to the
current expression in try/except. Also fix the false "medoid" wording in the comment and docstring.

## Medium impact

### 7. tab20 in index order gives adjacent clusters near-identical hues

tab20 is ordered as dark/pastel pairs of the same hue, so size-ranked neighbors differ mainly in
lightness (the dominant blue vs the pastel-blue neighbor in belief_umap_mu). Verified one-liner:
reindex the palette so the ten saturated distinct hues come first and both grays last (grays are
confusable with the #bbbbbb noise layer):
`palette = plt.cm.tab20(np.linspace(0, 1, 20))[[0,2,4,6,8,10,12,16,18,1,3,5,7,9,11,13,17,19,14,15]]`.
Defer the greedy spatial hue-separation upgrade unless figures still show confusable neighbors.

### 8. KDE underlay is dishonest and expensive; fixed point size/alpha saturates the core

`gaussian_kde` over all 32,768 coords on a 120x120 grid (~4.7e8 kernel evaluations, measured ~6 s per
figure) whose lowest contourf level paints the whole bbox gray — and standard UMAP (min_dist=0.1)
does not preserve density, so the underlay visualizes embedding artifacts. Verified fix: line
contours over the points (`ax.contour(..., colors='0.35', linewidths=0.5)` — note `linewidths`;
contour silently ignores `lw`) or an honest histogram2d+gaussian_filter count underlay; if a KDE
stays, fit it on a seeded <=8k subsample (measured 6.1 s -> 1.7 s). Scale per-cluster point size and
alpha by cluster population (`s = clip(9 - 1.2*log10(n_c), 2.5, 8)`, `alpha = clip(2500/n_c, 0.10,
0.75)`). Optionally expose umap-learn's `densmap=True` (the density-faithful variant, Narayan/Berger/
Cho 2021, already inside umap-learn 0.5.12) as a default-OFF worker flag. Avoid
`set_aspect('equal', adjustable='datalim')` — it recomputes limits at draw time and breaks the
callout geometry.

### 9. No parameters anywhere on the figure, while a methodology caveat squats in the title

Neither the UMAP settings, nor M, nor the per-channel metric (mu Euclidean / sigma log-Euclidean vech
/ phi gauge coords), nor the clustering method appear on the PNG — and the silent KMeans fallback
means "(0% noise)" is ambiguous between algorithms. Verified fix: shorten the title to one line;
demote the function/content-silhouette sentence to a fontsize-7 gray `fig.text` footnote; add a
right-aligned parameter footer (UMAP params from module-level constants shared with `umap_embed`'s
defaults — they are not otherwise in `plot_belief_umap`'s scope; M; seqs; per-channel metric;
clustering method). Have `_cluster_embedding` return `(labels, method_desc)` — print `eom` (or
whatever is then configured), NOT "leaf" as originally proposed, and update its only other caller in
tests/test_viz.py. Narrow the fallback catch to `ImportError` so genuine HDBSCAN runtime errors
surface through `_emit`'s existing best-effort log instead of silently switching algorithms.

### 10. No on-figure key for gray noise points or the star markers

Nothing on the PNG says gray = unclustered or star = anchor. Verified fix: proxy legend handles
(`unclustered ({noise:.0%})` gated on `nm.any()` — the KMeans fallback emits no noise, and a
fabricated 0% entry would mislead — plus a star handle), `ax.legend(loc='lower left', fontsize=7,
frameon=False)`. The proposed minimum-share floor on callouts was REJECTED by verification as dead
code: `min_cluster_size = max(20, M//60)` already guarantees every cluster >= ~1.67% of M once
M >= 1200. The real "leader ink exceeds cluster ink" problem is spatial diffuseness — if addressed,
gate callouts on compactness (RMS distance from anchor vs plot diagonal), a separate deliberate
change.

### 11. Token labels destroy BPE word-boundary information and render punctuation as noise

`_cluster_lift_labels` strips whitespace (figures.py:1669), discarding GPT-2's leading-space
word-start marker, so "omach" (stomach fragment) and genuine words are indistinguishable, and boxed
bare glyphs ("=", ",") read as noise. Verified fix: repr-quote punctuation-only tokens (classified
via `not any(ch.isalnum() ...)` — NOT `isalnum()`, which would quote "don't"); prefix continuation
subwords with a marker glyph (e.g. `·`) only for letter-bearing tokens; reuse `_bpe_category`'s
existing branches so boundary semantics live in one place; note the convention in the caption; update
the pinned expectations in tests/test_viz.py and tests/test_round3_viz.py.

### 12. scikit-learn and scipy are load-bearing but undeclared

`pyproject.toml`'s viz extra lists only umap-learn; sklearn (HDBSCAN, silhouette) and scipy (KDE,
curve_fit) arrive transitively. Verified fix: `viz = ["umap-learn", "scikit-learn>=1.3", "scipy"]`.
Modest urgency (umap-learn 0.5.12 already pins sklearn>=1.6) but it removes the silent
HDBSCAN-vs-KMeans divergence on older environments. Keep the runtime fallbacks.

### 13. Five full-M UMAP subprocesses per figure set, plus duplicated feature builds

Each of the five figures spawns a fresh subprocess (numba JIT warmup paid 5x) on the full bank, and
`plot_belief_category_separation` rebuilds all three channel feature matrices PER TAXONOMY (6 builds —
worse than the critic claimed; the full-sigma build is 32,768 KxK eigendecompositions). The
`glob` token-count dict (figures.py:1657) is one full-array pass per unique token. Verified fixes:
batch all channel embeddings through ONE worker invocation (loop over argv tuples; fresh
`umap.UMAP` per pair is numerically identical to separate processes — pin with a tiny-K equality
test; worker saves each out.npy as it completes so one mid-batch crash degrades per-channel, not
all-figures); compute `_belief_channel_features` once per bank and pass via optional
`features=None`/`coords=None` kwargs (standalone test callers keep working); hoist the per-taxonomy
rebuild; replace the glob dict with `np.unique(ids, return_counts=True)`.

### 14. The most informative overlays are free but unused

`belief_ce_bank` computes aligned per-token CE/confidence/tr(Sigma) over the same unshuffled loader
in the same pass, `seq_idx` is banked but never read by any figure, position is a `arange` away, and
sigma is in the bank while the mu UMAP renders uncolored. Verified plan: an opt-in `with_ce` bank
column behind the existing 8 GB full-vocab guard (do NOT row-join `belief_ce_bank` output — its batch
coverage diverges from `belief_bank`'s when `batch_size >= max_sequences`); bank `pos_idx`; compute
tr(Sigma) at figure time from the banked sigma; render as a small-multiples companion figure per
channel (CE percentile-rank, position, log tr(Sigma)) from a single embedding. Layer-0 (stack-input)
beliefs are already materialized and discarded in `belief_bank`'s loop — banking them enables a cheap
before/after-inference UMAP pair (label the panel "stack input"; under `s_e_step` it is the s-refined
anchor, not the raw encode).

### 15. No embedding-faithfulness or stability evidence

One seed, no trustworthiness score, no cluster-agreement check — yet the narrative ("N data-driven
clusters") rests on one realization. Verified fix: a best-effort `sklearn.manifold.trustworthiness`
annotation on a seeded <=3000-point subsample (mandatory cap — full M materializes two ~8.6 GB (M,M)
arrays; clamp `n_neighbors` for tiny test banks; label it "(3k subsample)"). Optionally, behind a
default-OFF config-plumbed toggle: embed at seeds {0,1,2}, cluster each, report mean pairwise
adjusted Rand computed on points non-noise in BOTH labelings of each pair (raw ARI lets the noise
mass dominate); below ~0.5, retitle to "candidate clusters."

## Suggested implementation order

Batch 1 (one session, all small, transforms the figures): 1 (HDBSCAN regime + n_neighbors scaling),
3 (lift tie-break + mixed-core gate + cluster sizes), 6 (density-peak anchor + halo), 7 (palette
reindex), 10 (legend key), 9 (parameter footer + title cleanup). Batch 2: 5 (badge+legend layout —
the single biggest legibility win, medium effort), 8 (underlay + size/alpha scaling), 11 (BPE-aware
labels), 12 (pyproject extra). Batch 3 (methodology): 2 (clustering-space embedding), 4 (strided
bank — regenerates all corpus-bank figures), 13 (batched subprocess + feature reuse), 15
(trustworthiness). Batch 4 (new capability): 14 (CE/position/tr-Sigma overlays, before/after pair).

Findings 1, 3, 6, 7, 9, 10 touch only `_cluster_embedding` / `_cluster_lift_labels` /
`plot_belief_umap` and compose without conflict. Finding 5 rewrites the layout that 6/9/10 decorate,
so implement 5 first if both batches land together. Figures regenerated after 1, 2, or 4 are not
comparable to previously generated PNGs (different clustering granularity, noise fraction, and bank
selection) — regenerate any run being actively compared.
