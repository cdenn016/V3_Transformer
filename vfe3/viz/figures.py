r"""Publication-quality figures for VFE_3.0 diagnostics (matplotlib; UMAP / networkx / sklearn).

Figure generators over beliefs (means / gauge frames), attention, covariance, and training
trajectories. Each returns a matplotlib Figure and optionally saves it; colourblind-safe
palette and journal-ish defaults via ``set_publication_style``. The heavier dependencies
(UMAP, networkx, scikit-learn) are imported lazily inside the function that needs them, so
the module imports even where one is absent (the function raises a clear message instead).
Tensors are accepted as torch or numpy; everything is detached to numpy for plotting.

A registry (``register_figure``) lets a new figure slot in by name.
"""

from typing import Callable, Dict, Optional

import matplotlib

matplotlib.use("Agg")                                            # non-interactive (headless / tests)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Wong colourblind-safe qualitative palette (used module-wide, incl. the trajectory defaults).
_CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def _np(x) -> np.ndarray:
    """Detach a torch tensor (or pass an array) to a contiguous numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def set_publication_style() -> None:
    """Colourblind-safe palette + journal-ish matplotlib defaults (call once)."""
    try:
        import seaborn as sns
        sns.set_palette("colorblind")
    except ImportError:
        pass
    plt.rcParams.update({
        "figure.dpi":      200,
        "savefig.dpi":     300,
        "savefig.bbox":    "tight",
        "font.size":       10,
        "axes.titlesize":  11,
        "axes.labelsize":  10,
        "axes.grid":       True,
        "grid.alpha":      0.25,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _save(fig, path: Optional[str]):
    if path is not None:
        fig.savefig(path)
    return fig


def umap_embed(
    features,                            # (N, D) tensor/array

    *,
    n_neighbors: int = 15,
    min_dist:    float = 0.1,
    seed:        int = 0,
):
    """2-D UMAP embedding of ``features`` ((N, D) -> (N, 2)). Lazy-imports umap-learn."""
    try:
        import umap
    except ImportError as exc:           # pragma: no cover
        raise ImportError("umap_embed needs umap-learn (pip install umap-learn)") from exc
    X = _np(features)
    # A fully collapsed channel (every point identical -> zero variance) has no embedding, and PCA
    # init would divide by total variance 0 and yield NaN. Return a trivial finite layout so the
    # downstream clustering / KDE stay valid (faithful: constant features carry no 2-D structure).
    if X.shape[0] < 3 or float(np.ptp(X, axis=0).max()) <= 0.0:
        return np.zeros((X.shape[0], 2), dtype=float)
    n_neighbors = min(n_neighbors, max(2, X.shape[0] - 1))
    # init="pca" skips UMAP's spectral eigensolver, which on disconnected / near-degenerate belief
    # clouds fails to converge (tiny eigengap) and silently falls back to uniform-random init; PCA is
    # deterministic and data-aware. n_jobs=1 matches what random_state already forces, so it also
    # silences UMAP's "n_jobs overridden" warning. Same compute, no warning cascade.
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2,
                        init="pca", random_state=seed, n_jobs=1)
    return reducer.fit_transform(X)


def plot_embedding(
    coords,                              # (N, 2) 2-D coordinates
    labels=None,                         # (N,) optional integer/float labels for colour

    *,
    title: str = "Belief embedding",
    path:  Optional[str] = None,
):
    """Scatter a 2-D embedding, coloured by ``labels``; returns (and optionally saves) the Figure."""
    c = _np(coords)
    fig, ax = plt.subplots(figsize=(5, 4))
    kw = {} if labels is None else {"c": _np(labels), "cmap": "viridis"}
    sc = ax.scatter(c[:, 0], c[:, 1], s=18, alpha=0.85, **kw)
    if labels is not None:
        fig.colorbar(sc, ax=ax, shrink=0.8, label="label")
    ax.set(title=title, xlabel="dim 1", ylabel="dim 2")
    return _save(fig, path)


def clustering_metrics(
    features,                            # (N, D)
    labels,                              # (N,) cluster/class labels

    *,
    sample_size: Optional[int] = None,   # cap silhouette's O(N^2) pairwise pass (None = full)
    seed:        int           = 0,
) -> Dict[str, float]:
    """Unsupervised cluster quality of ``features`` under ``labels``: silhouette + CH index.

    ``sample_size`` subsamples the silhouette computation (its pairwise distance is O(N^2)), keeping
    it fast when many points / many label sets are scored; CH is O(N) so it always uses all points."""
    from sklearn.metrics import calinski_harabasz_score, silhouette_score
    X = _np(features); y = _np(labels)
    if len(set(y.tolist())) < 2:
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan")}
    ss = sample_size if (sample_size is not None and sample_size < X.shape[0]) else None
    return {
        "silhouette":        float(silhouette_score(X, y, sample_size=ss, random_state=seed)),
        "calinski_harabasz": float(calinski_harabasz_score(X, y)),
    }


def attention_graph(
    beta,                                # (N, N) attention weights

    *,
    threshold: float = 0.05,
):
    """Build a weighted directed graph from the attention matrix (edges above ``threshold``)."""
    import networkx as nx
    B = _np(beta)
    N = B.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and B[i, j] > threshold:
                G.add_edge(i, j, weight=float(B[i, j]))
    return G


def plot_attention_graph(
    beta,                                # (N, N)

    *,
    threshold: float = 0.05,
    path:      Optional[str] = None,
):
    """Draw the attention graph (spring layout, edge width ~ weight)."""
    import networkx as nx
    G = attention_graph(beta, threshold=threshold)
    fig, ax = plt.subplots(figsize=(5, 5))
    pos = nx.spring_layout(G, seed=0)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=160, node_color="#4C72B0")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white")
    if weights:
        nx.draw_networkx_edges(G, pos, ax=ax, width=[2.5 * w for w in weights],
                               edge_color="#555555", alpha=0.6, arrowsize=8)
    ax.set_title("Attention graph"); ax.axis("off")
    return _save(fig, path)


def _attn_log_bounds(
    M:     np.ndarray,                   # attention weights (any shape)
    vmin:  Optional[float] = None,
    vmax:  Optional[float] = None,

    *,
    floor: float           = 1e-4,
) -> tuple:                              # (vmin, vmax) valid for a LogNorm scale
    r"""Positive-entry (vmin, vmax) for a log attention scale (scale bottoms out at ``floor`` = 1e-4).

    Causal-masked future positions are exact zeros (softmax over a -inf prior), so only the
    active (positive) entries set the scale. ``vmin`` is floored at ``floor`` (default 1e-4) so the
    log scale resolves attention weights down to 1e-4 while a long causal-softmax tail of near-zero
    weights cannot wash a panel out, and kept strictly below ``vmax`` so a uniform map stays a valid
    LogNorm. Pass both bounds to share one scale across several panels.
    """
    if vmax is None:
        pos = M[M > 0]
        vmax = float(pos.max()) if pos.size else 1.0
    if vmin is None:
        pos = M[M > 0]
        vmin = float(pos.min()) if pos.size else vmax * floor
    vmin = max(vmin, floor)                                       # bottom the log scale at 1e-4
    if vmin >= vmax:                                             # degenerate / uniform map
        vmin = vmax * floor
    return float(vmin), float(vmax)


def _attn_imshow(ax, B: np.ndarray, *, vmin: float, vmax: float, log: bool = True, cmap: str = "magma"):
    r"""imshow one (N, N) attention map. ``log`` (default) uses ``LogNorm`` to resolve the peaky
    off-diagonal structure a linear scale washes to black; exact-zero (causal-masked) entries are
    non-positive, so ``LogNorm`` masks them and ``set_bad`` renders them black. ``cmap`` selects the
    colour family so the channels read apart: belief beta = 'magma' (warm), model gamma = 'viridis' (cool)."""
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")
    if log:
        return ax.imshow(B, cmap=cmap_obj, aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax))
    return ax.imshow(B, cmap=cmap_obj, aspect="auto", vmin=0.0, vmax=vmax)


def plot_attention_heatmap(
    beta,                                # (N, N)

    *,
    log:    bool            = True,
    title:  str             = "Attention",
    cmap:   str             = "magma",   # 'magma' belief beta, 'viridis' model gamma
    symbol: str             = r"\beta",  # colorbar math symbol (\beta belief, \gamma model)
    vmin:   Optional[float] = None,
    vmax:   Optional[float] = None,
    path:   Optional[str]   = None,
):
    r"""Log-scaled heatmap of one attention map (rows = queries i, cols = keys j).

    Attention is a peaky causal softmax (most mass on a few keys, exact zeros above the diagonal),
    so the default ``log`` scale (matplotlib ``LogNorm`` on beta) resolves the off-diagonal
    structure a linear scale collapses to black; the causal-masked zeros render as the 'bad'
    colour. Pass shared ``vmin``/``vmax`` to make several panels comparable; otherwise the positive
    entries set the scale (log floor at 1e-4). ``cmap``/``symbol`` select the
    channel identity: belief beta ('magma', \beta) vs model gamma ('viridis', \gamma).
    """
    B = _np(beta)
    vlo, vhi = _attn_log_bounds(B, vmin, vmax)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = _attn_imshow(ax, B, vmin=vlo, vmax=vhi, log=log, cmap=cmap)
    label = rf"${symbol}_{{ij}}$ (log scale)" if log else rf"${symbol}_{{ij}}$"
    fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    ax.set(title=title, xlabel="key j", ylabel="query i")
    return _save(fig, path)


def plot_attention_grid(
    maps,                                # (L, H, N, N) per-layer per-head attention (or (H,N,N) / (N,N))

    *,
    log:   bool          = True,
    title: str           = "Attention",
    path:  Optional[str] = None,
):
    """Grid of attention heatmaps: rows = layers, cols = heads (rows query i, cols key j).

    Accepts a per-layer/per-head stack ``(L, H, N, N)`` (as :meth:`VFEModel.attention_maps`
    returns), a single layer ``(H, N, N)``, or a single map ``(N, N)``. A shared LOG colour scale
    across all panels (default ``log``) makes heads/layers comparable and resolves the peaky
    off-diagonal structure; ``squeeze=False`` keeps the L==1 / H==1 axes array 2-D so indexing is
    uniform. For one figure per (layer, head) instead of a grid, call :func:`plot_attention_heatmap`
    per panel (as :meth:`RunArtifacts.save_attention_maps` does).
    """
    M = _np(maps)
    if M.ndim == 2:                      # (N, N) -> one layer, one head
        M = M[None, None]
    elif M.ndim == 3:                    # (H, N, N) -> one layer
        M = M[None]
    L, H = M.shape[0], M.shape[1]
    fig, axes = plt.subplots(L, H, figsize=(2.6 * H + 1.0, 2.6 * L + 0.6), squeeze=False)
    vlo, vhi = _attn_log_bounds(M)                                # one scale shared across panels
    im = None
    for li in range(L):
        for hi in range(H):
            ax = axes[li][hi]
            im = _attn_imshow(ax, M[li, hi], vmin=vlo, vmax=vhi, log=log)
            ax.set_xticks([]); ax.set_yticks([])
            if li == 0:
                ax.set_title(f"head {hi}")
            if hi == 0:
                ax.set_ylabel(f"layer {li}\nquery $i$")
            if li == L - 1:
                ax.set_xlabel("key $j$")
    if im is not None:
        label = r"$\beta_{ij}$ (log scale)" if log else r"$\beta_{ij}$"
        fig.colorbar(im, ax=list(axes.ravel()), shrink=0.85, label=label)
    fig.suptitle(title)
    return _save(fig, path)


def plot_covariance_ellipses(
    mu,                                  # (N, K) belief means
    sigma,                               # (N, K) diagonal variances

    *,
    dims:  tuple = (0, 1),
    path:  Optional[str] = None,
):
    """Plot 1-sigma Gaussian ellipses for two belief coordinates (diagonal beliefs)."""
    from matplotlib.patches import Ellipse
    m = _np(mu); s = _np(sigma)
    a, b = dims
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for i in range(m.shape[0]):
        e = Ellipse((m[i, a], m[i, b]),
                    width=2 * np.sqrt(s[i, a]), height=2 * np.sqrt(s[i, b]),
                    alpha=0.3, facecolor="#4C72B0", edgecolor="#26456E")
        ax.add_patch(e)
    ax.scatter(m[:, a], m[:, b], s=10, color="#26456E")
    ax.set(title="Belief covariance ellipses", xlabel=f"dim {a}", ylabel=f"dim {b}")
    ax.autoscale_view()
    return _save(fig, path)


def _rolling_mean(v: np.ndarray, window: int) -> np.ndarray:
    r"""Centered moving average of ``v`` with edge-normalized windows (no zero-padding bias)."""
    if window <= 1 or v.size <= 2:
        return v
    window = int(min(window, v.size))
    kernel = np.ones(window)
    # NaN-aware: average only the FINITE taps in each window, so a single non-positive point (NaN'd
    # for a log-y axis) does not blank a whole window-width segment of the trend (audit r2 id13).
    # All-finite input -> mask all True -> byte-identical to the plain edge-normalized average.
    mask = np.isfinite(v)
    num = np.convolve(np.where(mask, v, 0.0), kernel, mode="same")
    den = np.convolve(mask.astype(v.dtype), kernel, mode="same")  # finite-tap count per position
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def _kstep(v, _pos=None) -> str:
    r"""Compact training-step tick label: 5000 -> '5k', 12500 -> '12.5k' (5-digit step labels
    overrun into an unreadable smear at the trajectory figure width otherwise)."""
    v = float(v)
    if abs(v) >= 1000.0:
        s = v / 1000.0
        return f"{s:.0f}k" if abs(s - round(s)) < 1e-9 else f"{s:.1f}k"
    return f"{v:.0f}"


def _step_xaxis(ax) -> None:
    r"""Uncrowded training-step x-axis: at most ~6 integer ticks rendered via :func:`_kstep`.
    Applied to every step-indexed trajectory so the default ~9 five-digit labels stop colliding."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(_kstep))


def plot_trajectory(
    values,                              # (T,) a scalar series (loss / free energy / diagnostic)
    steps          = None,               # (T,) x-coordinates (real training step); None -> sample index

    *,
    ylabel:         str           = "loss",
    title:          str           = "Training trajectory",
    color:          str           = _CB[0],
    logy:           bool          = False,
    annotate_final: bool          = False,
    median_line:    bool          = False,
    smooth:         int           = 0,
    annotate:       Optional[str] = None,   # 'min' | 'max' -> mark + label the running best point
    path:           Optional[str] = None,
):
    r"""Line plot of a scalar trajectory over the real training step.

    The x-axis is ``steps`` when given (else the sample index, so the axis is no longer mislabelled
    as a step). A long noisy series (pass ``smooth=w``) draws a faint raw line under a centered
    ``w``-point rolling mean and no per-point markers, which a thousand-point series otherwise
    collapses into a solid band; a short series (<=50 points) keeps markers. ``logy`` is for wide-range
    or heavy-tailed series (perplexity, holonomy). ``annotate='min'|'max'`` marks that extremum with
    its step, ``annotate_final`` tags the last value at the right edge, and ``median_line`` draws the
    series median as a dashed reference (the typical value on a heavy-tailed log axis where the
    spikes sit decades above the floor).
    """
    v = _np(values).reshape(-1).astype(float)
    x = np.arange(v.size, dtype=float) if steps is None else _np(steps).reshape(-1).astype(float)
    vplot = np.where(v > 0, v, np.nan) if logy else v            # log axis drops non-positive entries
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    if smooth and smooth > 1 and v.size > 50:
        ax.plot(x, vplot, lw=0.7, color=color, alpha=0.25)            # faint raw underlay
        ax.plot(x, _rolling_mean(vplot, smooth), lw=1.8, color=color) # rolling-mean trend
    elif v.size <= 50:
        ax.plot(x, vplot, marker="o", ms=3, lw=1.5, color=color)
    else:
        ax.plot(x, vplot, lw=1.2, color=color)
    if logy:
        ax.set_yscale("log")
    if median_line and v.size:
        med = float(np.median(v))
        ax.axhline(med, color="#444444", ls="--", lw=1)
        ax.annotate(f"median {med:.2g}", xy=(0.985, med), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7, color="#444444")
    if annotate in ("min", "max") and v.size:
        idx = int(np.argmin(v) if annotate == "min" else np.argmax(v))
        tag = "best" if annotate == "min" else "max"
        ax.scatter([x[idx]], [v[idx]], s=28, color="black", zorder=5)
        # Offset the callout INTO the plot, away from the edge the extremum sits against: a 'min'
        # hugs the floor -> label upward, a 'max' hugs the ceiling -> label downward, so the text
        # never overruns the title (the grad_norm.png 'max'-into-title collision).
        dy, va = (30, "bottom") if annotate == "min" else (-30, "top")
        ax.annotate(f"{tag} {v[idx]:.1f}\n@ step {int(x[idx]):,}",
                    xy=(x[idx], v[idx]), xytext=(-12, dy), textcoords="offset points",
                    fontsize=7.5, ha="right", va=va,
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
    if annotate_final and v.size:
        ax.annotate(f"{v[-1]:.4g}", xy=(x[-1], v[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, fontweight="bold", color=color)
    if steps is not None and v.size:
        ax.set_xlim(float(np.min(x)), float(np.max(x)))
        _step_xaxis(ax)
    ax.set(title=title, xlabel="training step", ylabel=ylabel)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Figure registry: name -> generator. New figures slot in by name.
# ---------------------------------------------------------------------------
_FIGURES: Dict[str, Callable] = {
    "embedding":           plot_embedding,
    "attention_graph":     plot_attention_graph,
    "attention_heatmap":   plot_attention_heatmap,
    "attention_grid":      plot_attention_grid,
    "covariance_ellipses": plot_covariance_ellipses,
    "trajectory":          plot_trajectory,
}


def register_figure(name: str) -> Callable:
    """Decorator registering a figure generator under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        _FIGURES[name] = fn
        return fn
    return _wrap


def get_figure(name: str) -> Callable:
    """Return the registered figure generator (KeyError if absent)."""
    if name not in _FIGURES:
        raise KeyError(f"no figure {name!r}; available: {sorted(_FIGURES)}")
    return _FIGURES[name]


# ===========================================================================
# Publication figures (the claim-linked set; see
# docs/superpowers/specs/2026-06-04-publication-figures-design.md). Each consumes
# the new vfe3.metrics measurements / vfe3.viz.extract runner outputs, builds a
# multi-panel composite, registers by name, and returns the Figure.
# ===========================================================================

_EPS_F32 = 1.1920929e-7                                           # float32 machine epsilon


def _median_band(
    ax,
    x:     np.ndarray,
    mat:   np.ndarray,                   # (T, M) -> median + 10/90 band over axis 1
    color: str,
    label: str,
    lo:    float = 10.0,
    hi:    float = 90.0,
):
    """Plot the median of ``mat`` over axis 1 with a 10-90 percentile band (single-seed spread)."""
    med = np.median(mat, axis=1)
    ax.plot(x, med, color=color, lw=1.8, label=label)
    ax.fill_between(x, np.percentile(mat, lo, axis=1), np.percentile(mat, hi, axis=1),
                    color=color, alpha=0.2)
    return med


def _ecdf(ax, x, color: str, label: str):
    """Plot the empirical CDF of ``x`` (clamped positive for a log axis)."""
    xs = np.sort(np.clip(_np(x).ravel(), 1e-12, None))
    if xs.size == 0:
        return
    ax.plot(xs, np.arange(1, xs.size + 1) / xs.size, color=color, lw=1.8, label=label)


def _belief_channel_features(bank: Dict, channel: str):
    r"""Map a belief channel to its UMAP feature space, faithful to its geometry.

    ``mu`` -> the means directly (Euclidean); ``phi`` -> the gauge coordinates directly;
    ``sigma`` -> the log-Euclidean chart: log-variances for a diagonal covariance, ``vech(log Sigma)``
    for a full one (so the Euclidean UMAP metric respects the SPD cone).
    """
    import torch
    if channel == "mu":
        return bank["mu"]
    if channel == "phi":
        return bank["phi"]
    if channel == "sigma":
        sig = bank["sigma"]
        if sig.dim() == 2:                                       # (M, K) diagonal -> log-variances
            return torch.log(sig.clamp(min=1e-12))
        sym = 0.5 * (sig + sig.transpose(-1, -2))               # (M, K, K) full -> vech(log Sigma)
        w, q = torch.linalg.eigh(sym)
        log_sigma = (q * torch.log(w.clamp(min=1e-12)).unsqueeze(-2)) @ q.transpose(-1, -2)
        K = log_sigma.shape[-1]
        iu = torch.triu_indices(K, K)
        vech = log_sigma[..., iu[0], iu[1]].clone()
        # Scale off-diagonals by sqrt(2) so the Euclidean vech metric equals the Frobenius / log-
        # Euclidean distance the UMAP embedding claims to respect (each off-diagonal appears twice in
        # the symmetric matrix; the diagonal once) (audit r2 id12).
        offdiag = iu[0] != iu[1]
        vech[..., offdiag] = vech[..., offdiag] * (2.0 ** 0.5)
        return vech
    raise ValueError(f"unknown belief channel {channel!r} (expected mu / sigma / phi)")


@register_figure("free_energy_descent")
def plot_free_energy_descent(
    history: Dict,                       # step, self_coupling, belief_coupling, attention_entropy, val_ce, [hyper_prior_weighted, gamma_*]

    *,
    lambda_beta:               'float | np.ndarray' = 1.0,
    lambda_gamma:              float = 0.0,
    include_attention_entropy: bool  = True,
    self_div:                  Optional[object] = None,   # (M,) converged self-divergences for the violin
    path:                      Optional[str]    = None,
):
    r"""F1: the per-token complexity free-energy stack over training plus the F-vs-CE co-descent.

    DESCRIPTIVE per-eval converged-belief snapshots from a representative batch (logged off the graph),
    all in NATS PER TOKEN so the terms are commensurate (the caller normalizes the per-sequence-sum
    diagnostics by the token count before logging). Panel A stacks the complexity / inference F that the
    E-step descends -- self-coupling, the lambda_beta-scaled belief-coupling and attention-entropy, and
    (when the model channel is live) the weighted hyper-prior and gamma model-coupling; the stack height
    is that F. The data/likelihood term (CE) is NOT in the stack: there is no observation channel in the
    LM, so the held-out CE is a readout, drawn separately. Panel B plots that SAME complexity F against
    the held-out val CE on a twin axis -- the co-descent, evidence that minimizing F tracks the loss.
    Panel C (when ``self_div`` is given) is the per-token self-divergence violin. ``lambda_beta`` accepts
    a per-row vector (the learned-coupling trajectory) as well as a scalar; ``total`` matches the
    free_energy_total column (the complexity F).
    """
    step, comps, total, ce = _fe_terms(history, lambda_beta, lambda_gamma=lambda_gamma,
                                       include_attention_entropy=include_attention_entropy)
    stack  = np.vstack([c for _, c in comps])
    labels = [_FE_LABELS[k][0] for k, _ in comps]
    ncol = 3 if self_div is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 3.6))
    axes[0].stackplot(step, stack, colors=_CB[:len(comps)], alpha=0.85, labels=labels)
    axes[0].set(xlabel="training step", ylabel="free energy (nats/token)", title="Complexity-F decomposition")
    axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    axes[1].plot(step, total, color=_CB[0], lw=2)                # complexity F (matches the panel-A stack height)
    axes[1].set(xlabel="training step", ylabel="F total (nats/token)", title="Co-descent (descriptive)")
    ax2 = axes[1].twinx()
    ax2.plot(step, ce, color=_CB[1], lw=2, ls="--")
    ax2.set_ylabel("val CE (nats)", color=_CB[1])
    if self_div is not None:
        axes[2].violinplot([_np(self_div).ravel()], showmeans=True)
        axes[2].set(xticks=[1], xticklabels=[r"$D(q_i\|p_i)$"], ylabel="nats",
                    title="Self-divergence (per token)")
    fig.tight_layout()
    return _save(fig, path)


# Canonical key -> (legend label, multi-line bar name) for the complexity-F components, in the
# order they stack. Belief channel always present; the model channel (hyper-prior, model-coupling)
# only when those columns are logged. The data term (CE) is NOT a component of F -- it is co-plotted
# as a held-out reference, never summed in.
_FE_LABELS = {
    "self":              ("self-coupling",     "self-coupling\n$\\mathrm{KL}(q\\|p)$"),
    "belief":            ("belief coupling",   "belief-coupling\n$\\lambda_\\beta\\sum\\mathrm{KL}(q\\|\\Omega q)$"),
    "attention_entropy": ("attention entropy", "attention-entropy\n$\\lambda_\\beta\\tau\\beta\\log(\\beta/\\pi)$"),
    "hyper_prior":       ("hyper-prior",       "hyper-prior\n$\\lambda_h\\mathrm{KL}(s\\|r)$"),
    "model_coupling":    ("model coupling",    "model-coupling\n$\\gamma[\\sum\\mathrm{KL}(s\\|\\Omega s)+\\mathrm{ent}]$"),
}


def _fe_terms(history: Dict, lambda_beta, *, lambda_gamma=0.0, include_attention_entropy=True) -> tuple:
    r"""Per-eval complexity free-energy components (nats/token) shared by the descent/decomposition/
    co-descent figures.

    Returns ``(step, components, total, ce)`` where ``components`` is an ordered list of
    ``(canonical_key, array)`` pairs whose sum is ``total`` -- the per-token COMPLEXITY / inference free
    energy that the E-step descends. ``total`` does NOT include the data/likelihood term: there is no
    observation channel in the LM, so the cross-entropy is a held-out readout, returned separately as
    ``ce`` for co-plotting, never summed into F. The belief blocks carry the (scalar or per-row)
    ``lambda_beta`` weight; the attention-entropy block is included only when
    ``include_attention_entropy`` (matching ``free_energy_terms``' gate on ``total``). The model channel
    is added only when its columns are present: ``hyper_prior_weighted`` is the EXACT weighted hyper-prior
    folded into the runtime total (so state_dependent/learnable lambda_h need no reconstruction), and the
    gamma blocks are scaled by ``lambda_gamma``. The logged diagnostics are already per token, so every
    component is commensurate and ``total`` matches the ``free_energy_total`` column.
    """
    step = _np(history["step"]).astype(float)
    lb   = _np(lambda_beta).astype(float)
    zeros = np.zeros(step.shape, dtype=float)
    comps = [("self",   _np(history["self_coupling"]).astype(float)),
             ("belief", lb * _np(history["belief_coupling"]).astype(float))]
    if include_attention_entropy:
        comps.append(("attention_entropy", lb * _np(history["attention_entropy"]).astype(float)))
    if "hyper_prior_weighted" in history:                            # exact weighted contribution to total
        comps.append(("hyper_prior", _np(history["hyper_prior_weighted"]).astype(float)))
    gkeys = [k for k in ("gamma_coupling", "gamma_meta_entropy") if k in history]
    if gkeys:                                                        # gamma block scaled like the belief block
        gc = float(lambda_gamma) * sum((_np(history[k]).astype(float) for k in gkeys), zeros.copy())
        comps.append(("model_coupling", gc))
    total = sum((c for _, c in comps), zeros.copy())
    ce = _np(history["val_ce"]).astype(float) if "val_ce" in history else np.full(step.shape, np.nan)
    return step, comps, total, ce


@register_figure("free_energy_codescent")
def plot_free_energy_codescent(
    history: Dict,                       # step, self_coupling, belief_coupling, attention_entropy, val_ce, [hyper_prior_weighted, gamma_*]

    *,
    lambda_beta:               'float | np.ndarray' = 1.0,
    lambda_gamma:              float = 0.0,
    include_attention_entropy: bool  = True,
    path:                      Optional[str] = None,
):
    r"""F-vs-CE co-descent: the per-token complexity free energy and the held-out loss fall together.

    Twin y-axes over the real training step -- the complexity / inference free energy F (self-coupling +
    the lambda_beta-scaled belief-coupling and attention-entropy + the weighted hyper-prior and gamma
    model-coupling when the model channel is live, left, solid) and the held-out validation CE (right,
    dashed) -- each a faint raw line under a rolling-mean trend. F is the quantity the E-step minimizes;
    the CE is NOT part of F (there is no observation channel in the LM), it is the held-out readout the
    descent is meant to track. The final values are tagged and the Pearson correlation of the two curves
    is in the title; a high positive r is the co-descent signature, the evidence that minimizing the
    inference F lowers held-out loss.
    """
    step, _comps, total, ce = _fe_terms(history, lambda_beta, lambda_gamma=lambda_gamma,
                                        include_attention_entropy=include_attention_entropy)
    keep = np.isfinite(total) & np.isfinite(ce)
    step, total, ce = step[keep], total[keep], ce[keep]
    w = max(5, total.size // 80)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(step, total, lw=0.7, color=_CB[0], alpha=0.25)
    # Final values fold into the legend labels; standalone right-edge annotations otherwise land on top
    # of the opposite (twin) axis tick labels -- the left-axis F tag overlaps the right CE ticks.
    flab = f"F total (left) = {total[-1]:.1f}" if step.size else "F total (left)"
    ln1, = ax.plot(step, _rolling_mean(total, w), lw=2.0, color=_CB[0], label=flab)
    ax.set(xlabel="training step", ylabel="free energy F (nats/token)")
    ax.yaxis.label.set_color(_CB[0]); ax.tick_params(axis="y", colors=_CB[0])
    if step.size:
        ax.set_xlim(float(step.min()), float(step.max()))
        _step_xaxis(ax)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)                         # set_publication_style hides it by default
    ax2.plot(step, ce, lw=0.7, color=_CB[1], alpha=0.25)
    clab = f"val CE (right) = {ce[-1]:.2f}" if step.size else "val CE (right)"
    ln2, = ax2.plot(step, _rolling_mean(ce, w), lw=2.0, color=_CB[1], ls="--", label=clab)
    ax2.set_ylabel("validation CE (nats/token)", color=_CB[1]); ax2.tick_params(axis="y", colors=_CB[1])
    if total.size > 2:
        r = float(np.corrcoef(total, ce)[0, 1])
        ax.set_title(rf"Free-energy / data-term co-descent (Pearson $r={r:+.2f}$)")
    else:
        ax.set_title("Free-energy / data-term co-descent")
    ax.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()], loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("free_energy_decomposition")
def plot_free_energy_decomposition(
    history: Dict,                       # step, self_coupling, belief_coupling, attention_entropy, val_ce, [hyper_prior_weighted, gamma_*]

    *,
    lambda_beta:               'float | np.ndarray' = 1.0,
    lambda_gamma:              float = 0.0,
    include_attention_entropy: bool  = True,
    path:                      Optional[str] = None,
):
    r"""The per-token complexity free-energy budget at convergence and how its terms move over training.

    Panel A: the F-contributions at the LAST eval on a LOG x-axis with the value past each bar, so the
    dominant self-coupling and the order-of-magnitude-smaller terms are all legible (a linear bar
    collapses the small ones to slivers, the failure mode of the old single bar). Panel B: the same terms
    at the early/mid/late thirds of training on a log y-axis -- self-coupling sits flat near its value
    while the belief-coupling, attention-entropy, and model-channel terms carry the descent. The bars are
    the complexity / inference F (the quantity the E-step minimizes): self-coupling, the lambda_beta-scaled
    belief-coupling and attention-entropy, and -- when the model channel is live -- the weighted hyper-prior
    and gamma model-coupling. The data/likelihood term (CE) is NOT shown: it is not part of F (no
    observation channel in the LM); see the co-descent figure for F vs the held-out CE.
    """
    step, comps, _total, _ce = _fe_terms(history, lambda_beta, lambda_gamma=lambda_gamma,
                                         include_attention_entropy=include_attention_entropy)
    names  = [_FE_LABELS[k][1] for k, _ in comps]
    labels = [_FE_LABELS[k][0] for k, _ in comps]
    series = [c for _, c in comps]
    colors = _CB[:len(series)]
    last = np.array([s[-1] for s in series])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    # Panel A -- convergence snapshot (largest at top), log x so every term reads despite the ~30x gap.
    y = np.arange(len(series))[::-1]
    axes[0].barh(y, last, color=colors, height=0.62)
    axes[0].set_xscale("log")
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=8)
    for yi, val in zip(y, last):
        axes[0].annotate(f"{val:.1f}", xy=(val, yi), xytext=(4, 0), textcoords="offset points",
                         va="center", ha="left", fontsize=9)
    axes[0].set_xlim(float(last.min()) * 0.5, float(last.max()) * 3.0)
    axes[0].set(xlabel="free-energy contribution (nats/token, log scale)",
                title=f"Budget at step {int(step[-1]):,}")
    # Panel B -- early/mid/late medians, log y so the flat dominant term and the moving terms coexist.
    thirds  = np.array_split(np.arange(step.size), 3)
    centers = np.arange(3)
    n_bars  = len(series)
    width   = min(0.2, 0.8 / max(n_bars, 1))                         # keep the grouped bars inside one tick
    off0    = (n_bars - 1) / 2.0                                     # center the group regardless of bar count
    for j, arr in enumerate(series):
        meds = [float(np.median(arr[idx])) if idx.size else np.nan for idx in thirds]   # skip empty thirds
        axes[1].bar(centers + (j - off0) * width, meds, width=width, color=colors[j], label=labels[j])
    axes[1].set_yscale("log")
    axes[1].set_ylim(top=axes[1].get_ylim()[1] * 2.2)            # headroom for the legend above the bars
    axes[1].set_xticks(centers); axes[1].set_xticklabels(["early", "mid", "late"])
    axes[1].set(xlabel="training third", ylabel="F term (median, nats/token, log scale)",
                title="Self-coupling flat; other terms descend")
    axes[1].legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("model_channel_terms")
def plot_model_channel_terms(
    history: Dict,                       # step + any of: hyper_prior, gamma_coupling, gamma_meta_entropy

    *,
    path:    Optional[str] = None,
):
    r"""The model-channel (s) free-energy blocks over training: the s-channel analogue of the belief
    decomposition.

    Plots whichever of the hyper-prior KL(s_i||r), the gamma model-coupling sum_j gamma_ij KL(s_i||Omega s_j),
    and its meta-entropy tau_g sum_j gamma_ij log(gamma_ij/pi^s_ij) are logged (per token, RAW/unweighted --
    the s-channel siblings of self-coupling / belief-coupling / attention-entropy). Each is a faint raw
    line under a rolling-mean trend on a symlog y (the blocks are non-negative and can sit near zero at
    init), with the final value tagged. These are the blocks the model channel descends -- under
    ``s_e_step`` inside the s E-step, otherwise as additive loss terms -- so a healthy channel pulls them
    down over training. A separate figure from the belief decomposition because the model channel is a
    distinct hierarchical tier (h -> s -> p -> q).
    """
    step = _np(history["step"]).astype(float)
    spec = [
        ("hyper_prior",        r"$\mathrm{KL}(s_i\|r)$ (hyper-prior)",                              _CB[0]),
        ("gamma_coupling",     r"$\sum_j\gamma_{ij}\mathrm{KL}(s_i\|\Omega s_j)$ (model-coupling)", _CB[1]),
        ("gamma_meta_entropy", r"$\tau_g\sum_j\gamma_{ij}\log(\gamma_{ij}/\pi^s)$ (meta-entropy)",  _CB[2]),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    w = max(5, step.size // 80)
    plotted = False
    for key, label, color in spec:
        if key not in history:
            continue
        v = _np(history[key]).astype(float)
        keep = np.isfinite(v)
        x, vv = step[keep], v[keep]
        if not vv.size:
            continue
        plotted = True
        ax.plot(x, vv, lw=0.7, color=color, alpha=0.25)
        # final value folded into the legend label (a separate right-edge tag collides with the
        # upper-right legend; the legend is the single, collision-free home for the value).
        ax.plot(x, _rolling_mean(vv, w), lw=2.0, color=color, label=f"{label}  ={vv[-1]:.3g}")
    if plotted:
        ax.set_yscale("symlog", linthresh=1e-4)                   # non-negative blocks, near-zero at init
        if step.size:
            ax.set_xlim(float(step.min()), float(step.max()))
            _step_xaxis(ax)
        ax.legend(fontsize=8, frameon=False, loc="best")
    ax.set(xlabel="training step", ylabel="model-channel F block (nats/token)",
           title="Model-channel free-energy blocks (raw)")
    fig.tight_layout()
    return _save(fig, path)


def _grad_norm_decomp_fig(
    history: Dict,
    spec:    list,                       # [(column_key, latex_label, color, role_name), ...]
    title:   str,                        # base title; a "(role / role)" suffix is appended from the
    #                                      components ACTUALLY plotted, so it never promises a role the
    #                                      figure does not show (e.g. phi when e_phi_lr=0)
    ylabel:  str,
    path:    Optional[str],
):
    r"""Shared body for the mu / sigma / phi gradient-norm decomposition figures (M-step and E-step).

    Each series is a faint raw line under a rolling-mean trend on a log y (norms are strictly positive
    and span orders of magnitude), with its final value folded into the legend label (a separate
    right-edge tag collides with the upper-right legend). A component that is identically zero / absent
    (a dead or off substep) drops out under the log-y positive mask rather than flatlining at 0, and its
    role is omitted from the title suffix.
    """
    step = _np(history["step"]).astype(float)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    w = max(5, step.size // 80)
    roles = []
    for key, label, color, role in spec:
        if key not in history:
            continue
        v = _np(history[key]).astype(float)
        keep = np.isfinite(v) & (v > 0)                           # log y: drop non-positive (dead/off-substep) points
        x, vv = step[keep], v[keep]
        if not vv.size:
            continue
        roles.append(role)
        ax.plot(x, vv, lw=0.7, color=color, alpha=0.25)
        ax.plot(x, _rolling_mean(vv, w), lw=2.0, color=color, label=f"{label}  ={vv[-1]:.3g}")
    if roles:
        ax.set_yscale("log")
        if step.size:
            ax.set_xlim(float(step.min()), float(step.max()))
            _step_xaxis(ax)
        ax.legend(fontsize=8, frameon=False, loc="best")
        title = f"{title} ({' / '.join(roles)})"
    ax.set(xlabel="training step", ylabel=ylabel, title=title)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("grad_norm_decomposition")
def plot_grad_norm_decomposition(
    history: Dict,                       # step + any of: grad_norm_mu, grad_norm_sigma, grad_norm_phi

    *,
    path:    Optional[str] = None,
):
    r"""M-STEP per-role pre-clip parameter-gradient L2 norms over training: the mu / sigma / phi
    decomposition of the aggregate ``grad_norm`` curve.

    Each optimizer group is tagged with a role in {mu, sigma, phi} (build_optimizer); the logged
    norm aggregates the pre-clip grad over ALL groups of a role, so it isolates WHICH belief-component
    family the LEARNING signal flows into regardless of which tables are live (e.g. the s_* tables under
    prior_source='model_channel') -- the aggregate grad_norm.png sums them and hides that. Captured
    AFTER unscale_ but BEFORE clip in train_step, so these are the true pre-clip magnitudes; the E-step
    inference analogue is estep_grad_norm_decomposition.
    """
    spec = [
        ("grad_norm_mu",    r"$\|\nabla_\mu \mathcal{L}\|_2$",    _CB[0], "mu"),
        ("grad_norm_sigma", r"$\|\nabla_\Sigma \mathcal{L}\|_2$", _CB[1], "sigma"),
        ("grad_norm_phi",   r"$\|\nabla_\phi \mathcal{L}\|_2$",   _CB[2], "phi"),
    ]
    return _grad_norm_decomp_fig(
        history, spec, "M-step gradient norm decomposition",
        r"pre-clip $\|\nabla_\theta \mathcal{L}\|_2$ (per role)", path)


@register_figure("estep_grad_norm_decomposition")
def plot_estep_grad_norm_decomposition(
    history: Dict,                       # step + any of: estep_grad_norm_mu, estep_grad_norm_sigma, estep_grad_norm_phi

    *,
    path:    Optional[str] = None,
):
    r"""E-STEP per-component belief-gradient L2 norms over training: the mu / sigma / phi decomposition
    of the inference free-energy gradient.

    The companion to the M-step grad_norm_decomposition: where that shows how hard the LEARNING (M-step,
    parameter) gradient pushes each belief-component family, this shows how hard the INFERENCE (E-step)
    gradient ``\nabla F`` pushes the belief tuple ``(mu, Sigma, phi)`` itself -- the RAW gradient inside
    the last E-step iteration, before the Fisher / natural-gradient preconditioning, captured by
    model.forward(estep_grad_out=...). A component reads 0 (dropped on the log y) when its substep is off
    (e.g. phi when e_phi_lr=0).
    """
    spec = [
        ("estep_grad_norm_mu",    r"$\|\nabla_\mu F\|_2$",    _CB[0], "mu"),
        ("estep_grad_norm_sigma", r"$\|\nabla_\Sigma F\|_2$", _CB[1], "sigma"),
        ("estep_grad_norm_phi",   r"$\|\nabla_\phi F\|_2$",   _CB[2], "phi"),
    ]
    return _grad_norm_decomp_fig(
        history, spec, "E-step gradient norm decomposition",
        r"E-step $\|\nabla F\|_2$ (per component)", path)


@register_figure("estep_convergence")
def plot_estep_convergence(
    trace:    Dict,                      # e_step_belief_trace output: mu, sigma, phi, free_energy

    *,
    diagonal: Optional[bool] = None,
    path:     Optional[str]  = None,
):
    r"""F2: E-step convergence -- free energy and belief motion across inner iterations.

    Panel A is the global F(t) over inner iterations with the trained budget marked. Panel B is the
    belief-motion residuals on a log axis -- the mean step, the affine-invariant SPD covariance step,
    and the gauge step -- each median + 10-90 band over tokens, shrinking toward a fixed point.
    """
    from vfe3 import metrics
    fe = _np(trace["free_energy"])
    res = metrics.estep_residuals(trace["mu"], trace["sigma"], trace["phi"], diagonal=diagonal)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(np.arange(fe.size), fe, "o-", color=_CB[0], ms=4, lw=1.6)
    axes[0].axhline(fe[-1], color="#888888", ls=":", lw=1)
    axes[0].set(xlabel="E-step inner iteration", ylabel="free energy (nats)", title="E-step descent")
    t = np.arange(1, _np(res["r_mu"]).shape[0] + 1)
    _median_band(axes[1], t, _np(res["r_mu"]), _CB[0], r"$r_\mu$")
    _median_band(axes[1], t, _np(res["r_sigma"]), _CB[1], r"$r_\Sigma$ (SPD)")
    _median_band(axes[1], t, _np(res["r_phi"]), _CB[2], r"$r_\phi$")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="E-step inner iteration", ylabel="belief step length", title="Convergence to fixed point")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_s_channel_refinement(
    s_data:  Dict,                       # extract.s_channel_refinement output (caller skips when None)

    *,
    path:    Optional[str] = None,
):
    r"""Model-channel (s) refinement under ``s_e_step=True``: how the s E-step moves the model belief.

    Panel A is the per-position KL to the frozen hyper-prior centroid r BEFORE (static ``s0``) and
    AFTER (refined ``s1``) the s E-step -- a healthy model channel pulls toward r, so the refined
    bars sit at or below the static ones. Panel B is the per-position refinement magnitude
    ``||Delta mu_s||`` and ``||Delta log sigma_s||`` -- where on the sequence the s-channel acts.
    """
    kl0 = _np(s_data["kl_s0_r"]); kl1 = _np(s_data["kl_s1_r"])
    dmu = _np(s_data["mu_delta"]); dls = _np(s_data["logsigma_delta"])
    pos = np.arange(kl0.size)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    w = 0.4
    axes[0].bar(pos - w / 2, kl0, width=w, color=_CB[0], label=r"$\mathrm{KL}(s_0\|r)$ (static)")
    axes[0].bar(pos + w / 2, kl1, width=w, color=_CB[1], label=r"$\mathrm{KL}(s_1\|r)$ (refined)")
    axes[0].set(xlabel="token position", ylabel="KL to hyper-prior $r$ (nats)",
                title="Model-channel consensus toward $r$")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot(pos, dmu, "o-", color=_CB[2], ms=4, lw=1.5, label=r"$\|\Delta\mu_s\|$")
    axes[1].plot(pos, dls, "s-", color=_CB[3 % len(_CB)], ms=4, lw=1.5, label=r"$\|\Delta\log\sigma_s\|$")
    axes[1].set(xlabel="token position", ylabel="refinement magnitude", title="s-channel E-step motion")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_model_channel_belief(
    s_data:  Dict,                       # extract.model_channel_belief output (caller skips when None)

    *,
    path:    Optional[str] = None,
):
    r"""The model channel ``s`` (the ``s`` figure): the per-token model-channel beliefs s_i = N(s_mu, s_sigma).

    Panel A is s_mu per coordinate (mean +/- sd over tokens) -- the model channel's mean structure and its
    spread across the sequence. Panel B is the per-token variance spectrum (sorted descending), median with
    a 10-90 band over tokens on a log axis -- how confident / how anisotropic the model beliefs are. The s
    tables are always diagonal (V, K), so the spectrum IS the variances.
    """
    mu_m = _np(s_data["mu_mean"]); mu_s = _np(s_data["mu_std"]); spec = _np(s_data["spectrum"])
    coord = np.arange(mu_m.size)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].errorbar(coord, mu_m, yerr=mu_s, fmt="o", color=_CB[0], ms=4, capsize=2,
                     label=r"$s_\mu$ (mean $\pm$ sd over tokens)")
    axes[0].axhline(0.0, color="#888888", ls=":", lw=1)
    axes[0].set(xlabel="coordinate $k$", ylabel=r"$s_\mu$", title="Model channel $s$: mean per coordinate")
    axes[0].legend(fontsize=8, frameon=False)
    rank = np.arange(spec.shape[1])
    _median_band(axes[1], rank, spec.T, _CB[1], r"$s_\Sigma$ spectrum (median, 10-90)")   # (K, N): median over tokens
    axes[1].set_yscale("log")
    axes[1].set(xlabel="spectral rank", ylabel=r"$s$ variance", title="Model-channel variance spectrum")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_hyper_prior_centroid(
    r_data:  Dict,                       # extract.hyper_prior_centroid output (caller skips when None)

    *,
    path:    Optional[str] = None,
):
    r"""The hyper-prior centroid ``r`` (the ``r`` figure) and how the model channel ``s`` clusters around it.

    Panel A overlays the centroid mean r_mu on the s_mu population (mean +/- sd over tokens) per coordinate
    -- r is the consensus the model beliefs are regularized toward by lambda_h KL(s||r). Panel B is the
    centroid variance r_Sigma against the mean model-channel variance, log y. A frozen r (the default) is a
    fixed reference; a learnable r tracks the s population.
    """
    r_mu = _np(r_data["r_mu"]); r_sig = _np(r_data["r_sigma"])
    s_mu_m = _np(r_data["s_mu_mean"]); s_mu_s = _np(r_data["s_mu_std"]); s_sig_m = _np(r_data["s_sigma_mean"])
    coord = np.arange(r_mu.size)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].fill_between(coord, s_mu_m - s_mu_s, s_mu_m + s_mu_s, color=_CB[0], alpha=0.2,
                         label=r"$s_\mu$ population (mean $\pm$ sd)")
    axes[0].plot(coord, s_mu_m, color=_CB[0], lw=1.5)
    axes[0].plot(coord, r_mu, "o-", color=_CB[1], lw=1.5, ms=4, label=r"centroid $r_\mu$")
    axes[0].set(xlabel="coordinate $k$", ylabel="mean", title=r"Hyper-prior centroid $r$ vs model channel $s$")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot(coord, r_sig, "o-", color=_CB[1], lw=1.5, ms=4, label=r"$r_\Sigma$ (centroid)")
    axes[1].plot(coord, s_sig_m, "s-", color=_CB[0], lw=1.5, ms=4, label=r"$s_\Sigma$ (mean over tokens)")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="coordinate $k$", ylabel="variance", title="Centroid vs model-channel variance")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_hyper_prior_coupling(
    h_data:  Dict,                       # extract.hyper_prior_coupling output (caller skips when None)

    *,
    path:    Optional[str] = None,
):
    r"""The hyper-prior coupling block ``h`` (the ``h`` figure): per-token KL(s_i||r), the lambda_h block.

    The per-position divergence of the model channel from the hyper-prior centroid -- the integrand of the
    hyper-prior term lambda_h mean_i KL(s_i||r) that the model channel descends. The dashed line is the
    mean (the value lambda_h scales); a healthy / converged channel sits low and flat across positions.
    """
    kl = _np(h_data["kl_s_r"]); pos = np.arange(kl.size); lam = float(h_data.get("lambda_h", 0.0))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(pos, kl, color=_CB[2])
    ax.axhline(float(np.mean(kl)) if kl.size else 0.0, color="#888888", ls="--", lw=1,
               label=f"mean {float(np.mean(kl)) if kl.size else 0.0:.3g}")
    ax.set(xlabel="token position", ylabel=r"$\mathrm{KL}(s_i\|r)$ (nats)",
           title=rf"Hyper-prior coupling (the $\lambda_h$ block, $\lambda_h={lam:g}$)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("ln3_symmetry_breaking")
def plot_ln3_symmetry_breaking(
    frozen:  Dict,                       # {step, val_ce, omega (N,N,K,K), beta (H,N,N)}
    learned: Dict,

    *,
    period:  int = 3,
    path:    Optional[str] = None,
):
    r"""F3: the gauge does the work -- ln(3) symmetry-breaking on the period-3 stream.

    Panel A: val CE vs step for the frozen (gauge off) and learned (gauge on) arms, with the
    analytic averaging floor CE = ln 3 drawn. Panel B: the directed transport asymmetry of the
    learned arm. Panel C: per-head period-3 / prev-token structure scores, frozen vs learned.
    """
    from vfe3 import metrics
    ln3 = float(np.log(3.0))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    axes[0].plot(_np(frozen["step"]), _np(frozen["val_ce"]), color=_CB[1], lw=2, label="frozen (gauge off)")
    axes[0].plot(_np(learned["step"]), _np(learned["val_ce"]), color=_CB[2], lw=2, label="learned (gauge on)")
    axes[0].axhline(ln3, color="#444444", ls="--", lw=1.2, label=r"$\ln 3$ averaging floor")
    axes[0].set(xlabel="training step", ylabel="val CE (nats)", title="Order learning vs the averaging floor")
    axes[0].legend(fontsize=8, frameon=False)
    asym = _np(metrics.transport_asymmetry(learned["omega"]))
    vmax = float(np.abs(asym).max()) or 1.0
    im = axes[1].imshow(asym, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=axes[1], shrink=0.8, label=r"$\|\Omega_{ij}-\Omega_{ji}\|_F$")
    axes[1].set(xlabel="key j", ylabel="query i", title="Transport asymmetry (learned)")
    fr = metrics.structured_head_scores(frozen["beta"], period=period)
    le = metrics.structured_head_scores(learned["beta"], period=period)
    labels = ["prev-token", f"period-{period}"]
    xs = np.arange(len(labels))
    axes[2].bar(xs - 0.2, [float(_np(fr["prev_token"]).mean()), float(_np(fr["period_match"]).mean())],
                width=0.4, color=_CB[1], label="frozen")
    axes[2].bar(xs + 0.2, [float(_np(le["prev_token"]).mean()), float(_np(le["period_match"]).mean())],
                width=0.4, color=_CB[2], label="learned")
    axes[2].set(xticks=xs, xticklabels=labels, ylabel="mean attention mass", title="Structured-head scores")
    axes[2].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("belief_trajectories")
def plot_belief_trajectories(
    trace:       Dict,                   # e_step_belief_trace output (mu (T+1,N,K), ...)
    layer_trace: Optional[Dict] = None,  # across_layer_belief_trace output (d_ai, effective_rank)

    *,
    path:        Optional[str] = None,
):
    r"""F4: belief trajectories across E-step iterations (mean-space path) and across layers (SPD).

    Panel A: a shared-PCA 2-D quiver of the belief means as inference iterates (arrows mu_t ->
    mu_{t+1}, coloured by token position). Panel B: the across-layer cumulative affine-invariant SPD
    geodesic distance and mean effective rank. The across-training axis is out of scope (needs
    per-checkpoint replay).
    """
    mu = _np(trace["mu"])                                         # (T+1, N, K)
    t1, n, k = mu.shape
    flat = mu.reshape(-1, k) - mu.reshape(-1, k).mean(0)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    proj = (flat @ vt[:2].T).reshape(t1, n, 2)
    ncol = 2 if layer_trace is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.2), squeeze=False)
    ax = axes[0][0]
    cmap = plt.cm.viridis(np.linspace(0, 1, n))
    for j in range(n):
        ax.plot(proj[:, j, 0], proj[:, j, 1], "-", color=cmap[j], lw=0.8, alpha=0.7)
        ax.quiver(proj[:-1, j, 0], proj[:-1, j, 1],
                  np.diff(proj[:, j, 0]), np.diff(proj[:, j, 1]),
                  angles="xy", scale_units="xy", scale=1, color=cmap[j], width=0.004, alpha=0.7)
    ax.scatter(proj[0, :, 0], proj[0, :, 1], s=14, color="#333333", marker="o", label="init")
    ax.set(xlabel="PC 1", ylabel="PC 2", title="Belief-mean path over E-step iterations")
    if layer_trace is not None:
        axb = axes[0][1]
        layers = np.arange(_np(layer_trace["d_ai"]).size)
        axb.plot(layers, _np(layer_trace["d_ai"]), "o-", color=_CB[0], label=r"$d_{AI}(\Sigma^0,\Sigma^l)$")
        axb.set(xlabel="layer", ylabel="cumulative SPD distance", title="Belief geometry vs depth")
        axc = axb.twinx()
        axc.plot(layers, _np(layer_trace["effective_rank"]), "s--", color=_CB[1])
        axc.set_ylabel("mean effective rank", color=_CB[1])
        axb.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


# --- linguistic-category labelling of gpt2-BPE tokens (colour + legend for the belief UMAP) -----

# A compact English function-word (stopword) set for the function/content taxonomy. Matched on the
# stripped, lower-cased decoded token, so a BPE word-start (" the") and a bare token ("the") both hit.
_STOPWORDS = frozenset((
    "a an the this that these those some any all each every no none "
    "i you he she it we they me him her us them my your his its our their mine yours hers ours theirs "
    "is are was were be been being am do does did doing have has had having "
    "will would shall should can could may might must ought "
    "of in on at by for with about against between into through during before after "
    "above below to from up down out off over under again further then once here there "
    "and or but nor so yet because as if while although though unless until than "
    "not only own same too very just more most other such "
    "what which who whom whose where when why how "
    "s t re ve ll d m"                                       # common BPE contraction fragments
).split())

_BPE_CAT_NAMES  = ["punctuation", "number", "word (lc)", "word (Cap)", "subword", "space/other"]
_FUNC_CAT_NAMES = ["punctuation", "number", "function", "content", "other"]


def _bpe_category(text: str) -> int:
    r"""Index into ``_BPE_CAT_NAMES`` from a decoded gpt2 token's STRUCTURE.

    Leverages BPE structure: a leading space marks a word-start, its absence a continuation subword
    (e.g. ``ing``). Categories: punctuation/symbol (no letters), number (all-digit), word-start
    lower-case, word-start Capitalized, continuation subword, whitespace/other.
    """
    if not text or text.isspace():
        return 5
    core = text.strip()
    if not core:
        return 5
    if core.isdigit():
        return 1
    if not any(c.isalpha() for c in core):
        return 0
    if text[:1] == " ":                                      # leading space -> a new word starts here
        return 3 if core[:1].isupper() else 2
    return 4                                                 # no leading space + letters -> subword piece


def _funccontent_category(text: str) -> int:
    r"""Index into ``_FUNC_CAT_NAMES``: punctuation, number, function-word, content-word, or other.

    Linguistic split on the stripped lower-cased token: a purely-alphabetic token is a FUNCTION word
    if it is in ``_STOPWORDS`` else a CONTENT word; non-alphabetic tokens fall to punctuation / number
    / other.
    """
    core = text.strip().lower()
    if not core:
        return 4
    if core.isdigit():
        return 1
    if not any(c.isalpha() for c in core):
        return 0
    if not core.isalpha():
        return 4
    return 2 if core in _STOPWORDS else 3


def _token_category_labels(
    token_ids: object,                   # (M,) token ids
    decode:    object,                   # decode(list[int]) -> str (e.g. the gpt2 tiktoken decoder)
    taxonomy:  str,                      # 'bpe' or 'function_content'
) -> tuple:                              # (labels (M,) int, names list[str])
    r"""Per-token category index under ``taxonomy`` (each unique id decoded + classified once)."""
    if taxonomy == "bpe":
        fn, names = _bpe_category, _BPE_CAT_NAMES
    elif taxonomy == "function_content":
        fn, names = _funccontent_category, _FUNC_CAT_NAMES
    else:
        raise ValueError(f"unknown taxonomy {taxonomy!r} (expected 'bpe' / 'function_content')")
    ids = _np(token_ids).astype(int)
    cat = {int(t): fn(decode([int(t)])) for t in np.unique(ids)}
    return np.array([cat[int(t)] for t in ids]), names


def _scatter_by_category(ax, coords: np.ndarray, labels, names) -> None:
    r"""Scatter ``coords`` coloured by integer ``labels`` with a per-category legend (count shown).

    ``labels is None`` (no decoder available) degrades to a single-colour scatter and no legend."""
    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.5, color="#999999", linewidths=0)
        return
    for idx, name in enumerate(names):
        m = labels == idx
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.6, linewidths=0,
                   color=_CB[idx % len(_CB)], label=f"{name} ({int(m.sum())})")
    ax.legend(fontsize=6.5, frameon=False, markerscale=1.6, loc="best")


def _annotate_frequent_tokens(ax, coords: np.ndarray, token_ids: np.ndarray, decode, n_label: int) -> None:
    r"""Mark + label the ``n_label`` most frequent tokens at the CENTROID of their occurrences."""
    if n_label <= 0:
        return
    uniq, counts = np.unique(token_ids, return_counts=True)
    for t in uniq[np.argsort(counts)[::-1][:n_label]]:
        m = token_ids == t
        cx, cy = float(coords[m, 0].mean()), float(coords[m, 1].mean())
        txt = (decode([int(t)]).strip() if decode is not None else str(int(t))) or "·"
        ax.scatter([cx], [cy], s=45, marker="*", facecolor="white", edgecolor="black",
                   linewidths=0.8, zorder=5)
        ax.annotate(txt, (cx, cy), fontsize=8, fontweight="bold", zorder=6,
                    xytext=(3, 3), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))


def _cluster_embedding(
    coords: np.ndarray,                  # (M, 2) UMAP coordinates

    *,
    seed:   int = 0,
) -> np.ndarray:                         # (M,) integer cluster labels (-1 = noise)
    r"""Density-cluster a 2-D embedding into data-driven groups (HDBSCAN; KMeans fallback).

    HDBSCAN finds variable-density clusters AND an explicit noise label (-1) -- the right tool for the
    belief embedding's tight peripheral function-word islands around a diffuse content core -- with
    ``min_cluster_size`` scaled to the cloud size. Falls back to KMeans (no noise label) only when the
    installed sklearn predates ``cluster.HDBSCAN`` (<1.3). Deterministic given ``coords``."""
    M = coords.shape[0]
    try:
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=max(20, M // 60), min_samples=5).fit_predict(coords)
    except Exception:                                            # pragma: no cover - old sklearn only
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=max(2, min(14, M // 50)), n_init=10, random_state=seed).fit_predict(coords)


def _cluster_lift_labels(
    token_ids: np.ndarray,               # (M,) token ids
    labels:    np.ndarray,               # (M,) cluster labels (-1 = noise)
    decode:    Optional[object],         # decode([id]) -> str; None -> raw id string

    *,
    k:         int = 3,
    floor:     int = 2,
) -> Dict[int, str]:
    r"""Per-cluster DISTINCTIVE-token label by enrichment (lift): one comma-joined string per cluster.

    The globally frequent tokens (the, comma, of) occur in EVERY cluster, so raw-frequency labels are
    uninformative; the lift score lift(t, c) = (count of t in c / |c|) / (count of t / M) ranks a token by
    how CONCENTRATED it is in cluster c -- surfacing what the cluster is about. Keeps candidates with
    in-cluster count >= ``floor``, ranks by lift, and returns the top-``k`` unique decoded strings
    (stripped; replacement-char / non-printable BPE-byte fragments dropped). ``decode is None`` -> raw id.
    """
    ids = token_ids.astype(int)
    M = ids.size
    uniq = np.unique(ids)
    sm = {int(t): (decode([int(t)]) if decode is not None else str(int(t))) for t in uniq}
    glob = {int(t): int((ids == t).sum()) for t in uniq}
    out: Dict[int, str] = {}
    for c in sorted(set(labels.tolist()) - {-1}):
        m = labels == c
        n_c = int(m.sum())
        if not n_c:
            continue
        loc_ids, loc_ct = np.unique(ids[m], return_counts=True)
        scored = []
        for t, ct in zip(loc_ids.tolist(), loc_ct.tolist()):
            if ct < floor:
                continue
            s = str(sm[int(t)]).strip()
            if not s or "�" in s or not s.isprintable():    # drop empty / replacement-char / byte fragments
                continue
            scored.append(((ct / n_c) / (glob[int(t)] / M), s))
        scored.sort(reverse=True)
        seen, toks = set(), []
        for _, s in scored:
            if s not in seen:
                seen.add(s)
                toks.append(s)
            if len(toks) >= k:
                break
        if toks:
            out[c] = ", ".join(toks)
    return out


@register_figure("belief_umap")
def plot_belief_umap(
    bank:             Dict,              # belief_bank output: mu, sigma, phi, token_ids, seq_idx
    channel:          str = "mu",        # which belief channel to embed: 'mu' / 'sigma' / 'phi'

    *,
    kind:             str              = "Belief",  # title noun: 'Belief' (q channel) / 'Model' (s channel)
    decode:           Optional[object] = None,   # decode(list[int]) -> str; None -> id labels
    n_clusters_label: int              = 14,     # annotate the N largest clusters
    seed:             int              = 0,
    sil_sample:       int              = 2000,
    path:             Optional[str]    = None,
):
    r"""F5: data-driven cluster map of one belief channel, each cluster labelled by its distinctive tokens.

    ONE figure per channel (the caller emits mu / sigma / phi separately). The channel is embedded
    faithfully to its geometry (mu Euclidean, Sigma in the log-Euclidean chart, phi in the gauge
    coordinates), the 2-D cloud is density-clustered (:func:`_cluster_embedding`), and the
    ``n_clusters_label`` largest clusters are annotated -- a star at the cluster MEDOID and a leader to a
    collision-free margin label of its top distinctive tokens by enrichment/lift (:func:`_cluster_lift_labels`),
    so the reader sees WHAT each cluster is. A faint grey kernel-density underlay makes the dense core read
    as density rather than an opaque blob; partial-opacity rasterized points keep overplotting legible; and
    HDBSCAN noise is drawn light grey behind. This replaces the a-priori linguistic-category colouring,
    which does NOT separate the belief geometry (silhouette near zero) -- that quantitative view lives in the
    companion :func:`plot_belief_category_separation`; the function/content silhouette is noted in the title
    for context when ``decode`` is available. ``decode is None`` falls back to raw token-id labels.
    """
    feats = _belief_channel_features(bank, channel)
    coords = _np(umap_embed(feats, seed=seed)).astype(float)
    token_ids = _np(bank["token_ids"]).astype(int)
    labels = _cluster_embedding(coords, seed=seed)
    cl = sorted(set(labels.tolist()) - {-1}, key=lambda c: -int((labels == c).sum()))
    noise = float((labels == -1).mean())
    lab_text = _cluster_lift_labels(token_ids, labels, decode, k=3)

    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
    ymin, ymax = float(coords[:, 1].min()), float(coords[:, 1].max())
    rx, ry = (xmax - xmin) or 1.0, (ymax - ymin) or 1.0
    try:                                                         # faint density underlay (best-effort)
        from scipy.stats import gaussian_kde
        gx, gy = np.mgrid[xmin:xmax:120j, ymin:ymax:120j]
        zz = gaussian_kde(coords.T)(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        ax.contourf(gx, gy, zz, levels=8, cmap="Greys", alpha=0.28, zorder=0)
    except Exception:                                            # singular cloud / no scipy -> skip underlay
        pass
    nm = labels == -1
    if nm.any():
        ax.scatter(coords[nm, 0], coords[nm, 1], s=4, c="#bbbbbb", alpha=0.30, linewidths=0,
                   rasterized=True, zorder=1)
    palette = plt.cm.tab20(np.linspace(0, 1, 20))
    col = {c: palette[i % 20] for i, c in enumerate(cl)}
    for c in cl:
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], s=6, color=col[c], alpha=0.5, linewidths=0,
                   rasterized=True, zorder=2)
    # Collision-free margin callouts: assign each labelled cluster to its NEAREST margin (top / bottom /
    # left / right), then space the labels evenly along that margin with a leader to the cluster medoid.
    # Four sides (vs one top + one bottom row) spread ~14 labels to ~3-4 per side, so neither crowds and
    # the leaders stay short -- the failure the single-row layout hit when most clusters sat one side of
    # the centre.
    cx, cy = float(np.median(coords[:, 0])), float(np.median(coords[:, 1]))
    items = []
    for c in cl[:n_clusters_label]:
        if c not in lab_text:
            continue
        pts = coords[labels == c]
        md = pts[np.argmin(((pts - pts.mean(0)) ** 2).sum(1))]   # medoid: robust in-cluster anchor
        items.append((c, md, lab_text[c]))
    sides: Dict[str, list] = {"top": [], "bottom": [], "left": [], "right": []}
    for c, md, lab in items:
        dx, dy = (md[0] - cx) / rx, (md[1] - cy) / ry
        if abs(dx) >= abs(dy):
            sides["right" if dx >= 0 else "left"].append((c, md, lab))
        else:
            sides["top" if dy >= 0 else "bottom"].append((c, md, lab))
    x_left, x_right = xmin - 0.04 * rx, xmax + 0.04 * rx
    y_top, y_bot = ymax + 0.15 * ry, ymin - 0.15 * ry

    def _callout(c, md, lab, lx, ly, ha, va):
        ax.plot([md[0], lx], [md[1], ly], color=col[c], lw=0.7, alpha=0.7, zorder=4)
        ax.scatter([md[0]], [md[1]], marker="*", s=55, color=col[c], edgecolor="black",
                   linewidths=0.5, zorder=6)
        ax.annotate(lab, xy=(lx, ly), fontsize=8, fontweight="bold", zorder=7, ha=ha, va=va,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col[c], lw=1.0, alpha=0.9))

    for side in ("top", "bottom"):
        row = sorted(sides[side], key=lambda it: it[1][0])      # order by x
        xs = np.linspace(xmin, xmax, len(row)) if len(row) > 1 else [0.5 * (xmin + xmax)]
        yrow, va = (y_top, "bottom") if side == "top" else (y_bot, "top")
        for (c, md, lab), lx in zip(row, xs):
            _callout(c, md, lab, lx, yrow, "center", va)
    for side in ("left", "right"):
        col_ = sorted(sides[side], key=lambda it: it[1][1])     # order by y
        ys = np.linspace(ymin, ymax, len(col_)) if len(col_) > 1 else [0.5 * (ymin + ymax)]
        xcol, ha = (x_left, "right") if side == "left" else (x_right, "left")
        for (c, md, lab), ly in zip(col_, ys):
            _callout(c, md, lab, xcol, ly, ha, "center")
    ax.set_xlim(xmin - 0.30 * rx, xmax + 0.30 * rx)             # room for the left / right label boxes
    ax.set_ylim(y_bot - 0.08 * ry, y_top + 0.08 * ry)
    ax.set_xticks([]); ax.set_yticks([])
    title = (f"{kind} {channel} — {len(cl)} data-driven clusters ({noise * 100:.0f}% noise); "
             f"labels = distinctive (lift) tokens")
    if decode is not None:
        try:
            cats, _ = _token_category_labels(token_ids, decode, "function_content")
            sil = clustering_metrics(_np(feats), cats, sample_size=sil_sample)["silhouette"]
            title += (f"\nfunction/content category silhouette {sil:+.2f} "
                      f"(~0 -> a-priori categories do not separate; clusters above are data-driven)")
        except Exception:
            pass
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("belief_category_separation")
def plot_belief_category_separation(
    bank:       Dict,                    # belief_bank output

    *,
    decode:     Optional[object] = None,
    channels:   tuple            = ("mu", "sigma", "phi"),
    taxonomies: tuple            = ("bpe", "function_content"),
    sil_sample: int              = 2000,
    path:       Optional[str]    = None,
):
    r"""F5 (companion): how strongly each belief channel is organized by each linguistic taxonomy.

    Grouped bars of the native-space silhouette of the category labels, per channel (mu / Sigma / phi)
    and taxonomy (BPE structure / function-content). A bar near zero means that channel does not
    separate that taxonomy; higher means the linguistic category is geometrically encoded. The
    quantitative companion to the per-channel UMAP scatters. Empty (NaN) bars when ``decode is None``.
    """
    token_ids = _np(bank["token_ids"]).astype(int)
    x = np.arange(len(channels))
    width = 0.8 / max(len(taxonomies), 1)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for j, tax in enumerate(taxonomies):
        sils = []
        for ch in channels:
            if decode is None:
                sils.append(float("nan"))
                continue
            labels, _ = _token_category_labels(token_ids, decode, tax)
            sils.append(clustering_metrics(_np(_belief_channel_features(bank, ch)),
                                           labels, sample_size=sil_sample)["silhouette"])
        ax.bar(x + (j - (len(taxonomies) - 1) / 2) * width, sils, width=width,
               color=_CB[j % len(_CB)], label=tax.replace("_", "/"))
    ax.axhline(0.0, color="#444444", lw=0.8)
    ax.set(xticks=x, xticklabels=list(channels), ylabel="silhouette (native space)",
           title="Belief organization by linguistic category")
    ax.legend(fontsize=8, frameon=False, title="taxonomy")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("gauge_equivariance")
def plot_gauge_equivariance(
    resid: Dict,                         # gauge_equivariance_residual output

    *,
    path:  Optional[str] = None,
):
    r"""F6: gauge-equivariance certificate -- energy and attention invariant to the structure group.

    Log-scale ECDFs of the relative residual of E_ij and beta_ij under random IN-group elements
    (clustered near float32 eps) vs a matched OUT-of-group control (far right), with a machine-eps
    reference line. A real symmetry, not a decorative one.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in ((axes[0], "energy", r"energy $E_{ij}$"), (axes[1], "beta", r"attention $\beta_{ij}$")):
        _ecdf(ax, resid[f"{key}_in_group"], _CB[2], "in-group")
        _ecdf(ax, resid[f"{key}_out_group"], _CB[1], "out-of-group")
        ax.axvline(_EPS_F32, color="#444444", ls=":", lw=1, label="float32 eps")
        ax.set_xscale("log")
        ax.set(xlabel="relative residual", ylabel="ECDF", title=f"Equivariance of {title}")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("gauge_head_specialization")
def plot_gauge_head_specialization(
    per_head:     Dict,                  # per_head_gauge_invariants output: logdet (M,H), anisotropy (M,H)
    head_entropy: Optional[object] = None,  # (H,) per-head mean attention entropy

    *,
    path:         Optional[str] = None,
):
    r"""F7: per-head gauge specialization and its link to attention structure (FINAL layer).

    The gauge frame here is the converged FINAL-block frame (``cstate["exp_phi"]``), so "head" is the
    per-irrep-block index of that single layer, not a per-layer head; pair it with the FINAL layer's
    per-head entropy (see report.py) so both axes describe the same depth. Per-LAYER gauge action is
    in the ``per_layer_diagnostics`` figure. Panel A: per-head violins of the group-correct gauge
    invariant (log-volume). Panel B: per-head violins of the shear/anisotropy, or -- when
    ``head_entropy`` is given -- a scatter of per-head gauge magnitude vs attention row entropy.
    """
    logdet = _np(per_head["logdet"])                             # (M, H) -- H = irrep blocks of the FINAL frame
    aniso = _np(per_head["anisotropy"])
    h = logdet.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].violinplot([logdet[:, i] for i in range(h)], showmeans=True)
    axes[0].axhline(0.0, color="#888888", ls=":", lw=1)
    axes[0].set(xticks=range(1, h + 1), xticklabels=[f"h{i}" for i in range(h)],
                ylabel=r"$\log|\det\exp\phi^{(h)}|$", title="Per-head gauge volume (final layer)")
    if head_entropy is not None:
        axes[1].scatter(np.abs(logdet).mean(0), _np(head_entropy), s=40, color=_CB[0])
        axes[1].set(xlabel="mean |gauge volume|", ylabel="attention row entropy (nats)",
                    title="Gauge action vs attention (final layer)")
    else:
        axes[1].violinplot([aniso[:, i] for i in range(h)], showmeans=True)
        axes[1].set(xticks=range(1, h + 1), xticklabels=[f"h{i}" for i in range(h)],
                    ylabel=r"$s_{\max}/s_{\min}$", title="Per-head shear (final layer)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("attention_structure")
def plot_attention_structure(
    beta: object,                        # (L,H,N,N) or (H,N,N) attention weights

    *,
    path: Optional[str] = None,
):
    r"""F8: attention structure -- per-(layer,head) entropy, redundancy, and distance decay.

    For a multi-layer model the input is (L, H, N, N); the leading (layer, head) axes are flattened
    to L*H curves but each keeps its identity as ``L{l}H{h}`` (NOT renumbered 0..L*H-1), so depth is
    legible. Panel A: per-(layer,head) row-entropy violins vs the uniform log N reference (does
    attention sharpen). Panel B: the (layer,head)x(layer,head) Jensen-Shannon redundancy heatmap
    (which maps are specialized vs redundant, across layers too). Panel C: the per-(layer,head)
    attention-vs-offset profile on a log axis (positional decay). A single-layer (H, N, N) input
    falls back to plain ``h{h}`` labels.
    """
    from vfe3 import metrics
    import torch
    b = beta if hasattr(beta, "dim") else torch.as_tensor(_np(beta))
    if b.dim() == 4:                                             # (L, H, N, N) -> (L*H, N, N), KEEP (layer,head)
        L, H = int(b.shape[0]), int(b.shape[1])                  # identity: labels track the reshape C-order
        labels = [f"L{l}H{hh}" for l in range(L) for hh in range(H)]   # L0H0, L0H1, L1H0, ... matches reshape(-1)
        b = b.reshape(-1, b.shape[-2], b.shape[-1])
    else:                                                        # (H, N, N) single layer -> plain head labels
        labels = [f"h{hh}" for hh in range(int(b.shape[0]))]
    rows = _np(metrics.attention_entropy_rows(b))               # (L*H, N)
    n = b.shape[-1]
    h = b.shape[0]
    _rot = 90 if h > 3 else 0                                    # rotate crowded (layer,head) ticks
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    axes[0].violinplot([rows[i] for i in range(h)], showmeans=True)
    axes[0].axhline(float(np.log(n)), color="#444444", ls="--", lw=1, label=r"uniform $\log N$")
    axes[0].set(xticks=range(1, h + 1), ylabel="row entropy (nats)", title="Per-(layer,head) attention entropy")
    axes[0].set_xticklabels(labels, rotation=_rot, fontsize=7)
    axes[0].legend(fontsize=8, frameon=False)
    js = _np(metrics.head_redundancy_js(b))                     # (L*H, L*H) over all (layer,head) maps
    im = axes[1].imshow(js, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="JS divergence (nats)")
    axes[1].set(xticks=range(h), yticks=range(h), xlabel="layer-head", ylabel="layer-head", title="Head redundancy")
    axes[1].set_xticklabels(labels, rotation=90, fontsize=6)
    axes[1].set_yticklabels(labels, fontsize=6)
    dd = metrics.attention_distance_decay(b)
    prof = _np(dd["profile"])                                    # (L*H, N)
    off = _np(dd["offsets"])
    for i in range(h):
        axes[2].plot(off, np.clip(prof[i], 1e-8, None), color=plt.cm.viridis(i / max(h - 1, 1)), lw=1.2, label=labels[i])
    axes[2].set_yscale("log")
    axes[2].set(xlabel="offset |i - j|", ylabel=r"mean $\beta$", title="Attention distance decay")
    axes[2].legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("per_layer_diagnostics")
def plot_per_layer_diagnostics(
    per_layer: Dict,                     # diagnostics_per_layer output: each value an (L,) sequence

    *,
    path:      Optional[str] = None,
):
    r"""F8b: per-LAYER (inference-depth) diagnostics -- the depth axis the stack collapses.

    The aggregate metrics.csv and every converged-state figure show only the FINAL block; this
    plots ``VFEModel.diagnostics_per_layer`` block-by-block so depth is legible: which layer carries
    the coupling vs entropy mass, and whether gauge holonomy / curvature and belief rank build up
    with depth. Panel A: belief-channel free-energy budget (self-coupling, belief-coupling, attention
    entropy, total) vs layer. Panel B: holonomy / Wilson curvature vs layer (~0 on the flat cocycle).
    Panel C: gauge action (trace spread, group-invariant spread, mean ``||phi||``) vs layer. Panel D:
    belief geometry (effective rank, attention entropy; condition number on a twin log axis) vs layer.
    """
    g = lambda k: _np(per_layer[k]).reshape(-1)
    L = g("self_coupling").size if "self_coupling" in per_layer else g(next(iter(per_layer))).size
    x = np.arange(L)
    _m = "o-" if L > 1 else "o"                                  # a single layer has no line to draw
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for k, c, lab in (("self_coupling", _CB[0], r"self $\alpha\,$KL$(q\|p)$"),
                      ("belief_coupling", _CB[1], r"belief $\sum\beta\,$KL"),
                      ("attention_entropy", _CB[2], "attn entropy term"),
                      ("total", _CB[7], "total (belief F)")):
        if k in per_layer:
            axes[0][0].plot(x, g(k), _m, color=c, lw=1.4, label=lab)
    axes[0][0].set(xticks=x, xlabel="layer", ylabel="nats", title="Free-energy budget by layer")
    axes[0][0].legend(fontsize=8, frameon=False)
    for k, c, lab in (("holonomy_deviation", _CB[0], r"$\|H-I\|_F$"),
                      ("holonomy_wilson", _CB[1], r"Wilson $1-\mathrm{Re}\,\mathrm{Tr}\,H/K$")):
        if k in per_layer:
            axes[0][1].plot(x, g(k), _m, color=c, lw=1.4, label=lab)
    axes[0][1].set(xticks=x, xlabel="layer", ylabel="deviation", title="Holonomy / curvature by layer")
    axes[0][1].legend(fontsize=8, frameon=False)
    for k, c, lab in (("gauge_trace_spread", _CB[0], "tr spread"),
                      ("gauge_invariant_spread", _CB[1], "group-invariant spread"),
                      ("phi_norm_mean", _CB[2], r"mean $\|\phi\|$")):
        if k in per_layer:
            axes[1][0].plot(x, g(k), _m, color=c, lw=1.4, label=lab)
    axes[1][0].set(xticks=x, xlabel="layer", ylabel="magnitude", title="Gauge action by layer")
    axes[1][0].legend(fontsize=8, frameon=False)
    ax = axes[1][1]
    if "effective_rank" in per_layer:
        ax.plot(x, g("effective_rank"), _m, color=_CB[0], lw=1.4, label="effective rank")
    if "attn_entropy" in per_layer:
        ax.plot(x, g("attn_entropy"), ("s--" if L > 1 else "s"), color=_CB[2], lw=1.2, label="attn entropy (nats)")
    ax.set(xticks=x, xlabel="layer", ylabel="rank / nats", title="Belief geometry by layer")
    if "belief_cond_median" in per_layer:
        axr = ax.twinx()
        axr.plot(x, g("belief_cond_median"), ("^:" if L > 1 else "^"), color=_CB[1], lw=1.2, label="cond median")
        axr.set_ylabel("condition number")
        axr.set_yscale("log")
        axr.grid(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("belief_spectrum")
def plot_belief_spectrum(
    sigma:     object,                   # (N, K) diagonal OR (N, K, K) full converged covariances

    *,
    eps:       float = 1e-6,
    sigma_max: float = 5.0,
    diagonal:  Optional[bool] = None,
    path:      Optional[str]  = None,
):
    r"""F9: belief covariance geometry -- effective rank, guarded scree, and conditioning.

    Panel A: the per-token effective-rank violin (the distribution the logged mean discards).
    Panel B: the eigenvalue scree (median + 10-90 band, log axis) with the eps / sigma_max
    retraction guard lines, exposing whether apparent rank collapse rides the floor. Panel C: the
    per-token spectral condition-number histogram.
    """
    from vfe3 import metrics
    sp = metrics.belief_spectrum(sigma, diagonal=diagonal)
    eig = _np(sp["eigenvalues"])                                 # (N, K) descending
    rank = _np(sp["effective_rank"])
    cond = _np(sp["condition"])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    axes[0].violinplot([rank], showmeans=True)
    axes[0].set(xticks=[1], xticklabels=["beliefs"], ylabel="effective rank",
                title="Per-token effective rank")
    kk = np.arange(1, eig.shape[1] + 1)
    _median_band(axes[1], kk, eig.T, _CB[0], "eigenvalue")
    axes[1].axhline(eps, color=_CB[1], ls="--", lw=1, label=r"$\varepsilon$ floor")
    axes[1].axhline(sigma_max, color=_CB[2], ls="--", lw=1, label=r"$\sigma_{\max}$ ceiling")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="eigenvalue index", ylabel="eigenvalue", title="Guarded spectrum (scree)")
    axes[1].legend(fontsize=8, frameon=False)
    axes[2].hist(cond, bins=24, color=_CB[0], alpha=0.85)
    axes[2].set(xlabel=r"condition number $\lambda_{\max}/\lambda_{\min}$", ylabel="tokens",
                title="Belief conditioning")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("spd_ellipses")
def plot_spd_ellipses(
    mu:       object,                    # (N, K) belief means
    sigma:    object,                    # (N, K) diagonal OR (N, K, K) full covariances

    *,
    dims:     tuple = (0, 1),
    diagonal: Optional[bool] = None,
    path:     Optional[str]  = None,
):
    r"""F9 (companion): correlation-bearing SPD covariance ellipses, coloured by effective rank.

    For a full covariance the ellipse orientation and axes come from the eigendecomposition of the
    2x2 coordinate sub-block (the old ellipse used diagonal variances only, showing no correlation);
    a diagonal belief degrades to axis-aligned. Facecolor encodes per-token effective rank.
    """
    from matplotlib.patches import Ellipse
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from vfe3 import metrics
    m = _np(mu)
    s = _np(sigma)
    a, b = dims
    is_full = s.ndim >= 3 and s.shape[-1] == s.shape[-2] if diagonal is None else (not diagonal)
    rank = _np(metrics.effective_rank_per_token(sigma, diagonal=diagonal))
    norm = Normalize(vmin=rank.min(), vmax=rank.max() + 1e-12)
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(5, 4.4))
    for i in range(m.shape[0]):
        if is_full:
            sub = s[i][np.ix_([a, b], [a, b])]
            w, v = np.linalg.eigh(0.5 * (sub + sub.T))
            ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
            width, height = 2 * np.sqrt(np.clip(w[1], 0, None)), 2 * np.sqrt(np.clip(w[0], 0, None))
        else:
            ang, width, height = 0.0, 2 * np.sqrt(s[i, a]), 2 * np.sqrt(s[i, b])
        ax.add_patch(Ellipse((m[i, a], m[i, b]), width=width, height=height, angle=ang,
                             alpha=0.35, facecolor=cmap(norm(rank[i])), edgecolor="#26456E", lw=0.5))
    ax.scatter(m[:, a], m[:, b], s=8, color="#26456E")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8, label="effective rank")
    ax.set(title="Belief SPD covariance ellipses", xlabel=f"dim {a}", ylabel=f"dim {b}")
    ax.autoscale_view()
    return _save(fig, path)


@register_figure("holonomy_curvature")
def plot_holonomy_curvature(
    flat:      Dict,                     # holonomy_deviation_sampled output for a flat run
    regime:    Optional[Dict] = None,    # ... for a regime_ii run

    *,
    curvature: Optional[object] = None,  # (N, N) curvature_field for a fixed anchor
    path:      Optional[str]    = None,
):
    r"""F10: holonomy / curvature with corrected sampling.

    Panel A: per-triangle holonomy violins -- the flat cocycle spikes at eps (a flatness
    certificate), regime_ii spreads to genuine curvature. Panel B: holonomy vs triangle span.
    Panel C (optional): the regime_ii spatial curvature field for a fixed anchor.
    """
    ncol = 2 + (1 if curvature is not None else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 3.7), squeeze=False)
    ax = axes[0][0]
    data = [np.clip(_np(flat["per_triple"]), 1e-12, None)]
    ticks = ["flat"]
    if regime is not None:
        data.append(np.clip(_np(regime["per_triple"]), 1e-12, None))
        ticks.append("regime II")
    ax.violinplot(data, showmeans=True)
    ax.set_yscale("log")
    ax.set(xticks=range(1, len(ticks) + 1), xticklabels=ticks,
           ylabel=r"$\|H_{ijk}-I\|_F$", title="Triangle holonomy")
    axb = axes[0][1]
    axb.scatter(_np(flat["span"]), np.clip(_np(flat["per_triple"]), 1e-12, None),
                s=10, color=_CB[0], alpha=0.5, label="flat")
    if regime is not None:
        axb.scatter(_np(regime["span"]), np.clip(_np(regime["per_triple"]), 1e-12, None),
                    s=10, color=_CB[1], alpha=0.5, label="regime II")
        axb.legend(fontsize=8, frameon=False)
    axb.set_yscale("log")
    axb.set(xlabel="triangle span max|i-j|", ylabel=r"$\|H_{ijk}-I\|_F$", title="Curvature vs separation")
    if curvature is not None:
        im = axes[0][2].imshow(_np(curvature), cmap="magma", aspect="auto")
        fig.colorbar(im, ax=axes[0][2], shrink=0.8, label=r"$\|H-I\|_F$")
        axes[0][2].set(xlabel="j", ylabel="i", title="Curvature field (regime II)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("capacity_scaling")
def plot_capacity_scaling(
    scaling: Dict,                       # {axis_name: {x, bpc, [lo, hi]}}

    *,
    path:    Optional[str] = None,
):
    r"""F11 Panel A: capacity scaling of val BPC vs each structural axis (K / heads / layers).

    Each sub-panel shares the BPC axis; a bootstrap-over-validation-SEQUENCES band (NOT a cross-seed
    CI -- single-seed protocol) is drawn when ``lo`` / ``hi`` are supplied.
    """
    keys = list(scaling)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        d = scaling[key]
        x, bpc = _np(d["x"]), _np(d["bpc"])
        order = np.argsort(x)
        ax.plot(x[order], bpc[order], "o-", color=_CB[0], lw=1.8)
        if "lo" in d and "hi" in d:
            ax.fill_between(x[order], _np(d["lo"])[order], _np(d["hi"])[order], color=_CB[0], alpha=0.2)
        ax.set(xlabel=key, ylabel="val BPC", title=f"Scaling vs {key}")
    fig.suptitle("Capacity scaling (within-run bootstrap band)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("estep_capacity")
def plot_estep_capacity(
    n_e_steps:    object,                # (T,) E-step iteration counts
    bpc:          object,                # (T,) val BPC
    free_energy:  object,                # (T,) converged free energy

    *,
    n_params:     Optional[int]   = None,
    wall_time:    Optional[object] = None,
    path:         Optional[str]   = None,
):
    r"""F11 Panel B: E-step-as-capacity -- more inner free-energy minimization lowers loss at flat params.

    val BPC and converged F vs the number of E-step iterations on a twin axis (the controlled
    intervention: parameters are constant, only inference depth changes). An optional wall-time inset
    shows the compute cost.
    """
    x = _np(n_e_steps)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(x, _np(bpc), "o-", color=_CB[0], lw=2, label="val BPC")
    ax.set(xlabel="E-step iterations (n_e_steps)", ylabel="val BPC")
    ax2 = ax.twinx()
    ax2.plot(x, _np(free_energy), "s--", color=_CB[1], lw=2)
    ax2.set_ylabel("converged F (nats)", color=_CB[1])
    note = "capacity from inference" + (f"; params flat at {n_params:,}" if n_params else "")
    ax.set_title(f"E-step-as-capacity ({note})")
    if wall_time is not None:
        ins = ax.inset_axes([0.6, 0.6, 0.36, 0.34])
        ins.plot(x, _np(wall_time), "o-", color="#888888", ms=3)
        ins.set(title="wall time (s)", xlabel="n_e_steps")
        ins.tick_params(labelsize=6)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("pareto_frontier")
def plot_pareto_frontier(
    points: Dict,                        # {bpc, n_params, [wall_time], [label]}

    *,
    path:   Optional[str] = None,
):
    r"""F11 Panel C: quality-vs-cost Pareto frontier of val BPC against parameters and wall time.

    Non-dominated cells (lower BPC at lower cost) are connected into a stepwise frontier; dominated
    points are faded.
    """
    bpc = _np(points["bpc"])
    npar = _np(points["n_params"])
    have_time = "wall_time" in points
    ncol = 2 if have_time else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.0), squeeze=False)

    def _draw(ax, cost, xlabel, logx):
        order = np.argsort(cost)
        best = np.inf
        front = np.zeros(cost.size, dtype=bool)
        for k in order:
            if bpc[k] < best:
                best = bpc[k]; front[k] = True
        ax.scatter(cost[~front], bpc[~front], s=24, color="#bbbbbb", label="dominated")
        ax.scatter(cost[front], bpc[front], s=40, color=_CB[1], label="frontier")
        fo = order[np.isin(order, np.where(front)[0])]
        ax.plot(cost[fo], bpc[fo], "-", color=_CB[1], lw=1.5)
        if logx:
            ax.set_xscale("log")
        ax.set(xlabel=xlabel, ylabel="val BPC")
        ax.legend(fontsize=8, frameon=False)

    _draw(axes[0][0], npar, "parameters", True)
    axes[0][0].set_title("Quality vs parameters")
    if have_time:
        _draw(axes[0][1], _np(points["wall_time"]), "wall time (s)", False)
        axes[0][1].set_title("Quality vs compute")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("ablation_forest")
def plot_ablation_forest(
    rows: list,                          # [{label, delta, [lo], [hi]}] delta-BPC vs the full model

    *,
    path: Optional[str] = None,
):
    r"""F12 Panel A: baseline-ladder ablation forest of delta-BPC from the full model.

    Each disabling ablation is a point with a paired bootstrap-over-tokens interval (single-seed
    within-run, not a cross-seed CI), sorted by effect size; the zero line is the full model.
    """
    rows = sorted(rows, key=lambda r: r["delta"])
    y = np.arange(len(rows))
    delta = np.array([r["delta"] for r in rows])
    lo = np.array([r.get("lo", r["delta"]) for r in rows])
    hi = np.array([r.get("hi", r["delta"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(rows) + 1.5))
    ax.errorbar(delta, y, xerr=[delta - lo, hi - delta], fmt="o", color=_CB[0],
                ecolor="#888888", capsize=3)
    ax.axvline(0.0, color="#444444", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set(xlabel=r"$\Delta$ BPC vs full model", title="Component ablations (paired token bootstrap)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("lr_grid_heatmap")
def plot_lr_grid_heatmap(
    grid: Dict,                          # {x, y, z (len(y), len(x)), xlabel, ylabel, [baseline]}

    *,
    path: Optional[str] = None,
):
    r"""F12 Panel B: 2-D joint learning-rate sweep heatmap (val PPL), exposing ridge interactions.

    The basin minimum is starred and the baseline operating point marked, proving the tuned point is
    a joint minimum rather than a 1-D-slice artifact.
    """
    x, y, z = _np(grid["x"]), _np(grid["y"]), _np(grid["z"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.pcolormesh(x, y, z, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, label="val PPL")
    iy, ix = np.unravel_index(np.nanargmin(z), z.shape)
    ax.scatter([x[ix]], [y[iy]], marker="*", s=160, color="white", edgecolor="black", label="min")
    if "baseline" in grid:
        bx, by = grid["baseline"]
        ax.scatter([bx], [by], marker="o", s=60, facecolor="none", edgecolor="red", label="baseline")
    ax.set(xlabel=grid.get("xlabel", "lr x"), ylabel=grid.get("ylabel", "lr y"),
           title="Joint LR sensitivity")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("numerical_trust")
def plot_numerical_trust(
    guard:  Dict,                        # guard_saturation output
    health: Dict,                        # numerical_health output
    causal: Optional[Dict] = None,       # causal_sanity output

    *,
    path:   Optional[str] = None,
):
    r"""F13: numerical-trust panel -- guard saturation, finiteness, and causal correctness.

    Panel A: the fraction of converged entries pinned at each numerical guard boundary, with a 1%
    reference line. Panel B: the non-finite fraction of each intermediate. Panel C (optional):
    per-head future-attention leakage (must be ~0 under the causal mask).
    """
    ncol = 3 if causal is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 3.6), squeeze=False)
    gk = list(guard)
    axes[0][0].bar(range(len(gk)), [float(guard[k]) for k in gk], color=_CB[0])
    axes[0][0].axhline(0.01, color=_CB[1], ls="--", lw=1, label="1%")
    axes[0][0].set(xticks=range(len(gk)), ylabel="fraction pinned", title="Guard saturation")
    axes[0][0].set_xticklabels(gk, rotation=30, ha="right", fontsize=7)
    axes[0][0].legend(fontsize=8, frameon=False)
    hk = [k for k in health if k.startswith("nan")]
    axes[0][1].bar(range(len(hk)), [float(health[k]) for k in hk], color=_CB[2])
    axes[0][1].set(xticks=range(len(hk)), ylabel="non-finite fraction", title="Numerical health")
    axes[0][1].set_xticklabels(hk, rotation=30, ha="right", fontsize=7)
    if causal is not None:
        leak = _np(causal["future_leakage"])
        if leak.ndim == 2:                                       # (L, H) -> label each bar by (layer, head)
            lab = [f"L{l}H{hh}" for l in range(leak.shape[0]) for hh in range(leak.shape[1])]
            leak = leak.ravel()
        else:                                                    # (H,) single layer -> plain head labels
            leak = leak.ravel()
            lab = [f"h{hh}" for hh in range(leak.size)]
        axes[0][2].bar(range(leak.size), leak, color=_CB[3])
        axes[0][2].set(xticks=range(leak.size), xlabel="layer-head", ylabel="max future-attention",
                       title="Causal leakage (expect 0)")
        axes[0][2].set_xticklabels(lab, rotation=(90 if leak.size > 3 else 0), fontsize=6)
    fig.tight_layout()
    return _save(fig, path)


# ===========================================================================
# Scaling-law figures (x-axis = number of parameters N; y = test CE in nats/token).
# Consume the cross-run aggregate produced by scaling_analysis.py: a list of
# per-point dicts {x_key: N, "ce_seeds": [per-seed test_ce], "route", "label"}.
# The shared power-law fit lives here (_fit_power_law) so both the figures and the
# analyzer tables use one implementation.
# ===========================================================================

def _fit_power_law(
    N:       object,                     # (P,) parameter counts (or tokens / FLOPs)
    L:       object,                     # (P,) loss at each point (test CE)

    *,
    weights:     object = None,          # (P,) WLS weights (e.g. (L/SEM)^2); None -> ordinary LS
    with_offset: bool   = False,         # True -> Chinchilla L = E + A N^-alpha (scipy); else log-log power law
) -> Dict[str, object]:
    r"""Fit ``L(N)``. Default: log-log (weighted) least squares ``L = A * N^{-alpha}`` (returns
    ``alpha = -slope``); ``with_offset=True`` fits the Chinchilla irreducible-loss form
    ``L = E + A * N^{-alpha}`` via ``scipy.optimize.curve_fit`` and silently falls back to the
    log-log fit if scipy is absent or the solve fails. Never raises; degenerate input (<2 finite
    positive points) returns NaNs so the caller can skip the overlay."""
    N = _np(N).astype(float).reshape(-1)
    L = _np(L).astype(float).reshape(-1)
    m = np.isfinite(N) & np.isfinite(L) & (N > 0) & (L > 0)
    N, L = N[m], L[m]
    out: Dict[str, object] = {"alpha": float("nan"), "A": float("nan"), "E": 0.0,
                              "r2": float("nan"), "n_points": int(N.size), "form": "power_law"}
    if N.size < 2:
        return out
    x, y = np.log(N), np.log(L)
    w = None if weights is None else _np(weights).astype(float).reshape(-1)[m]
    if with_offset and N.size >= 3:
        try:
            from scipy.optimize import curve_fit
            p0 = [max(0.0, float(L.min()) * 0.5), float(np.exp(np.mean(y))), 0.3]
            popt, _ = curve_fit(lambda n, E, A, al: E + A * np.power(n, -al), N, L, p0=p0,
                                bounds=([0.0, 1e-12, 1e-3], [float(L.min()), np.inf, 3.0]),
                                maxfev=20000)
            E, A, al = (float(v) for v in popt)
            Lp = E + A * np.power(N, -al)
            ss_res = float(np.sum((L - Lp) ** 2))
            ss_tot = float(np.sum((L - L.mean()) ** 2))
            return {"alpha": al, "A": A, "E": E, "n_points": int(N.size), "form": "offset_power_law",
                    "r2": (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))}
        except Exception:
            pass                                                # fall through to the log-log fit
    import warnings
    with warnings.catch_warnings():                          # bootstrap resamples can repeat x's -> ill-conditioned
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        coef = np.polyfit(x, y, 1, w=(None if w is None else np.sqrt(np.clip(w, 0.0, None))))
    slope, intercept = float(coef[0]), float(coef[1])
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out.update({"alpha": -slope, "A": float(np.exp(intercept)),
                "r2": (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))})
    return out


def _t95(n: int) -> float:
    r"""Two-sided 95% t-multiplier for ``n`` samples (``n-1`` dof); 0 for n<2, 1.96 if scipy absent."""
    if n < 2:
        return 0.0
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1))
    except Exception:
        return 1.96


def _scaling_point_stats(points: list, x_key: str = "n_params") -> Dict[str, np.ndarray]:
    r"""Per-point seed aggregation: returns x, mean, sem, ci95, n (one entry per point with >=1 finite
    seed) plus the flattened per-seed (seed_x, seed_y) cloud for the faint scatter."""
    x, mean, sem, ci, ns, sx, sy = [], [], [], [], [], [], []
    for p in points:
        seeds = [float(s) for s in p.get("ce_seeds", []) if np.isfinite(s)]
        if not seeds or not np.isfinite(p.get(x_key, float("nan"))):
            continue
        arr = np.asarray(seeds, dtype=float)
        xv = float(p[x_key])
        x.append(xv); mean.append(float(arr.mean())); ns.append(arr.size)
        s = float(arr.std(ddof=1)) / np.sqrt(arr.size) if arr.size > 1 else 0.0
        sem.append(s); ci.append(_t95(arr.size) * s)
        sx.extend([xv] * arr.size); sy.extend(seeds)
    return {"x": np.asarray(x), "mean": np.asarray(mean), "sem": np.asarray(sem),
            "ci95": np.asarray(ci), "n": np.asarray(ns, dtype=int),
            "seed_x": np.asarray(sx), "seed_y": np.asarray(sy)}


@register_figure("scaling_law")
def plot_scaling_law(
    points: list,                        # [{n_params, ce_seeds:[...], route, label}, ...]

    *,
    x_key:       str  = "n_params",
    xlabel:      str  = "parameters N",
    with_offset: bool = False,
    title:       str  = "Scaling law (test CE vs parameters)",
    path:        Optional[str] = None,
):
    r"""Headline scaling figure: log-log ``test_ce`` vs ``N`` with per-seed points, cross-seed 95% CI
    error bars, the fitted power law overlaid, and a residual subpanel.

    The fit is a (SEM-weighted) log-log least squares ``L = A N^{-alpha}`` (or the Chinchilla
    ``E + A N^{-alpha}`` when ``with_offset``); the exponent ``alpha`` and ``R^2`` are annotated.
    The residual panel plots ``log(mean / fit)`` against ``log N`` -- systematic curvature there is
    the signature of a pre-asymptotic (not-yet-power-law) regime. The point count is annotated so a
    two-or-three-size fit is never read as more precise than it is."""
    st = _scaling_point_stats(points, x_key)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(5.8, 5.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    if st["seed_x"].size:
        ax.scatter(st["seed_x"], st["seed_y"], s=14, color=_CB[0], alpha=0.22, label="per seed", zorder=2)
    if st["x"].size:
        ax.errorbar(st["x"], st["mean"], yerr=st["ci95"], fmt="o", color=_CB[1], ms=6,
                    capsize=3, lw=0, elinewidth=1.2, label="mean (95% CI)", zorder=3)
    fit = None
    if st["x"].size >= 2:
        w = np.where(st["sem"] > 0, (st["mean"] / np.where(st["sem"] > 0, st["sem"], 1.0)) ** 2, 1.0)
        fit = _fit_power_law(st["x"], st["mean"], weights=w, with_offset=with_offset)
        xx = np.geomspace(st["x"].min(), st["x"].max(), 100)
        yy = fit["E"] + fit["A"] * np.power(xx, -fit["alpha"])
        lbl = (rf"fit $\alpha$={fit['alpha']:.3f}, $R^2$={fit['r2']:.3f}"
               + (f", E={fit['E']:.3f}" if with_offset else ""))
        ax.plot(xx, yy, "--", color="#444444", lw=1.6, label=lbl, zorder=4)
        yfit_pts = fit["E"] + fit["A"] * np.power(st["x"], -fit["alpha"])
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = np.log(st["mean"] / yfit_pts)
        axr.axhline(0.0, color="#444444", ls="--", lw=1)
        axr.scatter(st["x"], resid, s=28, color=_CB[1])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(ylabel="test CE (nats/token)", title=f"{title}  (P={int(st['x'].size)} sizes)")
    ax.legend(fontsize=8, frameon=False)
    axr.set_xscale("log")
    axr.set(xlabel=xlabel, ylabel=r"log(obs/fit)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("scaling_routes")
def plot_scaling_routes(
    points: list,                        # [{n_params, ce_seeds:[...], route, label}, ...]

    *,
    x_key:  str = "n_params",
    xlabel: str = "parameters N",
    title:  str = "Routes to N: does the frontier collapse?",
    path:   Optional[str] = None,
):
    r"""Multi-route overlay: ``test_ce`` vs ``N`` with points colored AND markered by which knob grew
    ``N`` (embed_dim / gauge block size / depth / ...), each route's own power-law fit, and a dashed
    pooled fit. Routes that fall on one line share a frontier; a route offset above/below it (or a
    different slope) does not -- the visual companion to the analyzer's ANCOVA test. Marker shape
    duplicates the color so the routes survive greyscale printing."""
    routes = sorted({str(p.get("route", "?")) for p in points})
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for i, r in enumerate(routes):
        st = _scaling_point_stats([p for p in points if str(p.get("route", "?")) == r], x_key)
        if not st["x"].size:
            continue
        c, mk = _CB[i % len(_CB)], markers[i % len(markers)]
        ax.errorbar(st["x"], st["mean"], yerr=st["ci95"], fmt=mk, color=c, ms=6, capsize=2,
                    lw=0, elinewidth=1.0)
        if st["x"].size >= 2:
            fit = _fit_power_law(st["x"], st["mean"])
            xx = np.geomspace(st["x"].min(), st["x"].max(), 60)
            ax.plot(xx, fit["A"] * np.power(xx, -fit["alpha"]), color=c, lw=1.4,
                    label=rf"{r} ($\alpha$={fit['alpha']:.3f})")
        else:
            ax.plot([], [], marker=mk, color=c, lw=0, label=f"{r} (1 pt)")
    stall = _scaling_point_stats(points, x_key)
    if stall["x"].size >= 2:
        g = _fit_power_law(stall["x"], stall["mean"])
        xx = np.geomspace(stall["x"].min(), stall["x"].max(), 60)
        ax.plot(xx, g["A"] * np.power(xx, -g["alpha"]), "--", color="#444444", lw=1.6,
                label=rf"pooled ($\alpha$={g['alpha']:.3f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlabel=xlabel, ylabel="test CE (nats/token)", title=title)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("inference_capacity")
def plot_inference_capacity(
    series: Dict,                        # {knob_name: [{x, ce_seeds:[...]}, ...]} at FLAT params

    *,
    n_params: Optional[int] = None,
    title:    str = "Inference-compute capacity (flat N)",
    path:     Optional[str] = None,
):
    r"""Flat-``N`` inference-compute frontier: ``test_ce`` vs each inference depth knob (``n_e_steps``,
    ``n_layers``) at CONSTANT parameter count -- the architecture's unique capacity-from-inference
    axis. One panel per knob, x linear, mean +/- 95% CI over seeds. Kept SEPARATE from the ``L(N)``
    figures because depth/T add zero parameters and would make an ``L(N)`` curve spuriously
    multi-valued at fixed ``N``."""
    keys = [k for k in series if series[k]]
    if not keys:
        keys = list(series)
    fig, axes = plt.subplots(1, max(1, len(keys)), figsize=(4.6 * max(1, len(keys)), 3.8),
                             squeeze=False)
    for ax, k in zip(axes[0], keys):
        st = _scaling_point_stats(series[k], x_key="x")
        order = np.argsort(st["x"]) if st["x"].size else np.array([], dtype=int)
        if order.size:
            ax.errorbar(st["x"][order], st["mean"][order], yerr=st["ci95"][order], fmt="o-",
                        color=_CB[0], capsize=3, lw=1.6)
        ax.set(xlabel=k, ylabel="test CE (nats/token)", title=f"vs {k}")
    note = f"params flat at {n_params:,}" if n_params else "params flat"
    fig.suptitle(f"{title} ({note})")
    fig.tight_layout()
    return _save(fig, path)


# ===========================================================================
# Vocabulary next-token probability figures. Each takes a LIST of arms -- one
# arm (the single-run pipeline) renders one column; two arms (the cross-run
# driver, e.g. K70 vs K120) render side-by-side panels for a collapse contrast.
# Data come from vfe3.viz.extract.vocab_prediction_stats / decode_readout.
# ===========================================================================


def _tok_label(decode, tid) -> str:
    r"""A short printable label for vocab id ``tid`` (decoded text, or the id with no decoder)."""
    if decode is None:
        return str(int(tid))
    try:
        s = decode([int(tid)])
    except Exception:
        return str(int(tid))
    if s == "" or s.isspace():
        return {" ": "·", "\n": "\\n", "\t": "\\t"}.get(s, "␣")  # show whitespace visibly
    s = s.replace("\n", "\\n")
    return s if len(s) <= 12 else s[:11] + "…"


@register_figure("vocab_probability_heatmap")
def plot_vocab_probability_heatmap(
    arms,                                # list of vocab_prediction_stats dicts (+ "label")

    *,
    decode: Optional[Callable] = None,
    cmap:   str = "magma",
    path:   Optional[str]      = None,
):
    r"""Seq x top-k heatmap of ``p(o_{n+1} | o_{<=n})``: rows are the most-probable tokens, columns
    the sequence positions, a green box marks the true next token. Bright horizontal bands (the same
    tokens lit at every position regardless of context) are the visual signature of prior collapse;
    a context-tracking model puts the bright cell on the boxed ground truth. One column per arm."""
    arms = list(arms)
    n = max(1, len(arms))
    R = max((int(_np(a["disp_probs"]).shape[0]) for a in arms), default=1)
    P = max((int(_np(a["disp_probs"]).shape[1]) for a in arms), default=1)
    fig, axes = plt.subplots(1, n, figsize=(max(4.5, 0.13 * P) * n, max(3.6, 0.17 * R)), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        prob = _np(arm["disp_probs"])                            # (R, P)
        rows = _np(arm["row_ids"]).astype(int)
        truth = _np(arm["disp_truth_row"]).astype(int)
        pos = prob[prob > 0]
        vmax = float(prob.max()) if prob.size else 1.0
        vmin = float(pos.min()) if pos.size else vmax * 1e-4
        vmin = max(vmin, vmax * 1e-4, 1e-12)
        im = ax.imshow(prob, aspect="auto", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.1)))
        ax.set_yticks(range(prob.shape[0]))
        ax.set_yticklabels([_tok_label(decode, t) for t in rows], fontsize=6)
        ax.set_xlabel("sequence position")
        for p in range(min(prob.shape[1], truth.shape[0])):
            r = int(truth[p])
            if r >= 0:
                ax.add_patch(plt.Rectangle((p - 0.5, r - 0.5), 1.0, 1.0, fill=False,
                                           edgecolor="#00E000", lw=0.7, zorder=5))
        ax.set_title(str(arm.get("label", "")))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="P(token | context)")
    fig.suptitle("Next-token vocabulary probabilities (rows = top predicted; green box = true next token)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("vocab_calibration")
def plot_vocab_calibration(
    arms,                                # list of vocab_prediction_stats dicts (+ "label")

    *,
    decode: Optional[Callable] = None,   # unused; kept for a uniform arm-figure signature
    path:   Optional[str]      = None,
):
    r"""Per-token mean predicted probability vs empirical next-token unigram frequency (log-log
    hexbin), with the ``y=x`` marginal-predictor reference. Mass hugging ``y=x`` together with a
    context-information gain ``H_uni - H_pred`` near zero means the model has collapsed to the
    marginal and is ignoring context. One column per arm."""
    arms = list(arms)
    n = max(1, len(arms))
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.3), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        uni = _np(arm["unigram"])
        mp = _np(arm["mean_pred_prob"])
        m = uni > 0
        x = np.log10(uni[m])
        y = np.log10(np.clip(mp[m], 1e-12, None))
        if x.size:
            hb = ax.hexbin(x, y, gridsize=40, bins="log", cmap="viridis", mincnt=1)
            fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02, label="tokens (log)")
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            ax.plot([lo, hi], [lo, hi], color=_CB[1], lw=1.2, ls="--", label="y=x (marginal predictor)")
            ax.legend(fontsize=7, frameon=False, loc="best")
        hp = float(arm.get("mean_pred_entropy", float("nan")))
        hu = float(arm.get("unigram_entropy", float("nan")))
        ax.set_xlabel("log10 empirical next-token freq")
        ax.set_ylabel("log10 mean predicted prob")
        ax.set_title(f"{arm.get('label', '')}  Hpred={hp:.2f}  Huni={hu:.2f}  gain={hu - hp:.2f}")
    fig.suptitle("Vocabulary calibration: predicted prob vs unigram (collapse -> mass on y=x, gain -> 0)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("vocab_confusion")
def plot_vocab_confusion(
    arms,                                # list of vocab_prediction_stats dicts (+ "label")

    *,
    decode:   Optional[Callable] = None,
    taxonomy: str = "function_content",
    path:     Optional[str]      = None,
):
    r"""Row-normalized next-token category confusion: true category (rows) vs argmax-predicted
    category (columns), bucketed by ``taxonomy`` (the 50257-token vocab is uncountable as a raw
    confusion matrix). Mass concentrating in a single predicted column -- everything predicted as
    one frequent category -- is collapse. Needs a decoder for the category bucketing. One column
    per arm."""
    if decode is None:
        raise ValueError("plot_vocab_confusion needs a token decoder for category bucketing")
    arms = list(arms)
    n = max(1, len(arms))
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.5), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        true_ids = _np(arm["true_ids"]).astype(int)
        pred_ids = _np(arm["pred_ids"]).astype(int)
        tcat, names = _token_category_labels(true_ids, decode, taxonomy)
        pcat, _ = _token_category_labels(pred_ids, decode, taxonomy)
        C = len(names)
        M = np.zeros((C, C), dtype=float)
        np.add.at(M, (tcat.astype(int), pcat.astype(int)), 1.0)
        Mn = M / np.clip(M.sum(axis=1, keepdims=True), 1.0, None)
        im = ax.imshow(Mn, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(C)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(C)); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("predicted category"); ax.set_ylabel("true next-token category")
        ax.set_title(str(arm.get("label", "")))
        for i in range(C):
            for j in range(C):
                ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if Mn[i, j] < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="P(pred cat | true cat)")
    fig.suptitle(f"Next-token category confusion ({taxonomy}, row-normalized)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("decode_readout")
def plot_decode_readout(
    arms,                                # list of decode_readout dicts (+ "label")

    *,
    decode: Optional[Callable] = None,
    path:   Optional[str]      = None,
):
    r"""The linear-decode readout matrix ``W`` (``logits = mu_q @ W^T``): the top-norm vocabulary
    rows over the ``K`` latent dimensions, diverging colormap centered at zero. Structure here is
    the learned output embedding; a near-flat or low-rank readout under-uses the latent space. One
    column per arm."""
    arms = list(arms)
    n = max(1, len(arms))
    R = max((int(_np(a["weight"]).shape[0]) for a in arms), default=1)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, max(4.2, 0.06 * R)), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        W = _np(arm["weight"])                                   # (R, K)
        rows = _np(arm["row_ids"]).astype(int)
        mag = float(np.abs(W).max()) or 1.0
        im = ax.imshow(W, cmap="coolwarm", vmin=-mag, vmax=mag, aspect="auto")
        step = max(1, W.shape[0] // 40)
        ax.set_yticks(range(0, W.shape[0], step))
        ax.set_yticklabels([_tok_label(decode, rows[i]) for i in range(0, W.shape[0], step)], fontsize=6)
        ax.set_xlabel("latent dimension k")
        ax.set_title(str(arm.get("label", "")))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="W[v, k]")
    fig.suptitle("Linear decode readout W (top-norm vocab rows): logits = mu_q @ W^T")
    fig.tight_layout()
    return _save(fig, path)


# ===========================================================================
# B1 / EXP-3  --  Sigma_q as calibrated Fisher uncertainty
# ===========================================================================

def _spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    r"""Spearman rank correlation of two 1-D arrays (Pearson on ranks); 0.0 if degenerate."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if x.size < 2:
        return 0.0
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float((rx * ry).sum() / den) if den > 0 else 0.0


@register_figure("reliability_diagram")
def plot_reliability_diagram(
    reliability,                         # list of {conf, acc, frac} bins (run_artifacts._calibration_and_strata)

    *,
    ece:  Optional[float] = None,
    path: Optional[str]   = None,
):
    r"""B1/EXP-3 control: decode-calibration reliability diagram (accuracy vs confidence).

    The uncalibrated decode-softmax confidence baseline the Sigma_q-conditioned recalibration is
    measured against. Each bar is a confidence bin: height = empirical accuracy, the ``y=x`` diagonal
    is perfect calibration, and the gap (over/under-confidence) integrated and frac-weighted is the
    ECE annotated in the title. Consumes the bins ``run_artifacts`` already computes (conf / acc /
    frac) but never plotted."""
    rel = list(reliability or [])
    conf = np.array([b["conf"] for b in rel], float)
    acc = np.array([b["acc"] for b in rel], float)
    frac = np.array([b.get("frac", 0.0) for b in rel], float)
    if ece is None and rel:
        ece = float((frac * np.abs(acc - conf)).sum())
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color=_CB[7], lw=1.2, ls="--", label="perfect calibration (y=x)")
    if conf.size:
        width = 1.0 / max(len(rel), 1)
        ax.bar(conf, acc, width=0.9 * width, color=_CB[0], alpha=0.55,
               edgecolor=_CB[0], label="empirical accuracy")
        ax.scatter(conf, acc, s=8.0 + 240.0 * frac, color=_CB[1], zorder=3,
                   label="bin (area ~ token frac)")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="decode confidence (max softmax prob)",
           ylabel="empirical accuracy")
    ax.set_title("Decode reliability" + (f"  (ECE={ece:.3f})" if ece is not None else ""))
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("sigma_stratified_error")
def plot_sigma_stratified_error(
    bank: Dict,                          # belief_ce_bank: tr_sigma (M,), ce (M,)

    *,
    n_bins:  int = 10,
    n_boot:  int = 400,
    path:    Optional[str] = None,
):
    r"""B1/EXP-3: held-out CE stratified by belief uncertainty tr(Sigma_q) (decile bins).

    The load-bearing calibration figure: if Sigma_q carries decode-time uncertainty, mean CE rises
    monotonically across increasing-tr(Sigma_q) bins. Tokens are split into ``n_bins`` equal-count
    quantile bins of tr(Sigma_q); each point is that bin's mean CE with a bootstrap 10-90 band
    (``n_boot`` resamples, fixed seed -> deterministic). A flat curve means the covariance channel is
    inert (see the CV>0.10 gate)."""
    tr = _np(bank["tr_sigma"]).ravel()
    ce = _np(bank["ce"]).ravel()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    if tr.size >= n_bins:
        edges = np.quantile(tr, np.linspace(0.0, 1.0, n_bins + 1))
        idx = np.clip(np.digitize(tr, edges[1:-1]), 0, n_bins - 1)   # interior edges -> n_bins bins
        rng = np.random.default_rng(0)
        xs, ms, los, his = [], [], [], []
        for b in range(n_bins):
            cb = ce[idx == b]
            if cb.size == 0:
                continue
            xs.append(float(tr[idx == b].mean()))
            ms.append(float(cb.mean()))
            boot = cb[rng.integers(0, cb.size, size=(n_boot, cb.size))].mean(axis=1)
            los.append(float(np.percentile(boot, 10)))
            his.append(float(np.percentile(boot, 90)))
        ax.plot(xs, ms, "o-", color=_CB[0], lw=1.8, label="mean CE per tr(Sigma_q) decile")
        ax.fill_between(xs, los, his, color=_CB[0], alpha=0.2, label="bootstrap 10-90 band")
        rho = _spearman_np(tr, ce)
        ax.set_title(f"CE stratified by belief uncertainty  (Spearman rho={rho:+.3f})")
    else:
        ax.set_title("CE stratified by belief uncertainty (insufficient tokens)")
    ax.set(xlabel=r"belief uncertainty tr$(\Sigma_q)$", ylabel="cross-entropy (nats)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("sigma_ce_scatter")
def plot_sigma_ce_scatter(
    bank: Dict,                          # belief_ce_bank: tr_sigma (M,), ce (M,)

    *,
    path: Optional[str] = None,
):
    r"""B1/EXP-3: per-token tr(Sigma_q) vs CE hexbin with the Spearman rho in the title.

    The raw joint distribution behind the stratified curve -- the headline statistic rho(tr Sigma_q,
    CE) is the rank correlation; a positive tilt means more-uncertain beliefs are harder tokens
    (the calibration channel carries signal)."""
    tr = _np(bank["tr_sigma"]).ravel()
    ce = _np(bank["ce"]).ravel()
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    if tr.size:
        hb = ax.hexbin(tr, ce, gridsize=40, bins="log", cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02, label="tokens (log)")
    rho = _spearman_np(tr, ce) if tr.size >= 2 else float("nan")
    ax.set(xlabel=r"belief uncertainty tr$(\Sigma_q)$", ylabel="cross-entropy (nats)")
    ax.set_title(rf"$\mathrm{{tr}}(\Sigma_q)$ vs CE  (Spearman $\rho$={rho:+.3f}, n={tr.size})")
    fig.tight_layout()
    return _save(fig, path)


# ===========================================================================
# F2 / EXP-7  --  prior-anchoring resists rank collapse
# ===========================================================================

@register_figure("rank_residual_by_depth")
def plot_rank_residual_by_depth(
    arms,                                # dict {label: (L,) r(X)} OR list of {"label", "rank_one_residual"}

    *,
    path: Optional[str] = None,
):
    r"""F2/EXP-7: Dong rank-one residual r(X) vs inference depth, one line per anchoring arm.

    r(X) -> 0 is rank collapse (every token mean identical); a no-anchor arm decays faster per layer
    than an alpha.KL(q||p)-anchored arm if prior-anchoring is the FFN-brake substitute. Each arm's
    log-linear decay rate b (slope of log r(X) vs layer) is annotated in the legend -- the per-arm
    decay-rate comparison the experiment decides on (absolute level is not the endpoint; the
    no-anchor control plateaus rather than collapsing to rank one)."""
    if isinstance(arms, dict):
        items = [(str(k), _np(v).ravel()) for k, v in arms.items()]
    else:
        items = [(str(a.get("label", i)), _np(a["rank_one_residual"]).ravel())
                 for i, a in enumerate(arms)]
    items = [(lab, r) for lab, r in items if r.size >= 1]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for j, (lab, r) in enumerate(items):
        x = np.arange(1, r.size + 1)
        slope = ""
        if r.size >= 2 and float(np.min(r)) > 0:
            b = float(np.polyfit(np.arange(r.size), np.log(r), 1)[0])
            slope = f"  (b={b:+.3f})"
        ax.plot(x, r, "o-" if r.size > 1 else "o", color=_CB[j % len(_CB)], lw=1.6, label=f"{lab}{slope}")
    ax.set(xlabel="inference depth (layer)",
           ylabel=r"rank-one residual r(X) = $\|X-\mathbf{1}\bar x^T\|_F/\|X\|_F$")
    ax.set_title("Prior-anchoring vs rank collapse (Dong r(X) by depth)")
    if items:
        ax.set_xticks(np.arange(1, max(r.size for _, r in items) + 1))
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


# ===========================================================================
# C2 / EXP-5  --  structural non-Neal-Hinton EM: F-vs-CE decorrelation
# ===========================================================================

@register_figure("f_ce_decorrelation")
def plot_f_ce_decorrelation(
    arms,                                # list of {n_e_steps, final_f, ce, [label]} (one per n_e_steps cell)

    *,
    path: Optional[str] = None,
):
    r"""C2/EXP-5: converged final E-step free energy per token vs held-out CE, one point per n_e_steps.

    The structural non-Neal-Hinton EM prediction: the E-step descends its OWN target-blind functional
    F, NOT the likelihood, so across an n_e_steps sweep final F should fall steeply (Pearson(n_e_steps,
    F) < 0) while staying DECORRELATED from CE (Pearson(F, CE) ~ 0, or even > 0). A strongly NEGATIVE
    Pearson(F, CE) would instead say F tracks the loss, contradicting the EM separation. Points are
    ordered and annotated by n_e_steps; both Pearsons are in the title."""
    pts = sorted(arms, key=lambda a: float(a["n_e_steps"]))
    ne = np.array([float(a["n_e_steps"]) for a in pts])
    ff = np.array([float(a["final_f"]) for a in pts])
    ce = np.array([float(a["ce"]) for a in pts])

    def _pear(a, b):
        return (float(np.corrcoef(a, b)[0, 1])
                if a.size >= 2 and a.std() > 0 and b.std() > 0 else float("nan"))
    r_fce, r_nef = _pear(ff, ce), _pear(ne, ff)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    if ff.size:
        ax.plot(ff, ce, "-", color=_CB[7], lw=1.0, alpha=0.5, zorder=1)
        sc = ax.scatter(ff, ce, c=ne, cmap="viridis", s=90, zorder=3, edgecolor="k", linewidth=0.5)
        for a in pts:
            ax.annotate(f"T={int(float(a['n_e_steps']))}", (float(a["final_f"]), float(a["ce"])),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="n_e_steps")
    ax.set(xlabel="converged final E-step F / token (nats)", ylabel="held-out CE (nats)")
    ax.set_title(rf"E-step F vs CE  (Pearson$(F,CE)$={r_fce:+.3f}, Pearson$(T,F)$={r_nef:+.3f})")
    fig.tight_layout()
    return _save(fig, path)


# ===========================================================================
# C1 / EXP-4  --  canonical-F vs entropy-suppressed surrogate
# ===========================================================================

@register_figure("entropy_ppl_gap")
def plot_entropy_ppl_gap(
    cells,                               # list of {include_attention_entropy(bool), kappa, ppl}

    *,
    path: Optional[str] = None,
):
    r"""C1/EXP-4: validation-PPL, canonical (entropy ON) vs surrogate (entropy OFF), grouped by kappa.

    The production kernel descends the SURROGATE gradient, so CANON_ORACLE is the only path that ever
    exercises the canonical -tau^{-1} Cov_beta correction; a positive (surrogate - canonical) PPL gap
    means the entropy term is empirically load-bearing. Grouped bars per kappa; the gap is annotated."""
    cells = list(cells)
    kappas = sorted({float(c["kappa"]) for c in cells})

    def _ppl(k, ent):
        v = [float(c["ppl"]) for c in cells
             if abs(float(c["kappa"]) - k) < 1e-9 and bool(c["include_attention_entropy"]) == ent]
        return v[0] if v else float("nan")
    canon = [_ppl(k, True) for k in kappas]
    surr = [_ppl(k, False) for k in kappas]
    x = np.arange(len(kappas)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar(x - w / 2, canon, w, color=_CB[0], label="canonical (entropy ON)")
    ax.bar(x + w / 2, surr, w, color=_CB[1], label="surrogate (entropy OFF)")
    for i in range(len(kappas)):
        if np.isfinite(canon[i]) and np.isfinite(surr[i]):
            ax.annotate(f"Δ={surr[i] - canon[i]:+.2f}", (x[i], max(canon[i], surr[i])),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"κ={k:g}" for k in kappas])
    ax.set(ylabel="validation PPL")
    ax.set_title("Canonical-F vs entropy-suppressed surrogate (PPL gap by κ)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("cov_gap_vs_kappa")
def plot_cov_gap_vs_kappa(
    cells,                               # list of {include_attention_entropy(bool), kappa, cov_gap}

    *,
    path: Optional[str] = None,
):
    r"""C1/EXP-4: the -tau^{-1} Cov_beta(E, dE) gradient-gap magnitude vs kappa.

    The belief-gradient contribution the canonical entropy term adds and the surrogate drops. Its
    kappa-dependence trades the 1/tau = 1/(kappa*sqrt(d)) prefactor against the beta diffuseness in
    Cov_beta, so the sign of the trend is empirical -- the plot measures it. One series per trained
    arm (the gap is measured on each arm's converged belief)."""
    cells = list(cells)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for ent, lab, col in ((True, "canonical-trained", _CB[0]), (False, "surrogate-trained", _CB[1])):
        pts = sorted([(float(c["kappa"]), float(c["cov_gap"])) for c in cells
                      if bool(c["include_attention_entropy"]) == ent
                      and np.isfinite(float(c["cov_gap"]))], key=lambda t: t[0])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, lw=1.8, label=lab)
    ax.set(xlabel=r"$\kappa$ (softmax temperature scale)",
           ylabel=r"$\|-\tau^{-1}\mathrm{Cov}_\beta(E,\nabla E)\|$ per token")
    ax.set_title("Attention-entropy gradient gap vs κ")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


# ===========================================================================
# D1 / EXP-8  --  pullback natural-gradient gauge M-step convergence
# ===========================================================================

@register_figure("wallclock_convergence")
def plot_wallclock_convergence(
    arms,                                # list of {label, val_ppl:[...], wall_clock_s:[...], step:[...]}

    *,
    target: Optional[float] = None,      # shared target PPL; None -> the worst arm's best (all reach it)
    path:   Optional[str]   = None,
):
    r"""D1/EXP-8: validation PPL vs cumulative wall time, one line per gauge M-step arm.

    The per-wall-clock convergence curve (the per-token pullback matrix_exp solve is the dominant
    added cost, so a per-STEP advantage can vanish per second). A shared target PPL (default: the
    worst arm's best, so every arm reaches it) is drawn; each arm's wall-time and step count TO that
    target are annotated in the legend -- the steps-to-target / wall-to-target convergence-speed
    readout. log-y so multiplicative gaps are legible."""
    arms = [a for a in arms if len(a.get("val_ppl", [])) >= 1 and len(a.get("wall_clock_s", [])) >= 1]
    if target is None and arms:
        mins = [min(a["val_ppl"]) for a in arms if a["val_ppl"]]
        target = max(mins) if mins else None
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for j, a in enumerate(arms):
        t = np.asarray(a["wall_clock_s"], float)
        y = np.asarray(a["val_ppl"], float)
        s = np.asarray(a.get("step", np.arange(t.size)), float)
        order = np.argsort(t)
        t, y, s = t[order], y[order], s[order]
        lab = str(a.get("label", j))
        if target is not None:
            hit = np.where(y <= target)[0]
            if hit.size:
                lab += f"  (→ {t[hit[0]]:.0f}s / {int(s[hit[0]])} steps)"
        ax.plot(t, y, "o-", color=_CB[j % len(_CB)], lw=1.6, ms=4, label=lab)
    if target is not None and np.isfinite(target):
        ax.axhline(target, color=_CB[7], ls="--", lw=1.0, alpha=0.6, label=f"target PPL={target:.2f}")
    ax.set_yscale("log")
    ax.set(xlabel="cumulative wall time (s)", ylabel="validation PPL (log)")
    ax.set_title("Gauge M-step convergence: val PPL vs wall-clock")
    if arms:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)
