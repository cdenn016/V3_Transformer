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
    n_neighbors = min(n_neighbors, max(2, X.shape[0] - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=seed)
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
    M:    np.ndarray,                    # attention weights (any shape)
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> tuple:                              # (vmin, vmax) valid for a LogNorm scale
    r"""Positive-entry (vmin, vmax) for a log attention scale (dynamic range capped at 3 decades).

    Causal-masked future positions are exact zeros (softmax over a -inf prior), so only the
    active (positive) entries set the scale. ``vmin`` is floored three decades below ``vmax`` so a
    few near-zero weights cannot wash a panel out, and kept strictly below ``vmax`` so a uniform
    map stays a valid LogNorm. Pass both bounds to share one scale across several panels.
    """
    if vmax is None:
        pos = M[M > 0]
        vmax = float(pos.max()) if pos.size else 1.0
    if vmin is None:
        pos = M[M > 0]
        vmin = float(pos.min()) if pos.size else vmax * 1e-3
    vmin = max(vmin, vmax * 1e-3)                                 # cap dynamic range at 3 decades
    if vmin >= vmax:                                             # degenerate / uniform map
        vmin = vmax * 1e-3
    return float(vmin), float(vmax)


def _attn_imshow(ax, B: np.ndarray, *, vmin: float, vmax: float, log: bool = True):
    r"""imshow one (N, N) attention map. ``log`` (default) uses ``LogNorm`` to resolve the peaky
    off-diagonal structure a linear scale washes to black; exact-zero (causal-masked) entries are
    non-positive, so ``LogNorm`` masks them and ``set_bad`` renders them black."""
    cmap = plt.cm.magma.copy()
    cmap.set_bad("black")
    if log:
        return ax.imshow(B, cmap=cmap, aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax))
    return ax.imshow(B, cmap=cmap, aspect="auto", vmin=0.0, vmax=vmax)


def plot_attention_heatmap(
    beta,                                # (N, N)

    *,
    log:   bool            = True,
    title: str             = "Attention",
    vmin:  Optional[float] = None,
    vmax:  Optional[float] = None,
    path:  Optional[str]   = None,
):
    r"""Log-scaled heatmap of one attention map (rows = queries i, cols = keys j).

    Attention is a peaky causal softmax (most mass on a few keys, exact zeros above the diagonal),
    so the default ``log`` scale (matplotlib ``LogNorm`` on beta) resolves the off-diagonal
    structure a linear scale collapses to black; the causal-masked zeros render as the 'bad'
    colour. Pass shared ``vmin``/``vmax`` to make several panels comparable; otherwise the positive
    entries set the scale (dynamic range capped at three decades).
    """
    B = _np(beta)
    vlo, vhi = _attn_log_bounds(B, vmin, vmax)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = _attn_imshow(ax, B, vmin=vlo, vmax=vhi, log=log)
    label = r"$\beta_{ij}$ (log scale)" if log else r"$\beta_{ij}$"
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
    num = np.convolve(v, kernel, mode="same")
    den = np.convolve(np.ones_like(v), kernel, mode="same")     # tap count per position (edge-correct)
    return num / den


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
        ax.annotate(f"{tag} {v[idx]:.1f}\n@ step {int(x[idx]):,}",
                    xy=(x[idx], v[idx]), xytext=(-12, 30), textcoords="offset points",
                    fontsize=7.5, ha="right",
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
    if annotate_final and v.size:
        ax.annotate(f"{v[-1]:.4g}", xy=(x[-1], v[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, fontweight="bold", color=color)
    if steps is not None and v.size:
        ax.set_xlim(float(np.min(x)), float(np.max(x)))
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
        iu = torch.triu_indices(log_sigma.shape[-1], log_sigma.shape[-1])
        return log_sigma[..., iu[0], iu[1]]
    raise ValueError(f"unknown belief channel {channel!r} (expected mu / sigma / phi)")


@register_figure("free_energy_descent")
def plot_free_energy_descent(
    history: Dict,                       # arrays over training steps: step, self_coupling, belief_coupling, attention_entropy, val_ce, [free_energy_total]

    *,
    lambda_beta: 'float | np.ndarray' = 1.0,
    self_div:    Optional[object]     = None,   # (M,) converged self-divergences for the violin
    path:        Optional[str]        = None,
):
    r"""F1: the per-token free-energy stack over training plus the F-vs-CE co-descent.

    DESCRIPTIVE, not a literal closed decomposition over one empirical object: the coupling terms
    are a per-eval converged-belief snapshot from a representative batch (logged off the graph),
    while the data term is the held-out val CE; both are in NATS PER TOKEN so they are commensurate
    (the caller normalizes the per-sequence-sum diagnostics by the token count before logging). Panel
    A stacks self-coupling, the lambda_beta-scaled belief-coupling and attention-entropy, and the
    data/likelihood term (val CE) -- the stack height is the full per-token F INCLUDING the data
    term. Panel B plots that SAME stacked total against val CE on a twin axis (so the two panels
    agree; the older path plotted a coupling-only total here that excluded the data term). Panel C
    (when ``self_div`` is given) is the per-token self-divergence violin. ``lambda_beta`` accepts a
    per-row vector (the learned-coupling trajectory) as well as a scalar; literal closure to the
    runtime F holds only at lambda_h = gamma_coupling = 0 (see free_energy_full_decomposition).
    """
    step = _np(history["step"])
    sc, bc = _np(history["self_coupling"]), _np(history["belief_coupling"])
    ae, ce = _np(history["attention_entropy"]), _np(history["val_ce"])
    lb = _np(lambda_beta)                                         # scalar OR (T,) learned trajectory
    stack = np.vstack([sc, lb * bc, lb * ae, ce])
    ncol = 3 if self_div is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 3.6))
    axes[0].stackplot(step, stack, colors=_CB[:4], alpha=0.85,
                      labels=["self-coupling", r"$\lambda_\beta\cdot$belief-coupling",
                              r"$\lambda_\beta\cdot$attention-entropy", "data term (CE)"])
    axes[0].set(xlabel="training step", ylabel="free energy (nats/token)", title="Free-energy decomposition")
    axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    ft = stack.sum(0)                                            # full per-token F incl. data term (matches panel A)
    axes[1].plot(step, ft, color=_CB[0], lw=2)
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


def _fe_terms(history: Dict, lambda_beta) -> tuple:
    r"""Per-eval free-energy term arrays (nats/token) shared by the decomposition + co-descent figures.

    Returns ``(step, self_coupling, lb*belief_coupling, lb*attention_entropy, data_ce, total)`` where the
    coupling terms carry the (scalar or per-row) ``lambda_beta`` weight that enters F and ``total`` is the
    full per-token F INCLUDING the data/likelihood term. The logged diagnostics are already per token, so
    the four contributions are commensurate.
    """
    step = _np(history["step"]).astype(float)
    sc   = _np(history["self_coupling"]).astype(float)
    lb   = _np(lambda_beta).astype(float)
    bc   = lb * _np(history["belief_coupling"]).astype(float)
    ae   = lb * _np(history["attention_entropy"]).astype(float)
    ce   = _np(history["val_ce"]).astype(float)
    return step, sc, bc, ae, ce, sc + bc + ae + ce


@register_figure("free_energy_codescent")
def plot_free_energy_codescent(
    history: Dict,                       # step, self_coupling, belief_coupling, attention_entropy, val_ce

    *,
    lambda_beta: 'float | np.ndarray' = 1.0,
    path:        Optional[str]        = None,
):
    r"""F-vs-CE co-descent: the full per-token free energy and the held-out loss fall together.

    Twin y-axes over the real training step -- the stacked total F (self-coupling + the lambda_beta-scaled
    belief-coupling and attention-entropy + the data/likelihood term, left, solid) and the held-out
    validation CE (right, dashed) -- each a faint raw line under a rolling-mean trend. The final values
    are tagged and the Pearson correlation of the two curves is in the title; a high positive r is the
    co-descent signature, the evidence that minimizing F lowers held-out loss.
    """
    step, sc, bc, ae, ce, total = _fe_terms(history, lambda_beta)
    keep = np.isfinite(total) & np.isfinite(ce)
    step, total, ce = step[keep], total[keep], ce[keep]
    w = max(5, total.size // 80)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(step, total, lw=0.7, color=_CB[0], alpha=0.25)
    ln1, = ax.plot(step, _rolling_mean(total, w), lw=2.0, color=_CB[0], label="F total (left)")
    ax.set(xlabel="training step", ylabel="free energy F (nats/token)")
    ax.yaxis.label.set_color(_CB[0]); ax.tick_params(axis="y", colors=_CB[0])
    if step.size:
        ax.set_xlim(float(step.min()), float(step.max()))
        ax.annotate(f"{total[-1]:.1f}", xy=(step[-1], total[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, fontweight="bold", color=_CB[0])
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)                         # set_publication_style hides it by default
    ax2.plot(step, ce, lw=0.7, color=_CB[1], alpha=0.25)
    ln2, = ax2.plot(step, _rolling_mean(ce, w), lw=2.0, color=_CB[1], ls="--", label="val CE (right)")
    ax2.set_ylabel("validation CE (nats/token)", color=_CB[1]); ax2.tick_params(axis="y", colors=_CB[1])
    if step.size:
        ax2.annotate(f"{ce[-1]:.2f}", xy=(step[-1], ce[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8, fontweight="bold", color=_CB[1])
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
    history: Dict,                       # step, self_coupling, belief_coupling, attention_entropy, val_ce

    *,
    lambda_beta: 'float | np.ndarray' = 1.0,
    path:        Optional[str]        = None,
):
    r"""The per-token free-energy budget at convergence and how its terms move over training.

    Panel A: the four F-contributions at the LAST eval on a LOG x-axis with the value past each bar, so
    the dominant self-coupling and the order-of-magnitude-smaller terms are all legible (a linear bar
    collapses the small ones to slivers, the failure mode of the old single bar). Panel B: the same terms
    at the early/mid/late thirds of training on a log y-axis -- self-coupling sits flat near its value
    while the belief-coupling, attention-entropy, and data terms carry the descent. Coupling terms carry
    the lambda_beta weight (so the bars are the actual F-contributions).
    """
    step, sc, bc, ae, ce, _ = _fe_terms(history, lambda_beta)
    names = ["self-coupling\n$\\mathrm{KL}(q\\|p)$",
             "belief-coupling\n$\\sum\\mathrm{KL}(q\\|\\Omega q)$",
             "attention-entropy\n$\\tau\\beta\\log(\\beta/\\pi)$",
             "data term\n$-\\mathbb{E}_q[\\log p]$"]
    labels = ["self-coupling", "belief coupling", "attention entropy", "data term"]
    series = [sc, bc, ae, ce]
    colors = _CB[:4]
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
    width   = 0.2
    for j, arr in enumerate(series):
        meds = [float(np.median(arr[idx])) if idx.size else np.nan for idx in thirds]   # skip empty thirds
        axes[1].bar(centers + (j - 1.5) * width, meds, width=width, color=colors[j], label=labels[j])
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
        ax.plot(x, _rolling_mean(vv, w), lw=2.0, color=color, label=label)
        ax.annotate(f"{vv[-1]:.3g}", xy=(x[-1], vv[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, fontweight="bold", color=color)
    if plotted:
        ax.set_yscale("symlog", linthresh=1e-4)                   # non-negative blocks, near-zero at init
        if step.size:
            ax.set_xlim(float(step.min()), float(step.max()))
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set(xlabel="training step", ylabel="model-channel F block (nats/token)",
           title="Model-channel free-energy blocks (raw)")
    fig.tight_layout()
    return _save(fig, path)


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


def plot_gamma_attention(
    g_data:  Dict,                       # extract.gamma_attention output (caller skips when None)

    *,
    log:     bool          = True,
    path:    Optional[str] = None,
):
    r"""Model-coupling attention gamma_ij (the gamma figure): the s-channel analogue of the belief beta maps.

    A grid of per-head gamma_ij = softmax_j(log pi^s_ij - E^s_ij/tau_g) heatmaps (rows = query i, cols =
    key j) on a shared LOG colour scale -- the model-channel consensus pattern, the structure the gamma
    block sum_ij gamma_ij KL(s_i||Omega s_j) weights. Causal-masked future positions are exact zeros
    (rendered black). Compare against :meth:`VFEModel.attention_maps`'s belief beta to see whether the
    model channel attends like the belief channel.
    """
    G = _np(g_data["gamma"])                                      # (H, N, N)
    if G.ndim == 2:
        G = G[None]
    H = G.shape[0]
    fig, axes = plt.subplots(1, H, figsize=(2.8 * H + 1.0, 3.0), squeeze=False)
    vlo, vhi = _attn_log_bounds(G)
    im = None
    for hi in range(H):
        ax = axes[0][hi]
        im = _attn_imshow(ax, G[hi], vmin=vlo, vmax=vhi, log=log)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"head {hi}")
        ax.set_xlabel("key $j$")
        if hi == 0:
            ax.set_ylabel("query $i$")
    fig.suptitle(r"Model-coupling attention $\gamma_{ij}$ (s-channel)", y=1.04)
    label = r"$\gamma_{ij}$ (log scale)" if log else r"$\gamma_{ij}$"
    fig.colorbar(im, ax=list(axes[0]), shrink=0.8, label=label)
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


@register_figure("belief_umap")
def plot_belief_umap(
    bank:       Dict,                    # belief_bank output: mu, sigma, phi, token_ids, seq_idx
    channel:    str = "mu",              # which belief channel to embed: 'mu' / 'sigma' / 'phi'

    *,
    decode:     Optional[object] = None,                       # decode(list[int]) -> str; None -> no categories
    taxonomies: tuple            = ("bpe", "function_content"),
    n_label:    int              = 10,
    seed:       int              = 0,
    sil_sample: int              = 2000,
    path:       Optional[str]    = None,
):
    r"""F5: UMAP of one belief channel, coloured by linguistic token category, with labelled words.

    ONE figure per channel (the caller emits mu / sigma / phi separately). The channel is embedded
    faithfully to its geometry (mu Euclidean, Sigma in the log-Euclidean chart, phi in the gauge
    coordinates), and the SAME 2-D embedding is shown once per taxonomy: a BPE-structure panel and a
    function/content panel, each coloured by category with a legend. The ``n_label`` most frequent
    tokens are decoded and annotated at their occurrence-centroid. Each panel title reports the
    silhouette + Calinski-Harabasz of the CATEGORY labels in the channel's NATIVE space, so the number
    measures how strongly the channel separates that linguistic taxonomy (not an arbitrary token id).
    With ``decode is None`` (no tokenizer) it degrades to a single-colour scatter with the same labels.
    """
    feats = _belief_channel_features(bank, channel)
    fnp = _np(feats)
    coords = umap_embed(feats, seed=seed)
    token_ids = _np(bank["token_ids"]).astype(int)
    fig, axes = plt.subplots(1, len(taxonomies), figsize=(6.4 * len(taxonomies), 5.4), squeeze=False)
    for ax, tax in zip(axes[0], taxonomies):
        labels, names = (None, None) if decode is None else _token_category_labels(token_ids, decode, tax)
        _scatter_by_category(ax, coords, labels, names)
        _annotate_frequent_tokens(ax, coords, token_ids, decode, n_label)
        title = f"{channel} · {tax.replace('_', '/')}"
        if labels is not None:
            cm = clustering_metrics(fnp, labels, sample_size=sil_sample)
            title += f"  (sil {cm['silhouette']:.2f}, CH {cm['calinski_harabasz']:.0f})"
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title=title)
    fig.suptitle(f"Belief semantic clustering — {channel} (colour = linguistic category)")
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
    r"""F7: per-head gauge specialization and its link to attention structure.

    Panel A: per-head violins of the group-correct gauge invariant (log-volume). Panel B: per-head
    violins of the shear/anisotropy, or -- when ``head_entropy`` is given -- a scatter of per-head
    gauge magnitude vs attention row entropy.
    """
    logdet = _np(per_head["logdet"])                             # (M, H)
    aniso = _np(per_head["anisotropy"])
    h = logdet.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].violinplot([logdet[:, i] for i in range(h)], showmeans=True)
    axes[0].axhline(0.0, color="#888888", ls=":", lw=1)
    axes[0].set(xticks=range(1, h + 1), xticklabels=[f"h{i}" for i in range(h)],
                ylabel=r"$\log|\det\exp\phi^{(h)}|$", title="Per-head gauge volume")
    if head_entropy is not None:
        axes[1].scatter(np.abs(logdet).mean(0), _np(head_entropy), s=40, color=_CB[0])
        axes[1].set(xlabel="mean |gauge volume|", ylabel="attention row entropy (nats)",
                    title="Gauge action vs attention")
    else:
        axes[1].violinplot([aniso[:, i] for i in range(h)], showmeans=True)
        axes[1].set(xticks=range(1, h + 1), xticklabels=[f"h{i}" for i in range(h)],
                    ylabel=r"$s_{\max}/s_{\min}$", title="Per-head shear")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("attention_structure")
def plot_attention_structure(
    beta: object,                        # (L,H,N,N) or (H,N,N) attention weights

    *,
    path: Optional[str] = None,
):
    r"""F8: attention structure -- per-head entropy, head redundancy, and distance decay.

    Panel A: per-head row-entropy violins vs the uniform log N reference (does attention sharpen).
    Panel B: head-redundancy Jensen-Shannon heatmap (specialized vs redundant heads). Panel C: the
    per-head attention-vs-offset profile on a log axis (positional decay).
    """
    from vfe3 import metrics
    import torch
    b = beta if hasattr(beta, "dim") else torch.as_tensor(_np(beta))
    if b.dim() == 4:                                             # (L, H, N, N) -> flatten to heads
        b = b.reshape(-1, b.shape[-2], b.shape[-1])
    rows = _np(metrics.attention_entropy_rows(b))               # (H, N)
    n = b.shape[-1]
    h = b.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    axes[0].violinplot([rows[i] for i in range(h)], showmeans=True)
    axes[0].axhline(float(np.log(n)), color="#444444", ls="--", lw=1, label=r"uniform $\log N$")
    axes[0].set(xticks=range(1, h + 1), xticklabels=[f"h{i}" for i in range(h)],
                ylabel="row entropy (nats)", title="Per-head attention entropy")
    axes[0].legend(fontsize=8, frameon=False)
    js = _np(metrics.head_redundancy_js(b))
    im = axes[1].imshow(js, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="JS divergence (nats)")
    axes[1].set(xlabel="head", ylabel="head", title="Head redundancy")
    dd = metrics.attention_distance_decay(b)
    prof = _np(dd["profile"])                                    # (H, N)
    off = _np(dd["offsets"])
    for i in range(h):
        axes[2].plot(off, np.clip(prof[i], 1e-8, None), color=plt.cm.viridis(i / max(h - 1, 1)), lw=1.2)
    axes[2].set_yscale("log")
    axes[2].set(xlabel="offset |i - j|", ylabel=r"mean $\beta$", title="Attention distance decay")
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
        leak = _np(causal["future_leakage"]).ravel()
        axes[0][2].bar(range(leak.size), leak, color=_CB[3])
        axes[0][2].set(xlabel="head", ylabel="max future-attention", title="Causal leakage (expect 0)")
    fig.tight_layout()
    return _save(fig, path)
