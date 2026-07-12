r"""Publication-quality figures for VFE_3.0 diagnostics (matplotlib; UMAP / networkx / sklearn).

Figure generators over beliefs (means / gauge frames), attention, covariance, and training
trajectories. Each returns a matplotlib Figure and optionally saves it; colourblind-safe
palette and journal-ish defaults via ``set_publication_style``. The heavier dependencies
(UMAP, networkx, scikit-learn) are imported lazily inside the function that needs them, so
the module imports even where one is absent (the function raises a clear message instead).
Tensors are accepted as torch or numpy; everything is detached to numpy for plotting.

A registry (``register_figure``) lets a new figure slot in by name.
"""

from typing import Callable, Dict, Mapping, Optional

import matplotlib

matplotlib.use("Agg")                                            # non-interactive (headless / tests)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Wong colourblind-safe qualitative palette (used module-wide, incl. the trajectory defaults).
_CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]

# tab20 reindex for cluster coloring: the ten saturated distinct hues first, their pastels second,
# both grays last (grays are confusable with the light-gray noise layer). tab20's native order is
# saturated/pastel pairs of the SAME hue, so size-adjacent clusters differed only in lightness.
_TAB20_DISTINCT = [0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 3, 5, 7, 9, 11, 13, 17, 19, 14, 15]


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


def _zoom_bar_value_axis(ax, values, *, vertical: bool = True,
                         lo_frac: float = 0.25, hi_frac: float = 0.12) -> None:
    r"""Zoom a bar chart's VALUE axis to a padded ``[min, max]`` window (not ``[0, max]``) so
    near-equal bars are distinguishable instead of all reading as full bars from zero. No-op unless
    there are at least two finite values that are not all equal (so a single- or constant-valued
    chart keeps its default autoscale)."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin:
        return
    span = vmax - vmin
    lo, hi = vmin - lo_frac * span, vmax + hi_frac * span
    (ax.set_ylim if vertical else ax.set_xlim)(lo, hi)


# Default UMAP hyper-parameters. Module-level so the on-figure parameter footer states the values
# actually used instead of restating literals (they are not otherwise in the plot functions' scope).
_UMAP_N_NEIGHBORS = 15                                           # local-neighborhood size (callers scale with N)
_UMAP_MIN_DIST    = 0.1                                          # display-tuned spacing (0.0 for clustering runs)

# Worker source for the ISOLATED umap embedding subprocess (see umap_embed).
# init="pca" skips UMAP's spectral eigensolver, which on disconnected / near-degenerate belief
# clouds fails to converge (tiny eigengap) and silently falls back to uniform-random init; PCA is
# deterministic and data-aware. n_jobs=1 matches what random_state already forces, so it also
# silences UMAP's "n_jobs overridden" warning. Same compute, no warning cascade.
_UMAP_WORKER_SRC = (
    "import sys\n"
    "import numpy as np\n"
    "import umap\n"
    "X = np.load(sys.argv[1])\n"
    "reducer = umap.UMAP(n_neighbors=int(sys.argv[3]), min_dist=float(sys.argv[4]),\n"
    "                    n_components=int(sys.argv[6]), init='pca', random_state=int(sys.argv[5]),\n"
    "                    n_jobs=1)\n"
    "np.save(sys.argv[2], reducer.fit_transform(X))\n"
)


def umap_embed(
    features,                            # (N, D) tensor/array

    *,
    n_neighbors:  int = _UMAP_N_NEIGHBORS,
    min_dist:     float = _UMAP_MIN_DIST,
    n_components: int = 2,
    seed:         int = 0,
):
    r"""UMAP embedding of ``features`` ((N, D) -> (N, n_components)), run in an ISOLATED subprocess.

    umap-learn's numba/llvmlite native layer can die with a Windows ACCESS VIOLATION when it
    initializes inside a long-running, heavily loaded process (observed on Python 3.14 after
    hundreds of tests: llvmlite ``check_jit_execution`` faults; a fresh process imports numba
    fine) -- a process-killing crash no in-process try/except can catch, taking the whole
    training finalize / test session down with it. Running the embedding in a fresh subprocess
    fully isolates the native layer: same computation, same seeded result. A failing subprocess
    (numba genuinely unsupported, umap-learn absent) raises the OSError/ImportError the umap
    consumers were already written to handle (audit 2026-07-05 verification fix)."""
    X = _np(features)
    # PCA init raises when n_components exceeds the feature dim or N-1 (e.g. the phi channel of a
    # small algebra, tiny CPU test banks), so clamp it the same way n_neighbors is clamped below.
    n_components = max(1, min(n_components, X.shape[1], max(1, X.shape[0] - 1)))
    # A fully collapsed channel (every point identical -> zero variance) has no embedding, and PCA
    # init would divide by total variance 0 and yield NaN. Return a trivial finite layout so the
    # downstream clustering / KDE stay valid (faithful: constant features carry no 2-D structure).
    if X.shape[0] < 3 or float(np.ptp(X, axis=0).max()) <= 0.0:
        return np.zeros((X.shape[0], n_components), dtype=float)
    n_neighbors = min(n_neighbors, max(2, X.shape[0] - 1))
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    workdir = tempfile.mkdtemp(prefix="vfe3_umap_")
    try:
        fin  = os.path.join(workdir, "in.npy")
        fout = os.path.join(workdir, "out.npy")
        np.save(fin, X)
        proc = subprocess.run(
            [sys.executable, "-c", _UMAP_WORKER_SRC,
             fin, fout, str(n_neighbors), str(min_dist), str(seed), str(n_components)],
            capture_output=True, timeout=1200,
        )
        if proc.returncode != 0:
            tail = proc.stderr.decode(errors="replace")[-500:]
            if "ModuleNotFoundError" in tail and "umap" in tail:
                raise ImportError("umap_embed needs umap-learn (pip install umap-learn)")
            raise OSError(f"umap embedding subprocess failed (rc={proc.returncode}): {tail}")
        return np.load(fout)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


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
        # ``:.4g`` (matching annotate_final) so a heavy-tailed log series whose extremum is O(1e-4)
        # -- holonomy.png -- prints its real value instead of rounding to "0.0" under ``:.1f``.
        ax.annotate(f"{tag} {v[idx]:.4g}\n@ step {int(x[idx]):,}",
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


def register_figure(name: str, *, override: bool = False) -> Callable:
    """Decorator registering a figure generator under ``name``.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable) -> Callable:
        if name in _FIGURES and not override:
            raise KeyError(f"figure {name!r} already registered; pass override=True to replace")
        _FIGURES[name] = fn
        return fn
    return _wrap


def get_figure(name: str) -> Callable:
    """Return the registered figure generator (KeyError if absent)."""
    if name not in _FIGURES:
        raise KeyError(f"no figure {name!r}; available: {sorted(_FIGURES)}")
    return _FIGURES[name]


# ===========================================================================
# Publication figures (the claim-linked set). Each consumes the vfe3.metrics
# measurements / vfe3.viz.extract runner outputs, builds a multi-panel
# composite, registers by name, and returns the Figure.
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
    history: Dict,                       # step + single-batch or microbatch-mean E-step grad norms

    *,
    path:    Optional[str] = None,
):
    r"""E-STEP per-component belief-gradient L2 norms over training: the mu / sigma / phi decomposition
    of the inference free-energy gradient.

    The companion to the M-step grad_norm_decomposition: where that shows how hard the LEARNING (M-step,
    parameter) gradient pushes each belief-component family, this shows how hard the INFERENCE (E-step)
    gradient ``\nabla F`` pushes the belief tuple ``(mu, Sigma, phi)`` itself -- the RAW gradient inside
    the last E-step iteration, before the Fisher / natural-gradient preconditioning, captured by
    model.forward(estep_grad_out=...). Accumulated steps use the arithmetic mean across contributing
    microbatches. A component reads 0 (dropped on the log y) when its substep is off (e.g. phi when
    e_phi_lr=0).
    """
    mean_keys = (
        "estep_grad_norm_mu_microbatch_mean",
        "estep_grad_norm_sigma_microbatch_mean",
        "estep_grad_norm_phi_microbatch_mean",
    )
    keys = (mean_keys if any(key in history for key in mean_keys)
            else ("estep_grad_norm_mu", "estep_grad_norm_sigma", "estep_grad_norm_phi"))
    spec = [
        (keys[0], r"$\|\nabla_\mu F\|_2$",    _CB[0], "mu"),
        (keys[1], r"$\|\nabla_\Sigma F\|_2$", _CB[1], "sigma"),
        (keys[2], r"$\|\nabla_\phi F\|_2$",   _CB[2], "phi"),
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


# ---------------------------------------------------------------------------
# History dashboards: small-multiples over metrics.csv columns the training loop
# already logs but no standard figure surfaced (geometry / E-step / validation /
# optimizer geometry). Shared helpers below; one registered figure per family.
# ---------------------------------------------------------------------------

def _series_panel(
    ax,
    history: Dict,
    step:    np.ndarray,
    specs:   list,                       # [(column_key, label, color), ...]

    *,
    logy:    bool = False,
) -> int:
    r"""Plot each PRESENT, finite series of ``specs`` onto ``ax`` (a faint raw line under a rolling-mean
    trend for a long series, ``o-`` markers for a short one), masking non-finite samples per series so
    an eval-cadence column (NaN on non-eval rows) draws only on its eval steps. Returns the series count
    actually drawn."""
    drawn = 0
    for key, label, color in specs:
        if key not in history:
            continue
        v = _np(history[key]).astype(float)
        keep = np.isfinite(v)
        if logy:
            keep = keep & (v > 0)                                # a log axis drops non-positive samples
        x, vv = step[keep], v[keep]
        if not vv.size:
            continue
        drawn += 1
        if vv.size > 8:
            ax.plot(x, vv, lw=0.7, color=color, alpha=0.25)
            ax.plot(x, _rolling_mean(vv, max(5, vv.size // 80)), lw=2.0, color=color,
                    label=f"{label} ={vv[-1]:.3g}")
        else:
            ax.plot(x, vv, "o-", ms=4, lw=1.6, color=color, label=f"{label} ={vv[-1]:.3g}")
    if drawn:
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=7, frameon=False, loc="best")
    return drawn


def _history_dashboard(
    history:  Dict,
    panels:   list,                      # [{title, ylabel, series:[(key,label,color)], [logy]}, ...]
    suptitle: str,
    path:     Optional[str],

    *,
    ncols:    int = 3,
) -> object:
    r"""Small-multiples dashboard over training history: one mini-panel per group of commensurate
    columns. A panel whose every column is absent or all-NaN is dropped (so a run missing a metric
    family simply shows fewer panels), and the grid self-sizes to the surviving panel count."""
    step = _np(history["step"]).astype(float)
    live = []
    for p in panels:
        present = [(k, lab, c) for (k, lab, c) in p["series"]
                   if k in history and bool(np.isfinite(_np(history[k]).astype(float)).any())]
        if present:
            live.append({**p, "series": present})
    if not live:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.text(0.5, 0.5, "no history columns present", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    nrows = (len(live) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.1 * nrows), squeeze=False)
    for i, p in enumerate(live):
        ax = axes[i // ncols][i % ncols]
        _series_panel(ax, history, step, p["series"], logy=bool(p.get("logy", False)))
        ax.set(xlabel="training step", ylabel=p["ylabel"], title=p["title"])
        if step.size:
            ax.set_xlim(float(step.min()), float(step.max()))
            _step_xaxis(ax)
    for j in range(len(live), nrows * ncols):                    # blank the unused grid cells
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("geometry_health")
def plot_geometry_health(
    history: Dict,                       # step + gauge/SPD/Fisher/guard health columns (any subset)

    *,
    path:    Optional[str] = None,
):
    r"""Gauge / SPD / Fisher geometry-health dashboard over training.

    Surfaces the converged-state geometry scalars ``diagnostics()`` already logs to ``metrics.csv`` but
    that no standard figure plotted: holonomy / cocycle flatness, gauge-frame magnitude and spread,
    belief-covariance conditioning and effective rank, belief Fisher precision, numerical-guard
    saturation, attention-entropy collapse, and learned-connection / head-mixer trainability
    (norms of the opt-in ``connection_W`` / ``connection_M`` / ``connection_L`` and the head-mixer
    drift from identity -- audit 2026-07-01 round-3). Together they say whether the learned geometry
    stays meaningful rather than degenerating to a flat cocycle (phi -> 0, an UNGAUGED transformer) or
    a guard-pinned fixed point. Each panel self-gates on column presence."""
    panels = [
        {"title": "Holonomy / cocycle flatness", "ylabel": "curvature", "logy": True, "series": [
            ("holonomy_wilson", r"Wilson $1-\mathrm{Re}\,\mathrm{Tr}(H)/K$ (gauge-invariant)", _CB[0]),
            ("cocycle_residual", "cocycle residual", _CB[1]),
            ("holonomy_deviation", r"$\langle\|H-I\|_F\rangle$ (frame-dependent)", _CB[2])]},
        {"title": "Gauge-frame spread", "ylabel": "spread", "series": [
            ("gauge_invariant_spread", "gauge invariant", _CB[3]),
            ("gauge_head_logdet_spread", r"head $\log|\det|$", _CB[4])]},
        {"title": "Gauge-frame magnitude", "ylabel": r"$\|\phi\|$", "series": [
            ("phi_norm_mean", r"mean $\|\phi\|$", _CB[0]),
            ("phi_norm_std", r"std $\|\phi\|$", _CB[5])]},
        {"title": "Belief conditioning", "ylabel": r"$\kappa(\Sigma)$", "logy": True, "series": [
            ("belief_cond_p95", "p95", _CB[1]),
            ("belief_cond_max", "max", _CB[3])]},
        {"title": "Belief effective rank", "ylabel": r"$\mathrm{erank}(\Sigma)$", "series": [
            ("eff_rank_p5", "p5", _CB[2]),
            ("eff_rank_median", "median", _CB[0]),
            ("eff_rank_p95", "p95", _CB[4])]},
        {"title": "Belief precision", "ylabel": r"Half Fisher trace $\langle\mathrm{tr}\,\Sigma^{-1}\rangle/2$",
         "logy": True, "series": [("fisher_trace_mean", "Half Fisher trace", _CB[3])]},
        {"title": "Guard saturation", "ylabel": "fraction", "series": [
            ("guard_sigma_floor_frac", r"$\sigma$ floor", _CB[0]),
            ("guard_sigma_ceil_frac", r"$\sigma$ ceil", _CB[1]),
            ("guard_energy_klmax_frac", "energy klmax", _CB[2]),
            ("guard_selfdiv_klmax_frac", "selfdiv klmax", _CB[4])]},
        {"title": "Numerical safety", "ylabel": "fraction", "series": [
            ("nonfinite_frac", "nonfinite", _CB[1]),
            ("renyi_band_frac", "Renyi band", _CB[5])]},
        {"title": "Attention-entropy collapse", "ylabel": "count / nats", "series": [
            ("attn_entropy_collapsed_heads", "collapsed heads", _CB[0]),
            ("attn_entropy_min", "min row entropy", _CB[3])]},
        {"title": "Learned-connection trainability", "ylabel": "norm / drift", "logy": True, "series": [
            ("connection_w_norm", r"$\|W\|_F$ (regime II)", _CB[0]),
            ("connection_m_norm", r"$\|M\|_F$ (covariant)", _CB[1]),
            ("connection_l_norm", r"$\|L\|_F$ (direct link)", _CB[2]),
            ("connection_l_offdiag_norm", r"$\|L\|_F$ off-diag", _CB[4]),
            ("head_mixer_drift", "head-mixer drift", _CB[5])]},
    ]
    return _history_dashboard(history, panels, "Gauge / SPD / Fisher geometry health", path)


@register_figure("estep_quality")
def plot_estep_quality(
    history: Dict,                       # step + estep_f_drop / nondecreasing-frac / belief residuals

    *,
    path:    Optional[str] = None,
):
    r"""E-step inference-quality dashboard over training.

    Direct evidence for what the inner E-step contributes per eval: the free-energy drop across the
    inner loop (``estep_f_drop`` < 0 = F descended), the fraction of inner iterations that did NOT
    decrease F (a descent-quality readout for parallel mean-field -- EXPECTED nonzero, not a failure
    flag), and the last-iteration belief-change residuals for ``mu`` / ``Sigma`` / ``phi`` (the
    convergence certificate). Complements the standalone E-step convergence-trend curve."""
    panels = [
        {"title": "E-step free-energy drop", "ylabel": r"$F_{\mathrm{end}}-F_{\mathrm{start}}$",
         "series": [("estep_f_drop", "F drop", _CB[0])]},
        {"title": "Inner-loop monotonicity", "ylabel": "nondecreasing fraction",
         "series": [("estep_f_nondecreasing_frac", "nondecreasing frac", _CB[1])]},
        {"title": "Belief residuals (last iter)", "ylabel": "step length", "logy": True, "series": [
            ("estep_r_mu_last", r"$r_\mu$", _CB[0]),
            ("estep_r_sigma_last", r"$r_\Sigma$ (SPD)", _CB[1]),
            ("estep_r_phi_last", r"$r_\phi$", _CB[2])]},
    ]
    return _history_dashboard(history, panels, "E-step inference quality", path, ncols=3)


@register_figure("validation_sanity")
def plot_validation_sanity(
    history: Dict,                       # step + held-out generalization / positional / attention / geometry columns

    *,
    path:    Optional[str] = None,
):
    r"""Held-out validation-sanity dashboard over training.

    The per-eval probes computed on a HELD-OUT batch (``_val_diagnostics``) that otherwise live only in
    ``metrics.csv``: the generalization gap, within-sequence positional loss and its tail/early ratio,
    causal-mask sanity (future leakage, row-sum error), positional-vs-content structure, structured-head
    masses and head redundancy, and the held-out gauge / SPD / Fisher geometry (the more credible
    counterpart to the train-batch ``geometry_health`` figure). Each panel self-gates on column
    presence; eval-cadence columns draw only on their eval steps."""
    panels = [
        {"title": "Generalization gap", "ylabel": r"$\mathrm{CE}_{\mathrm{val}}-\mathrm{CE}_{\mathrm{train}}$",
         "series": [("generalization_gap", "gap", _CB[0])]},
        {"title": "Positional loss", "ylabel": "CE (nats)", "series": [
            ("pos_loss_first_q", "first quartile", _CB[0]),
            ("pos_loss_last_q", "last quartile", _CB[1])]},
        {"title": "Positional loss ratio", "ylabel": "last / first quartile",
         "series": [("pos_loss_ratio", "ratio", _CB[2])]},
        {"title": "Causal-mask sanity", "ylabel": "max error", "logy": True, "series": [
            ("val_future_leakage", "future leakage", _CB[0]),
            ("val_row_sum_error", "row-sum error", _CB[1])]},
        {"title": "Positional vs content", "ylabel": r"$R^2$",
         "series": [("val_pos_content_r2", "pos-content", _CB[3])]},
        {"title": "Structured heads", "ylabel": "mass / JS", "series": [
            ("val_prev_token_mass", "prev-token", _CB[0]),
            ("val_period_match_mass", "period-match", _CB[1]),
            ("val_head_redundancy_js", "redundancy JS", _CB[2])]},
        {"title": "Held-out flatness", "ylabel": "curvature", "logy": True, "series": [
            ("val_holonomy_wilson", "Wilson holonomy (gauge-invariant)", _CB[0]),
            ("val_cocycle_residual", "cocycle residual", _CB[1])]},
        {"title": "Held-out gauge spread", "ylabel": "spread",
         "series": [("val_gauge_invariant_spread", "gauge invariant", _CB[3])]},
        {"title": "Held-out conditioning", "ylabel": r"$\kappa(\Sigma)$", "logy": True,
         "series": [("val_belief_cond_p95", "p95", _CB[1])]},
        {"title": "Held-out precision", "ylabel": r"Half Fisher trace $\langle\mathrm{tr}\,\Sigma^{-1}\rangle/2$",
         "logy": True, "series": [("val_fisher_trace_mean", "Half Fisher trace", _CB[3])]},
        {"title": "Held-out guard saturation", "ylabel": "fraction", "series": [
            ("val_guard_sigma_floor_frac", r"$\sigma$ floor", _CB[0]),
            ("val_guard_sigma_ceil_frac", r"$\sigma$ ceil", _CB[1]),
            ("val_guard_energy_klmax_frac", "energy klmax", _CB[2])]},
        {"title": "Held-out frame norm", "ylabel": r"$\|\phi\|$", "series": [
            ("val_phi_norm_mean", r"mean $\|\phi\|$", _CB[0]),
            ("val_phi_norm_std", r"std $\|\phi\|$", _CB[5])]},
    ]
    return _history_dashboard(history, panels, "Held-out validation sanity", path, ncols=4)


@register_figure("optimizer_geometry")
def plot_optimizer_geometry(
    history: Dict,                       # step + cos_nat_phi / pullback cond / role weight & grad norms

    *,
    path:    Optional[str] = None,
):
    r"""Information-geometric optimizer dashboard over training.

    Evidence that the gauge M-step is natural-gradient (information-geometric) rather than plain Adam:
    the cosine between the natural and raw ``phi`` gradients, the pullback-metric condition number, the
    per-role parameter weight norms, and the gradient-to-weight ratio ``||grad_theta|| / ||theta||`` per
    role (synthesized from the logged grad and weight norms; multiply by the per-group LR for the true
    update-to-weight ratio). Each panel self-gates on column presence; the pullback panel appears only
    on the pullback precond modes, ``cos_nat_phi`` only on a natural-gradient gauge run."""
    hist = dict(history)
    for role in ("mu", "sigma", "phi"):                          # synthesize the gradient-to-weight ratio
        gk, wk = f"grad_norm_{role}", f"weight_norm_{role}"
        if gk in hist and wk in hist:
            g = _np(hist[gk]).astype(float)
            w = _np(hist[wk]).astype(float)
            hist[f"grad_weight_ratio_{role}"] = np.where(w > 0, g / w, np.nan)
    panels = [
        {"title": "Natural-gradient alignment",
         "ylabel": r"$\cos(\nabla^{\mathrm{nat}}_\phi,\nabla_\phi)$",
         "series": [("cos_nat_phi", r"$\cos(\mathrm{nat},\mathrm{raw})$", _CB[0])]},
        {"title": "Pullback conditioning", "ylabel": "condition number", "logy": True, "series": [
            ("pullback_cond_median", "median", _CB[1]),
            ("pullback_cond_max", "max", _CB[3])]},
        {"title": "Role weight norms", "ylabel": r"$\|\theta\|_2$", "logy": True, "series": [
            ("weight_norm_mu", r"$\mu$", _CB[0]),
            ("weight_norm_sigma", r"$\Sigma$", _CB[1]),
            ("weight_norm_phi", r"$\phi$", _CB[2])]},
        {"title": "Gradient-to-weight ratio", "ylabel": r"$\|\nabla_\theta\|/\|\theta\|$", "logy": True,
         "series": [
            ("grad_weight_ratio_mu", r"$\mu$", _CB[0]),
            ("grad_weight_ratio_sigma", r"$\Sigma$", _CB[1]),
            ("grad_weight_ratio_phi", r"$\phi$", _CB[2])]},
    ]
    return _history_dashboard(hist, panels, "Optimizer information geometry", path, ncols=2)


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


@register_figure("kappa_history")
def plot_kappa_history(
    history: Dict,                       # step + kappa_{beta,gamma}_{mean,var}

    *,
    channel: str           = "beta",     # "beta" or "gamma"
    path:    Optional[str] = None,
):
    r"""Learned softmax-temperature trajectory over training.

    ``train`` logs ``kappa_<channel>_mean`` and ``kappa_<channel>_var`` for each metrics row when
    the corresponding learnable-kappa toggle is active. The mean is across irrep blocks/heads; the
    shaded band is mean +/- sqrt(population variance), clipped at 0 because kappa is strictly
    positive by construction (``kappa = exp(log_kappa)``).
    """
    prefix = f"kappa_{channel}"
    mean_key = f"{prefix}_mean"
    var_key  = f"{prefix}_var"
    mean = _np(history.get(mean_key, [])).reshape(-1).astype(float)
    if mean.size == 0:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.text(0.5, 0.5, f"no {prefix} history", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    step = _np(history.get("step", np.arange(mean.size))).reshape(-1).astype(float)
    if step.size != mean.size:
        step = np.arange(mean.size, dtype=float)
    var = _np(history.get(var_key, np.full(mean.shape, np.nan))).reshape(-1).astype(float)
    if var.size != mean.size:
        var = np.full(mean.shape, np.nan)
    keep = np.isfinite(step) & np.isfinite(mean)
    std = np.sqrt(np.maximum(var, 0.0))
    band = keep & np.isfinite(std)
    symbol = r"\beta" if channel == "beta" else r"\gamma"
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    if band.any():
        lower = np.maximum(mean - std, 0.0)
        upper = mean + std
        ax.fill_between(step, lower, upper, where=band, color=_CB[5], alpha=0.20,
                        label=r"$\pm\sqrt{\mathrm{var}}$ across blocks")
    if keep.any():
        ax.plot(step[keep], mean[keep], "o-", color=_CB[0], lw=1.7, ms=3,
                label=rf"mean $\kappa_{{{symbol}}}$")
        ax.set_xlim(float(step[keep].min()), float(step[keep].max()))
        _step_xaxis(ax)
    ax.set(xlabel="training step", ylabel=rf"$\kappa_{{{symbol}}}$",
           title=f"Learned {channel} kappa over training")
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("kappa_block_trajectory")
def plot_kappa_block_trajectory(
    history: Dict,                       # step + kappa_<ch>_b<i> + tau_<ch>_b<i>

    *,
    path: Optional[str] = None,
):
    r"""Per-irrep-block learned softmax-temperature trajectories over training.

    ``train`` logs ``kappa_<ch>_b<i>`` and the effective temperature ``tau_<ch>_b<i> = kappa_b *
    sqrt(d_block)`` for every irrep block whenever the corresponding learnable-kappa toggle is on.
    Rows are the channels present in the history (belief ``beta`` and model ``gamma``); the two
    columns are kappa (left) and tau (right); each panel draws one line per block. Both toggles on
    -> a 2x2 grid, one toggle -> a 1x2 row. History-only (no model re-run), so the lines are the
    values actually logged during training, not a re-inferred temperature.
    """
    step = _np(history.get("step", [])).reshape(-1).astype(float)
    channels = [ch for ch in ("beta", "gamma")
                if any(k.startswith(f"kappa_{ch}_b") for k in history)]
    if not channels or step.size == 0:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.text(0.5, 0.5, "no per-block kappa history", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    fig, axes = plt.subplots(len(channels), 2, figsize=(9.0, 3.4 * len(channels)), squeeze=False)
    for ri, ch in enumerate(channels):
        symbol = r"\beta" if ch == "beta" else r"\gamma"
        pre = f"kappa_{ch}_b"
        blocks = sorted(int(k[len(pre):]) for k in history
                        if k.startswith(pre) and k[len(pre):].isdigit())
        for ci, (quant, qsym) in enumerate((("kappa", rf"$\kappa_{{{symbol}}}$"),
                                            ("tau",   rf"$\tau_{{{symbol}}}$"))):
            ax = axes[ri][ci]
            for bi in blocks:
                y = _np(history.get(f"{quant}_{ch}_b{bi}", [])).reshape(-1).astype(float)
                if y.size != step.size:
                    continue
                keep = np.isfinite(step) & np.isfinite(y)
                if keep.any():
                    ax.plot(step[keep], y[keep], "o-", ms=2.5, lw=1.4,
                            color=_CB[bi % len(_CB)], label=f"b{bi}")
            _sf = step[np.isfinite(step)]
            if _sf.size:
                ax.set_xlim(float(_sf.min()), float(_sf.max()))
                _step_xaxis(ax)
            ax.set(xlabel="training step", ylabel=qsym,
                   title=f"{ch} {quant} per irrep block")
            if blocks:
                ax.legend(fontsize=7, frameon=False, ncol=max(1, (len(blocks) + 5) // 6))
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


def _cluster_embedding(
    coords: np.ndarray,                  # (M, D) clustering-space coordinates (2-D display or higher-D)

    *,
    seed:   int = 0,
) -> tuple:                              # ((M,) integer cluster labels (-1 = noise), method description)
    r"""Density-cluster an embedding into data-driven groups (HDBSCAN; KMeans fallback).

    HDBSCAN finds variable-density clusters AND an explicit noise label (-1) -- the right tool for the
    belief embedding's tight peripheral function-word islands around a diffuse content core. Parameter
    regime (audit 2026-07-11): ``leaf`` selection with a scale-relative ``cluster_selection_epsilon``
    instead of the default ``eom``, which on a diffuse-core-plus-islands cloud prefers the root-level
    blob (the one-giant-cluster failure); ``min_cluster_size`` grows as sqrt(M) so the 64-sequence
    finalize bank and the 256-sequence make_figures bank get comparable granularity (the old linear
    M//60 floor did not); ``min_samples`` is tied to the size floor rather than a fixed noisy 5.
    Falls back to KMeans (no noise label) only when the installed sklearn predates ``cluster.HDBSCAN``
    (<1.3); a genuine HDBSCAN runtime error propagates to the caller's best-effort guard instead of
    silently switching algorithms. Returns ``(labels, method_desc)`` where ``method_desc`` names the
    algorithm and parameters actually used (for the on-figure footer). Deterministic given ``coords``
    (HDBSCAN has no RNG; the KMeans fallback is seeded)."""
    M = coords.shape[0]
    if M < 3:                                                    # no clusters to find (HDBSCAN would raise)
        return np.full(M, -1, dtype=int), "degenerate (<3 points)"
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:                                          # pragma: no cover - old sklearn only
        from sklearn.cluster import KMeans
        k = max(2, min(14, M // 50))
        return (KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(coords),
                f"KMeans fallback (k={k}, no noise label)")
    mcs = max(2, min(max(20, int(np.sqrt(M))), M))               # clamps keep tiny test banks valid
    ms  = max(1, min(max(10, mcs // 20), M))
    eps = 0.02 * float(np.ptp(coords, axis=0).max())
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method="leaf",
                     cluster_selection_epsilon=eps).fit_predict(coords)
    return labels, f"HDBSCAN(mcs={mcs}, ms={ms}, leaf, eps={eps:.2g})"


def _density_peak_anchor(
    pts:    np.ndarray,                  # (n_c, 2) one cluster's display coordinates
    extent: list,                        # [[xmin, xmax], [ymin, ymax]] global display extent

    *,
    grid:   int = 200,
    smooth: float = 2.0,
) -> np.ndarray:                         # (2,) an actual member point at the cluster's density peak
    r"""The cluster MEMBER at the peak of a smoothed 2-D histogram of the cluster -- a robust anchor.

    The previous anchor (member nearest the arithmetic mean) lands in empty space for crescent or
    multi-island clusters, because the mean falls off-support. Binning the members onto a fixed grid,
    smoothing, and scoring each MEMBER by the smoothed density at its own bin guarantees an on-support
    anchor inside the dominant lobe (scoring bins rather than members could pick an empty bin after
    smoothing). np.argmax's first-index rule makes ties deterministic. Falls back to the old
    nearest-to-mean member if scipy is unavailable or the histogram degenerates."""
    try:
        from scipy.ndimage import gaussian_filter
        hist, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=grid, range=extent)
        hs = gaussian_filter(hist, sigma=smooth)
        ix = np.clip(np.searchsorted(xe, pts[:, 0], side="right") - 1, 0, grid - 1)
        iy = np.clip(np.searchsorted(ye, pts[:, 1], side="right") - 1, 0, grid - 1)
        return pts[int(np.argmax(hs[ix, iy]))]
    except Exception:                                            # scipy absent / degenerate extent
        return pts[np.argmin(((pts - pts.mean(0)) ** 2).sum(1))]


def _lift_label_display(raw: str) -> Optional[str]:
    r"""Render one decoded token for a cluster label, keeping BPE word-boundary information.

    The old ``strip()``-everything rendering made bare punctuation ("=", ",") read as noise and made
    continuation subwords ("omach", "ing") indistinguishable from whole words (GPT-2 BPE encodes the
    word boundary as a leading space, which strip destroyed). Punctuation-only tokens are repr-quoted
    (','), continuation subwords (letters, no leading space) get a middle-dot prefix (·ing); the
    classification reuses :func:`_bpe_category` so boundary semantics live in one place. Returns None
    for empty / replacement-char / non-printable BPE-byte fragments (dropped)."""
    core = raw.strip()
    if not core or "�" in core or not core.isprintable():
        return None
    cat = _bpe_category(raw)
    if cat == 0:                                                 # punctuation-only -> quote the glyphs
        return repr(core)
    if cat == 4:                                                 # continuation subword -> mark boundary
        return "·" + core
    return core


def _cluster_lift_labels(
    token_ids: np.ndarray,               # (M,) token ids
    labels:    np.ndarray,               # (M,) cluster labels (-1 = noise)
    decode:    Optional[object],         # decode([id]) -> str; None -> raw id string

    *,
    k:          int = 3,
    floor:      int = 2,
    with_stats: bool = False,
) -> Dict:
    r"""Per-cluster DISTINCTIVE-token label by enrichment: one comma-joined string per cluster.

    The globally frequent tokens (the, comma, of) occur in EVERY cluster, so raw-frequency labels are
    uninformative. Ranking is by the smoothed log-odds of membership,
    ``log((ct+a)/(n_c+a*V)) - log((glob-ct+a)/((M-n_c)+a*V))`` with ``a = 0.5`` and ``V`` the observed
    token types -- the additive smoothing breaks the raw-lift degeneracy where every cluster-exclusive
    token ties at exactly lift = M/n_c regardless of count (audit 2026-07-11: the old
    ``scored.sort(reverse=True)`` then broke those ties by decoded string DESCENDING, deterministically
    promoting rare high-codepoint accented glyphs -- the "ū, ō, ī" labels). Residual ties break by
    global count then FORWARD lexicographic order. Keeps candidates with in-cluster count >= ``floor``;
    tokens render via :func:`_lift_label_display`. ``decode is None`` -> raw id strings.
    ``with_stats=True`` returns ``{cluster: (label, top raw lift)}`` (the raw lift of the top-ranked
    token, for the caller's mixed-core gate); default returns ``{cluster: label}``.
    """
    ids = token_ids.astype(int)
    M = ids.size
    uniq, counts = np.unique(ids, return_counts=True)
    sm   = {int(t): (decode([int(t)]) if decode is not None else str(int(t))) for t in uniq}
    glob = {int(t): int(ct) for t, ct in zip(uniq.tolist(), counts.tolist())}
    a = 0.5
    V = uniq.size
    out: Dict = {}
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
            s = _lift_label_display(str(sm[int(t)]))
            if s is None:
                continue
            g = glob[int(t)]
            score = (np.log((ct + a) / (n_c + a * V))
                     - np.log((g - ct + a) / ((M - n_c) + a * V)))
            lift = (ct / n_c) / (g / M)
            scored.append((score, g, s, lift))
        scored.sort(key=lambda r: (-r[0], -r[1], r[2]))          # score desc, count desc, A->Z
        seen, toks, top_lift = set(), [], 0.0
        for _, _, s, lift in scored:
            if s not in seen:
                seen.add(s)
                toks.append(s)
                if len(toks) == 1:
                    top_lift = float(lift)
            if len(toks) >= k:
                break
        if toks:
            out[c] = (", ".join(toks), top_lift) if with_stats else ", ".join(toks)
    return out


@register_figure("belief_umap")
def plot_belief_umap(
    bank:             Dict,              # belief_bank output: mu, sigma, phi, token_ids, seq_idx
    channel:          str = "mu",        # which belief channel to embed: 'mu' / 'sigma' / 'phi'

    *,
    kind:             str              = "Belief",  # title noun: 'Belief' (q channel) / 'Model' (s channel)
    decode:           Optional[object] = None,   # decode(list[int]) -> str; None -> id labels
    n_clusters_label: int              = 14,     # legend/badge rows for the N largest clusters
    seed:             int              = 0,
    sil_sample:       int              = 2000,
    path:             Optional[str]    = None,
):
    r"""F5: data-driven cluster map of one belief channel -- numbered badges plus a legend band.

    ONE figure per channel (the caller emits mu / sigma / phi separately). The channel is embedded
    faithfully to its geometry (mu Euclidean, Sigma in the log-Euclidean chart, phi in the gauge
    coordinates). Clusters are NOT computed on the 2-D display embedding (which tears and compresses
    distances -- the documented UMAP anti-pattern): they come from a separate seeded clustering-space
    embedding of the SAME features (min_dist=0, up to 10 components; the native features directly when
    they are already that low-dimensional), so one cluster may legitimately render as several 2-D
    islands. Each of the ``n_clusters_label`` largest clusters gets a numbered badge at its density-peak
    member (:func:`_density_peak_anchor`) and a legend row -- number, color swatch, distinctive tokens
    by smoothed log-odds enrichment (:func:`_cluster_lift_labels`), and size -- replacing the old
    margin callouts whose full-span placement produced whole-plot leader lines. A dominant cluster
    whose top token is not actually distinctive (raw lift < 1.5, or >25% of the bank) is labelled
    "mixed core" instead of enrichment junk. Thin gray contours show DISPLAY-SPACE point density
    (overplotting relief only -- 2-D UMAP does not preserve feature-space density); per-cluster point
    size/alpha scale with population so the core stays translucent. The parameter footer states the
    embedding, clustering, bank size, and per-channel metric; the function/content silhouette (the
    a-priori-categories-do-not-separate caveat) is a small footnote when ``decode`` is available, with
    the quantitative view in :func:`plot_belief_category_separation`. ``decode is None`` -> raw id labels.
    """
    feats = _belief_channel_features(bank, channel)
    X = _np(feats).astype(float)
    M = X.shape[0]
    token_ids = _np(bank["token_ids"]).astype(int)
    n_disp = int(np.clip(round(np.sqrt(max(M, 1)) / 4.0), _UMAP_N_NEIGHBORS, 100))
    coords = _np(umap_embed(feats, n_neighbors=n_disp, seed=seed)).astype(float)
    if coords.shape[1] < 2:                                      # 1-D feature channel / M<=2: the
        coords = np.column_stack([coords, np.zeros(len(coords))])   # n_components clamp returns (M,1)
    cl_dim = min(10, X.shape[1], max(1, M - 1))
    if cl_dim >= X.shape[1]:                                     # features already low-D: cluster them directly
        cluster_coords, cluster_space = X, f"native {X.shape[1]}-D features"
    else:
        try:
            cluster_coords = _np(umap_embed(feats, n_neighbors=30, min_dist=0.0,
                                            n_components=cl_dim, seed=seed)).astype(float)
            cluster_space = f"{cl_dim}-D UMAP (min_dist=0)"
        except Exception:                                        # clustering embed failed -> status quo
            cluster_coords, cluster_space = coords, "2-D display embedding"
    labels, method_desc = _cluster_embedding(cluster_coords, seed=seed)
    cl = sorted(set(labels.tolist()) - {-1}, key=lambda c: -int((labels == c).sum()))
    noise = float((labels == -1).mean())
    lab_stats = _cluster_lift_labels(token_ids, labels, decode, k=3, with_stats=True)

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    fig.subplots_adjust(left=0.02, right=0.68, top=0.92, bottom=0.07)
    xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
    ymin, ymax = float(coords[:, 1].min()), float(coords[:, 1].max())
    rx, ry = (xmax - xmin) or 1.0, (ymax - ymin) or 1.0
    try:                                                         # display-space density relief (best-effort)
        from scipy.stats import gaussian_kde
        rng = np.random.default_rng(seed)
        sub = coords if M <= 8000 else coords[rng.choice(M, size=8000, replace=False)]
        gx, gy = np.mgrid[xmin:xmax:120j, ymin:ymax:120j]
        zz = gaussian_kde(sub.T)(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        lv = np.linspace(0.0, float(zz.max()), 10)[2:]           # drop the lowest bands: keep the page white
        ax.contour(gx, gy, zz, levels=lv, colors="0.45", linewidths=0.5, alpha=0.6, zorder=3)
    except Exception:                                            # singular cloud / no scipy -> skip relief
        pass
    nm = labels == -1
    if nm.any():
        ax.scatter(coords[nm, 0], coords[nm, 1], s=4, c="#bbbbbb", alpha=0.30, linewidths=0,
                   rasterized=True, zorder=1)
    palette = plt.cm.tab20(np.linspace(0, 1, 20))[_TAB20_DISTINCT]
    col = {c: palette[i % 20] for i, c in enumerate(cl)}
    for c in cl:
        m = labels == c
        n_c = int(m.sum())
        s_pt = float(np.clip(9.0 - 1.2 * np.log10(max(n_c, 1)), 2.5, 8.0))
        a_pt = float(np.clip(2500.0 / max(n_c, 1), 0.10, 0.75))
        ax.scatter(coords[m, 0], coords[m, 1], s=s_pt, color=col[c], alpha=a_pt, linewidths=0,
                   rasterized=True, zorder=2)
    # Numbered badges at density-peak anchors + legend band on the right. Badge numbers are the
    # descending-size rank (deterministic: seeded embeddings, RNG-free HDBSCAN, stable sort). A
    # fixed-order greedy nudge de-overlaps badges; a short leader marks any nudged badge's anchor.
    extent = [[xmin, xmax], [ymin, ymax]]
    r_sep = 0.045 * max(rx, ry)
    placed: list = []
    rows:   list = []
    for rank, c in enumerate(cl[:n_clusters_label], start=1):
        pts = coords[labels == c]
        anchor = _density_peak_anchor(pts, extent).astype(float)
        pos = anchor.copy()
        for _ in range(24):
            clash = next((q for q in placed
                          if abs(pos[0] - q[0]) < r_sep and abs(pos[1] - q[1]) < r_sep), None)
            if clash is None:
                break
            d = pos - clash
            nrm = float(np.hypot(*d))
            pos = pos + (d / nrm) * r_sep if nrm > 0 else pos + np.array([r_sep, 0.0])
        pos = np.array([np.clip(pos[0], xmin, xmax), np.clip(pos[1], ymin, ymax)])  # keep badges on-axes
        placed.append(pos)
        n_c = int((labels == c).sum())
        lab, top_lift = lab_stats.get(c, ("", 0.0))
        if not lab or n_c / M > 0.25 or top_lift < 1.5:          # nothing is CONCENTRATED here -> say so
            lab = "mixed core"
        if float(np.hypot(*(pos - anchor))) > 1e-9:
            ax.plot([anchor[0], pos[0]], [anchor[1], pos[1]], color=col[c], lw=0.7, alpha=0.8, zorder=5)
        ax.scatter([pos[0]], [pos[1]], s=150, marker="o", facecolor="white", edgecolor=col[c],
                   linewidths=1.5, zorder=6)
        ax.text(pos[0], pos[1], str(rank), fontsize=7.5, fontweight="bold",
                ha="center", va="center", zorder=7)
        rows.append((rank, c, lab, n_c))
    fig.text(0.70, 0.905, "clusters (by size) — distinctive tokens", fontsize=8.5,
             fontweight="bold", va="top")
    y = 0.865
    for rank, c, lab, n_c in rows:
        txt = lab if len(lab) <= 34 else lab[:33] + "…"
        fig.text(0.700, y, "■", color=col[c], fontsize=9, va="center")
        fig.text(0.716, y, f"{rank}. {txt}", fontsize=8, va="center")
        fig.text(0.716, y - 0.022, f"n={n_c:,} ({n_c / M:.0%})", fontsize=6.5, color="0.4", va="center")
        y -= 0.055
    if len(cl) > n_clusters_label:                               # no silent cap
        fig.text(0.70, y, f"+ {len(cl) - n_clusters_label} smaller clusters (points only)",
                 fontsize=7, color="0.4", va="center")
    if nm.any():
        ax.scatter([], [], s=12, c="#bbbbbb", label=f"unclustered ({noise:.0%})")
    ax.scatter([], [], s=60, marker="o", facecolor="white", edgecolor="0.35", linewidths=1.2,
               label="badge = cluster density peak")
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    ax.set_xlim(xmin - 0.04 * rx, xmax + 0.04 * rx)
    ax.set_ylim(ymin - 0.06 * ry, ymax + 0.06 * ry)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{kind} {channel} — {len(cl)} data-driven clusters, {noise:.0%} unclustered",
                 fontsize=11)
    if decode is not None:
        try:
            cats, _ = _token_category_labels(token_ids, decode, "function_content")
            sil = clustering_metrics(X, cats, sample_size=sil_sample)["silhouette"]
            fig.text(0.02, 0.038, f"function/content category silhouette {sil:+.2f} in native space "
                                  f"(~0 -> a-priori linguistic categories do not separate this channel)",
                     fontsize=6.5, color="0.4", ha="left", va="bottom")
        except Exception:
            pass
    metric = {"mu": "Euclidean", "sigma": "log-Euclidean vech", "phi": "gauge coords"}.get(channel, "Euclidean")
    n_seqs = int(np.unique(_np(bank["seq_idx"])).size) if "seq_idx" in bank else 0
    fig.text(0.98, 0.012,
             f"display: UMAP(n_neighbors={n_disp}, min_dist={_UMAP_MIN_DIST}, init=pca, seed={seed})"
             f" · clusters: {method_desc} in {cluster_space}"
             f" · M={M:,} tokens / {n_seqs} seqs · metric: {metric}",
             fontsize=6, color="0.4", ha="right", va="bottom")
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
    for k, c, lab in (("holonomy_deviation", _CB[0], r"$\|H-I\|_F$ (frame-dependent)"),
                      ("holonomy_wilson", _CB[1], r"Wilson $1-\mathrm{Re}\,\mathrm{Tr}\,H/K$ (gauge-invariant)")):
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
    handles, labels = ax.get_legend_handles_labels()
    if "belief_cond_median" in per_layer:
        axr = ax.twinx()
        # Name the axis in the label: this marker lives on the right (log) condition-number axis, not
        # the left rank/nats axis -- without it the lone triangle reads as a second entropy series.
        axr.plot(x, g("belief_cond_median"), ("^:" if L > 1 else "^"), color=_CB[1], lw=1.2,
                 label="cond median (right, log)")
        axr.set_ylabel("condition number")
        axr.set_yscale("log")
        axr.grid(False)
        h2, l2 = axr.get_legend_handles_labels()
        handles += h2
        labels += l2
    ax.legend(handles, labels, fontsize=8, frameon=False, loc="upper left")
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
        # The field is the transport's own curvature, ~0 for a flat (Regime-I) cocycle and genuinely
        # spread for regime_ii -- so the title must follow the run, not hard-code "regime II".
        field_regime = "regime II" if regime is not None else "flat"
        im = axes[0][2].imshow(_np(curvature), cmap="magma", aspect="auto")
        fig.colorbar(im, ax=axes[0][2], shrink=0.8, label=r"$\|H-I\|_F$")
        axes[0][2].set(xlabel="j", ylabel="i", title=f"Curvature field ({field_regime})")
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
    log-log fit if scipy is absent or the solve fails. Degenerate input (<2 distinct finite positive
    sizes) returns NaNs so the caller can skip the overlay; malformed weights raise ``ValueError``.
    """
    N = _np(N).astype(float).reshape(-1)
    L = _np(L).astype(float).reshape(-1)
    if N.size != L.size:
        raise ValueError(f"N and L must have equal lengths, got {N.size} and {L.size}")
    m = np.isfinite(N) & np.isfinite(L) & (N > 0) & (L > 0)
    w_all = None if weights is None else _np(weights).astype(float).reshape(-1)
    if w_all is not None and w_all.size != N.size:
        raise ValueError(f"weights must have length {N.size}, got {w_all.size}")
    N, L = N[m], L[m]
    w = None if w_all is None else w_all[m]
    if (w is not None and w.size > 0
            and (not np.all(np.isfinite(w)) or np.any(w < 0.0) or not np.any(w > 0.0))):
        raise ValueError("filtered weights must be finite, nonnegative, and contain a positive value")
    n_distinct_sizes = int(np.unique(N).size)
    out: Dict[str, object] = {"alpha": float("nan"), "A": float("nan"), "E": 0.0,
                              "r2": float("nan"), "n_points": int(N.size),
                              "n_distinct_sizes": n_distinct_sizes, "form": "power_law"}
    if with_offset and n_distinct_sizes < 4:
        out["form"] = "power_law_fallback_underdetermined"
    if N.size < 2 or n_distinct_sizes < 2:
        return out
    x, y = np.log(N), np.log(L)
    if with_offset and n_distinct_sizes >= 4:
        try:
            from scipy.optimize import curve_fit
            p0 = [max(0.0, float(L.min()) * 0.5), float(np.exp(np.mean(y))), 0.3]
            sig = None if w is None else (L / np.sqrt(np.clip(w, 1e-300, None)))   # m29: WLS (SEM=L/sqrt(w)); ignored when None
            popt, _ = curve_fit(lambda n, E, A, al: E + A * np.power(n, -al), N, L, p0=p0,
                                sigma=sig, absolute_sigma=False,
                                bounds=([0.0, 1e-12, 1e-3], [float(L.min()), np.inf, 3.0]),
                                maxfev=20000)
            E, A, al = (float(v) for v in popt)
            Lp = E + A * np.power(N, -al)
            ss_res = float(np.sum((L - Lp) ** 2))
            ss_tot = float(np.sum((L - L.mean()) ** 2))
            return {"alpha": al, "A": A, "E": E, "n_points": int(N.size),
                    "n_distinct_sizes": n_distinct_sizes, "form": "offset_power_law",
                    "r2": (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))}
        except Exception:
            out["form"] = "power_law_fallback_solver"
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
        try:
            xv = float(p.get(x_key, float("nan")))
        except (TypeError, ValueError):
            continue
        if not seeds or not np.isfinite(xv) or xv <= 0.0:
            continue
        arr = np.asarray(seeds, dtype=float)
        x.append(xv); mean.append(float(arr.mean())); ns.append(arr.size)
        s = float(arr.std(ddof=1)) / np.sqrt(arr.size) if arr.size > 1 else 0.0
        sem.append(s); ci.append(_t95(arr.size) * s)
        sx.extend([xv] * arr.size); sy.extend(seeds)
    return {"x": np.asarray(x), "mean": np.asarray(mean), "sem": np.asarray(sem),
            "ci95": np.asarray(ci), "n": np.asarray(ns, dtype=int),
            "seed_x": np.asarray(sx), "seed_y": np.asarray(sy)}


def _scaling_sem_weights(mean: object, sem: object) -> np.ndarray:
    r"""Inverse-variance weights for a log-loss fit from point means and their SEMs.

    By the delta method, ``Var(log(mean)) ~= (SEM / mean)^2``, so the WLS
    weight is ``(mean / SEM)^2``.  A zero SEM receives unit weight instead of
    infinite leverage, matching the pre-existing headline-fit policy.
    """
    mean_arr = _np(mean).astype(float).reshape(-1)
    sem_arr = _np(sem).astype(float).reshape(-1)
    if mean_arr.size != sem_arr.size:
        raise ValueError(f"mean and sem must have equal lengths, got {mean_arr.size} and {sem_arr.size}")
    weights = np.ones_like(mean_arr)
    positive = np.isfinite(mean_arr) & np.isfinite(sem_arr) & (mean_arr > 0.0) & (sem_arr > 0.0)
    weights[positive] = (mean_arr[positive] / sem_arr[positive]) ** 2
    return weights


def _scaling_route_label(route: str) -> str:
    if route == "blocks_K48_tied_2x":
        return f"{route} (tied structural ablation)"
    return route


def _scaling_fit_form_label(form: object) -> str:
    labels = {
        "offset_power_law": "offset power law",
        "power_law": "power law",
        "power_law_fallback_underdetermined": "power-law fallback (underdetermined)",
        "power_law_fallback_solver": "power-law fallback (solver failure)",
    }
    return labels.get(str(form), str(form).replace("_", " "))


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
    if np.unique(st["x"]).size >= 2:
        w = _scaling_sem_weights(st["mean"], st["sem"])
        fit = _fit_power_law(st["x"], st["mean"], weights=w, with_offset=with_offset)
        xx = np.geomspace(st["x"].min(), st["x"].max(), 100)
        yy = fit["E"] + fit["A"] * np.power(xx, -fit["alpha"])
        lbl = (rf"fit $\alpha$={fit['alpha']:.3f}, $R^2$={fit['r2']:.3f}"
               + (f", E={fit['E']:.3f}" if fit["form"] == "offset_power_law" else "")
               + f", {_scaling_fit_form_label(fit['form'])}")
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
    with_offset:      bool,

    x_key:            str = "n_params",
    xlabel:           str = "parameters N",
    title:            str = "Routes to N: does the frontier collapse?",

    weights_by_route: Optional[Mapping[str, np.ndarray]] = None,
    path:             Optional[str]                      = None,
):
    r"""Multi-route overlay: ``test_ce`` vs ``N`` with points colored AND markered by which knob grew
    ``N`` (embed_dim / gauge block size / depth / ...), each route's own power-law fit, and a dashed
    pooled fit. Routes that fall on one line share a frontier; a route offset above/below it (or a
    different slope) does not -- the visual companion to the analyzer's ANCOVA test. Marker shape
    duplicates the color so the routes survive greyscale printing."""
    routes = sorted({str(p.get("route", "?")) for p in points})
    route_stats = []
    for r in routes:
        st = _scaling_point_stats([p for p in points if str(p.get("route", "?")) == r], x_key)
        if not st["x"].size:
            continue
        if weights_by_route is None:
            weights = _scaling_sem_weights(st["mean"], st["sem"])
        else:
            if r not in weights_by_route:
                raise ValueError(f"weights_by_route is missing route {r!r}")
            weights = _np(weights_by_route[r]).astype(float).reshape(-1)
            if weights.size != st["x"].size:
                raise ValueError(
                    f"weights_by_route[{r!r}] has length {weights.size}; "
                    f"expected {st['x'].size} after point filtering"
                )
            if (not np.all(np.isfinite(weights)) or np.any(weights < 0.0)
                    or not np.any(weights > 0.0)):
                raise ValueError(
                    f"weights_by_route[{r!r}] must be finite, nonnegative, and contain a positive value"
                )
        route_stats.append((r, st, weights))

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for i, (r, st, weights) in enumerate(route_stats):
        c, mk = _CB[i % len(_CB)], markers[i % len(markers)]
        ax.errorbar(st["x"], st["mean"], yerr=st["ci95"], fmt=mk, color=c, ms=6, capsize=2,
                    lw=0, elinewidth=1.0)
        if np.unique(st["x"]).size >= 2:
            fit = _fit_power_law(st["x"], st["mean"], weights=weights, with_offset=with_offset)
            xx = np.geomspace(st["x"].min(), st["x"].max(), 60)
            yy = fit["E"] + fit["A"] * np.power(xx, -fit["alpha"])
            ax.plot(xx, yy, color=c, lw=1.4,
                    label=rf"{_scaling_route_label(r)} ($\alpha$={fit['alpha']:.3f}; "
                          rf"{_scaling_fit_form_label(fit['form'])})")
        else:
            ax.plot([], [], marker=mk, color=c, lw=0, label=f"{_scaling_route_label(r)} (1 pt)")
    if route_stats:
        pooled_x = np.concatenate([st["x"] for _, st, _ in route_stats])
        pooled_mean = np.concatenate([st["mean"] for _, st, _ in route_stats])
        pooled_weights = np.concatenate([weights for _, _, weights in route_stats])
    else:
        pooled_x = pooled_mean = pooled_weights = np.array([], dtype=float)
    if np.unique(pooled_x).size >= 2:
        g = _fit_power_law(pooled_x, pooled_mean, weights=pooled_weights, with_offset=with_offset)
        xx = np.geomspace(pooled_x.min(), pooled_x.max(), 60)
        yy = g["E"] + g["A"] * np.power(xx, -g["alpha"])
        ax.plot(xx, yy, "--", color="#444444", lw=1.6,
                label=rf"pooled ($\alpha$={g['alpha']:.3f}; {_scaling_fit_form_label(g['form'])})")
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
    _zoom_bar_value_axis(ax, canon + surr, hi_frac=0.20)   # padded window + headroom for the Δ labels
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


# ===========================================================================
# Lower-priority experiment figures (auto-dispatched after their sweep / analysis)
#   A1/EXP-2 gauge transport · A2/EXP-9 equivariance drift · A3/EXP-10 CG coupling
#   H2/EXP-11 kappa dispersion · F1/EXP-6 muP K-stability · I1/EXP-1 noise band
# ===========================================================================

@register_figure("gauge_transport_bars")
def plot_gauge_transport_bars(
    cells,                               # list of {mode in {on,frozen,off}, depth, ppl, omega_dev}

    *,
    path: Optional[str] = None,
):
    r"""A1/EXP-2: gauge ON vs frozen-random vs OFF(Omega=I) validation PPL, grouped by depth L.

    The program's central causal probe: does trained GL(K) transport beat the exact Omega=I control?
    The OFF arm's max|Omega-I| (omega_dev) is annotated to certify the frame really collapsed to the
    identity to float eps. Predict ON < FROZEN <= OFF if transport carries the advantage."""
    cells = list(cells)
    modes = [("on", _CB[2]), ("frozen", _CB[4]), ("off", _CB[1])]
    depths = sorted({str(c["depth"]) for c in cells})

    def _v(mode, depth, key):
        m = [c for c in cells if c["mode"] == mode and str(c["depth"]) == depth]
        return float(m[0][key]) if m and m[0].get(key) is not None else float("nan")
    x = np.arange(len(depths)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for i, (mode, col) in enumerate(modes):
        ax.bar(x + (i - 1) * w, [_v(mode, d, "ppl") for d in depths], w, color=col, label=mode)
    for di, d in enumerate(depths):
        dev = _v("off", d, "omega_dev")
        if np.isfinite(dev):
            ax.annotate(rf"$|\Omega-I|$={dev:.0e}", (x[di] + w, _v("off", d, "ppl")),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(depths)
    _zoom_bar_value_axis(                                  # padded window + headroom for the |Ω-I| labels
        ax, [_v(m, d, "ppl") for m, _ in modes for d in depths], hi_frac=0.20)
    ax.set(xlabel="depth", ylabel="validation PPL")
    ax.set_title("GL(K) gauge transport: ON vs frozen vs OFF (Ω=I)")
    ax.legend(fontsize=8, frameon=False, title="gauge")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("gauge_residual_drift")
def plot_gauge_residual_drift(
    arms,                                # list of {label, step:[...], resid:[...]}

    *,
    path: Optional[str] = None,
):
    r"""A2/EXP-9: builder-break gauge residual vs training step, tied (exact) vs untied (mixer drift).

    Step 0 is byte-identical (~eps) for both arms (identity-init mixer); only the untied block_glk arm
    climbs as the head mixer drifts from the identity and breaks equivariance. log-y so the climb from
    float-eps is legible."""
    arms = [a for a in arms if len(a.get("resid", [])) >= 1]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for j, a in enumerate(arms):
        s = np.asarray(a["step"], float)
        r = np.clip(np.asarray(a["resid"], float), 1e-12, None)
        order = np.argsort(s)
        ax.plot(s[order], r[order], "o-", color=_CB[j % len(_CB)], lw=1.6, ms=4, label=str(a.get("label", j)))
    ax.set_yscale("log")
    ax.set(xlabel="training step", ylabel="builder-break gauge residual (log)")
    ax.set_title("Tied (exact) vs untied (mixer drift) gauge equivariance")
    if arms:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("ppl_equivariance_bars")
def plot_ppl_equivariance_bars(
    cells,                               # list of {label, ppl, resid}

    *,
    path: Optional[str] = None,
):
    r"""A3/EXP-10: per-arm validation PPL with the median gauge equivariance residual on a twin axis.

    The Clebsch-Gordan cross-irrep coupling is exactly equivariant for ANY learned path weights, so the
    residual stays ~eps on both arms (off/on) and the bar contrast is a clean PPL-delta at matched
    equivariance -- the capacity the only exactly-equivariant inequivalent-irrep channel buys."""
    cells = list(cells)
    labels = [str(c["label"]) for c in cells]
    x = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ppls = [float(c["ppl"]) for c in cells]
    ax.bar(x, ppls, 0.5, color=_CB[0], label="val PPL")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    _zoom_bar_value_axis(ax, ppls, hi_frac=0.15)          # padded window so near-equal PPLs are legible
    for xi, v in zip(x, ppls):
        ax.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, clip_on=False)
    ax.set_ylabel("validation PPL")
    resid = [float(c["resid"]) if np.isfinite(_as_f(c.get("resid"))) else np.nan for c in cells]
    if any(np.isfinite(resid)):
        axr = ax.twinx()
        axr.plot(x, resid, "s--", color=_CB[1], lw=1.4, label="equivariance residual")
        axr.set_ylabel("median equivariance residual")
        axr.set_yscale("log"); axr.grid(False)
    ax.set_title("Clebsch-Gordan coupling: PPL + equivariance residual")
    fig.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("kappa_dispersion")
def plot_kappa_dispersion(
    cells,                               # list of {label, dispersion, ppl}

    *,
    path: Optional[str] = None,
):
    r"""H2/EXP-11: validation PPL vs per-head temperature dispersion std(kappa_beta).

    Tied arms (uniform, and the geo-mean-tau confound control) sit at dispersion 0; the dispersed arms
    hold the arithmetic mean at 1.0 so the x-axis isolates per-head asymmetry. A separation between the
    tied baselines and the dispersed arms is the per-head-temperature effect (the geo-mean baseline
    controls for the mean-tau shift the dispersed arms carry)."""
    cells = list(cells)
    pts = sorted([(float(c["dispersion"]), float(c["ppl"]), str(c.get("label", ""))) for c in cells],
                 key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=_CB[0], lw=1.6)
    for d, y, lab in pts:
        ax.annotate(lab, (d, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set(xlabel=r"per-head temperature dispersion std$(\kappa_\beta)$", ylabel="validation PPL")
    ax.set_title("Per-head temperature dispersion vs PPL")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("kmup_stability")
def plot_kmup_stability(
    routes,                              # dict {route: [{embed_dim, ppl_mean, ppl_sem}...]}

    *,
    path: Optional[str] = None,
):
    r"""F1/EXP-6: inverse-K scaling width-fixed (grow_K) vs muP-corrected (grow_K_mup) on a shared
    K=embed_dim axis.

    Test PPL vs K per route with cross-seed 95% CI bars (the small-sample t-quantile, not a fixed 1.96)
    and a requested offset power-law fit ``PPL = A K^{-alpha} + E`` overlaid when at least four
    distinct widths identify it. Smaller grids, including the live three-size ``grow_K`` route, show
    and label the realized log-log fallback without reporting an irreducible floor. The exponent
    ``b = -alpha`` is annotated per route. The grow_K_mup route is split into its matched /fixed and /mup arms upstream, so
    |b_fixed - b_muP| -- the width-stability readout -- is read directly off the two muP curves; a large
    gap means the headline exponent is partly optimization mis-tuning the muP correction removes."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for j, (name, pts) in enumerate(routes.items()):
        pts = sorted(pts, key=lambda p: float(p["embed_dim"]))
        K = np.array([float(p["embed_dim"]) for p in pts], float)
        y = np.array([float(p["ppl_mean"]) for p in pts], float)
        ci = np.array([_t95(int(p.get("n", 2))) * float(p.get("ppl_sem", 0.0)) for p in pts], float)
        col = _CB[j % len(_CB)]
        ax.errorbar(K, y, yerr=ci, fmt="o", color=col, capsize=3, lw=1.4, label=name)
        if np.unique(K).size >= 2:
            fit = _fit_power_law(K, y, with_offset=True)
            a = fit.get("alpha", float("nan"))
            if np.isfinite(a):
                Kf = np.linspace(float(K.min()), float(K.max()), 100)
                ax.plot(Kf, fit["A"] * Kf ** (-a) + fit.get("E", 0.0), "-", color=col, lw=1.3, alpha=0.7)
                ax.plot([], [], " ",
                        label=f"   {name}: {_scaling_fit_form_label(fit['form'])}, b={-a:+.3f}")
    ax.set(xlabel="K (embed_dim)", ylabel="test PPL")
    ax.set_title("μP width-stability of the inverse-K exponent")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("ppl_offset")
def plot_ppl_offset(
    points: list,                        # [{embed_dim, ppl_mean, [ppl_sem], [n_seeds]}, ...] pooled param points

    *,
    path:   Optional[str] = None,
):
    r"""Pooled test-PPL fit against width over all parameter points.

    The figure requests ``PPL = E + A K^{-alpha}`` when at least four distinct widths identify the
    three-parameter form and otherwise labels the realized log-log fallback without an ``E`` estimate.
    It pools every parameter route -- distinct from ``kmup_stability``, which splits
    the muP arms to read the width-stability contrast. Each point carries the across-seed mean PPL with
    a small-sample t-quantile CI when seeds are present; the realized fit and exponent ``b`` are overlaid."""
    pts = sorted([p for p in points
                  if np.isfinite(float(p.get("embed_dim", float("nan"))))
                  and np.isfinite(float(p.get("ppl_mean", float("nan")))) and float(p["ppl_mean"]) > 0],
                 key=lambda p: float(p["embed_dim"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if not pts:
        ax.text(0.5, 0.5, "no PPL-vs-embed_dim points", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    K = np.array([float(p["embed_dim"]) for p in pts], float)
    y = np.array([float(p["ppl_mean"]) for p in pts], float)
    ci = np.array([_t95(int(p.get("n_seeds", 2))) * float(p.get("ppl_sem", 0.0)) for p in pts], float)
    ax.errorbar(K, y, yerr=ci, fmt="o", color=_CB[0], capsize=3, lw=1.4, label="test PPL (across-seed)")
    if K.size >= 2 and float(K.max()) > float(K.min()):
        fit = _fit_power_law(K, y, with_offset=True)
        a = fit.get("alpha", float("nan"))
        if np.isfinite(a):
            Kf = np.linspace(float(K.min()), float(K.max()), 100)
            fit_label = f"{_scaling_fit_form_label(fit['form'])}  b={-a:+.3f}"
            if fit["form"] == "offset_power_law":
                fit_label += f"  E={fit['E']:.3f}"
            ax.plot(Kf, fit["A"] * Kf ** (-a) + fit.get("E", 0.0), "-", color=_CB[1], lw=1.6,
                    label=fit_label)
    ax.set(xlabel="K (embed_dim)", ylabel="test PPL", title="Pooled PPL fit vs width")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


# =============================================================================
# Multi-seed digest: publication labels.
# Raw metrics.csv column / research scalar key -> publication-quality (mathtext) display label
# used for the across-seed digest figures (titles, axes, bar ticks). One source of truth so the
# helpers below render manuscript notation -- KL(q_i||p_i), sum_ij beta_ij KL(q_i||Omega_ij q_j),
# KL(s_i||h), ... -- instead of the raw column names (gamma_coupling, ...). Math is $...$ mathtext;
# pub_label() falls back to a humanized name for any key not listed. Notation matches the canonical
# free energy: q_i belief, p_i prior, s_i model, r model-fiber prior centroid, Omega_ij transport,
# beta_ij/gamma_ij belief/model attention, pi_ij prior attention, tau softmax temperature,
# alpha self-coupling weight, Sigma belief covariance (Sigma^{-1} precision / mean-block Fisher).
PUB_LABELS: Dict[str, str] = {
    # -- belief-channel free-energy blocks (raw per-token block energies) --
    "self_coupling":      r"$\sum_i \alpha_i\,\mathrm{KL}(q_i\|p_i)$",
    "self_divergence":    r"$\sum_i \mathrm{KL}(q_i\|p_i)$",
    "belief_coupling":    r"$\sum_{ij}\beta_{ij}\,\mathrm{KL}(q_i\|\Omega_{ij}q_j)$",
    "attention_entropy":  r"$\tau\sum_{ij}\beta_{ij}\log(\beta_{ij}/\pi_{ij})$",
    "free_energy_total":  r"free energy $F$ (nats/token)",
    # -- model-channel (s) free-energy blocks (r = model-fiber prior centroid; manuscript symbol) --
    "hyper_prior":        r"$\mathrm{KL}(s_i\|r)$",
    "hyper_prior_weighted": r"$\lambda_h\,\mathrm{KL}(s_i\|r)$",
    "gamma_coupling":     r"$\sum_{ij}\gamma_{ij}\,\mathrm{KL}(s_i\|\Omega_{ij}s_j)$",
    "gamma_meta_entropy": r"$\tau_g\sum_{ij}\gamma_{ij}\log(\gamma_{ij}/\pi^{s}_{ij})$",
    # -- attention / gauge / belief-geometry diagnostics --
    "attn_entropy":       r"attention row entropy $H(\beta)=-\sum_j\beta_{ij}\log\beta_{ij}$",
    "effective_rank":     r"belief effective rank $\mathrm{erank}(\Sigma)$",
    "holonomy_deviation": r"holonomy $\langle\|H-I\|_F\rangle$ (frame-dependent)",
    "gauge_trace_spread": r"gauge spread $\mathrm{std}_i\,\log|\det\Omega_i|$",
    "belief_cond_median": r"belief conditioning $\mathrm{med}_i\,\kappa(\Sigma_i)$",
    "fisher_trace_mean":  r"Half Fisher trace $\langle\mathrm{tr}\,\Sigma^{-1}\rangle/2$",
    "grad_norm":          r"gradient norm $\|\nabla\|_2$",
    # -- losses / headline scalars --
    "train_ce":           "train cross-entropy (nats)",
    "val_ppl":            "validation perplexity",
    "generalization_gap": r"generalization gap $\mathrm{CE}_{\mathrm{val}}-\mathrm{CE}_{\mathrm{train}}$ (nats)",
    "test_ppl":           "test perplexity",
    "best_val_ppl":       "best validation perplexity",
    "test_ce":            "test cross-entropy (nats)",
    "test_bpc":           "test bits per character",
    "test_ce_no_estep":   "test CE without E-step (nats)",
    "estep_capacity_gain": "E-step capacity gain (nats)",
    "wall_time_s":        "wall-clock time (s)",
    "ece":                "expected calibration error",
    "overall_ce":         "overall cross-entropy (nats)",
    "sigma_trace_cv":     r"$\mathrm{CV}_i[\mathrm{tr}\,\Sigma_i]$",
    "sigma_ce_spearman":  r"$\rho_s(\mathrm{tr}\,\Sigma,\ \mathrm{CE})$",
    "fd_gradient_worst_rel_error": "finite-diff worst rel. error",
    "corpus_freq_strata_ce.rare":     r"$\mathrm{CE}_{\mathrm{rare}}$ (nats)",
    "corpus_freq_strata_ce.mid":      r"$\mathrm{CE}_{\mathrm{mid}}$ (nats)",
    "corpus_freq_strata_ce.frequent": r"$\mathrm{CE}_{\mathrm{frequent}}$ (nats)",
    # -- geometry-health dashboard (gauge / SPD / Fisher / guard) --
    "holonomy_wilson":         r"Wilson holonomy $1-\mathrm{Re}\,\mathrm{Tr}(H)/K$ (gauge-invariant)",
    "cocycle_residual":        "cocycle residual (flatness)",
    "gauge_invariant_spread":  r"gauge-invariant spread $\mathrm{std}_i\,I(\exp\phi_i)$",
    "phi_norm_mean":           r"gauge-frame norm $\langle\|\phi_i\|\rangle$",
    "belief_cond_p95":         r"belief conditioning $\kappa_{95}(\Sigma)$",
    "eff_rank_median":         r"belief effective rank $\mathrm{med}_i\,\mathrm{erank}(\Sigma_i)$",
    "fisher_trace_median":     r"Half Fisher trace $\mathrm{med}_i\,\mathrm{tr}\,\Sigma_i^{-1}/2$",
    "nonfinite_frac":          "non-finite fraction",
    "renyi_band_frac":         "Renyi cancellation-band fraction",
    "attn_entropy_min":        r"min row entropy $\min_{i,h}H(\beta)$",
    "attn_entropy_collapsed_heads": "collapsed-head count",
    # -- E-step inference quality --
    "estep_f_drop":            r"E-step $F_{\mathrm{end}}-F_{\mathrm{start}}$",
    "estep_f_nondecreasing_frac": "E-step nondecreasing fraction",
    "estep_r_mu_last":         r"E-step residual $r_\mu$ (last)",
    "estep_r_sigma_last":      r"E-step residual $r_\Sigma$ (last, SPD)",
    "estep_r_phi_last":        r"E-step residual $r_\phi$ (last)",
    # -- held-out validation sanity --
    "pos_loss_ratio":          r"positional loss ratio $\mathrm{CE}_{\mathrm{last}}/\mathrm{CE}_{\mathrm{first}}$",
    "val_future_leakage":      "causal future leakage (max)",
    "val_row_sum_error":       "attention row-sum error (max)",
    "val_pos_content_r2":      r"positional-content $R^2$",
    "val_head_redundancy_js":  "head redundancy (JS)",
    # -- optimizer information geometry --
    "cos_nat_phi":             r"$\cos(\nabla^{\mathrm{nat}}_\phi,\nabla_\phi)$",
    "pullback_cond_median":    r"pullback-metric conditioning $\mathrm{med}\,\kappa$",
}


def pub_label(name: str) -> str:
    r"""Publication-quality (mathtext) display label for a raw metric / column / scalar key.

    Returns the :data:`PUB_LABELS` entry, else a humanized fallback (underscores and dots -> spaces)
    so an unlisted column still renders a readable -- if non-mathematical -- axis."""
    return PUB_LABELS.get(name) or name.replace("_", " ").replace(".", " ")


@register_figure("ppl_noise_band")
def plot_ppl_noise_band(
    agg,                                 # aggregate_seed_metric output: {values, seeds, mean, sd}

    *,
    label: str  = "K=20 baseline",
    grid:  Optional[Dict] = None,        # optional {ablation label: single-seed ppl} overlay
    path:  Optional[str]  = None,
):
    r"""I1/EXP-1: per-seed test PPL with the across-seed mean ± 1 SD (and ± 2 SD) band -- the seed
    noise floor every ablation 'win' must clear.

    The band is the init+optimization spread (the per-run reseed shares the data order; a lower bound
    on deployment variance). An optional ``grid`` overlays single-seed ablation cells so a referee can
    read directly whether a cell's margin exceeds the band."""
    vals = np.asarray(agg.get("values", []), float)
    mean, sd = float(agg.get("mean", float("nan"))), float(agg.get("sd", float("nan")))
    disp = pub_label(label)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    if np.isfinite(mean):
        ax.axhline(mean, color=_CB[0], lw=1.6, label=f"mean={mean:.3f}")
        if np.isfinite(sd):
            ax.axhspan(mean - 2 * sd, mean + 2 * sd, color=_CB[0], alpha=0.08, label="±2 SD")
            ax.axhspan(mean - sd, mean + sd, color=_CB[0], alpha=0.18, label="±1 SD")
    seeds = agg.get("seeds", list(range(vals.size))) or list(range(vals.size))
    xs = np.arange(vals.size)
    ax.scatter(xs, vals, color=_CB[1], zorder=3, label="per-seed value")
    ticks, ticklabels = list(xs), [str(s) for s in seeds[:vals.size]]
    if grid:
        gl = list(grid.items())
        gx = np.arange(vals.size, vals.size + len(gl))
        ax.scatter(gx, [float(v) for _, v in gl], color=_CB[3], marker="^", zorder=3, label="ablation cells")
        ticks += list(gx); ticklabels += [str(k) for k, _ in gl]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, rotation=45, ha="right", fontsize=7)
    ax.set(xlabel="seed / cell", ylabel=disp)
    ax.set_title("Multi-seed variance floor (init + optimization spread)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_curve_band(
    steps: np.ndarray,                   # (S,) training-step grid
    mean:  np.ndarray,                   # (S,) across-seed mean per step
    sd:    np.ndarray,                   # (S,) across-seed SD (ddof=1; NaN where fewer than 2 seeds)

    *,
    label:  str  = "metric",
    ylabel: Optional[str] = None,        # y-axis; default -> pub_label(label) (publication math)
    n:      Optional[np.ndarray] = None, # (S,) seed count per step (for the title)
    logy:   bool = False,
    path:   Optional[str] = None,
):
    r"""Across-seed mean +/-1 SD (and +/-2 SD) ribbon of one training curve over steps.

    The band is the init+optimization spread (the per-run reseed shares the data order; a lower bound
    on deployment variance). Steps where only one seed reported (NaN SD, e.g. the sparse ``val_*``
    columns) draw the mean without a ribbon."""
    steps = np.asarray(steps, float)
    mean  = np.asarray(mean, float)
    sd    = np.asarray(sd, float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    finite = np.isfinite(mean)
    ax.plot(steps[finite], mean[finite], color=_CB[0], lw=1.6, label="mean")
    band = finite & np.isfinite(sd)
    if band.any():
        ax.fill_between(steps[band], (mean - 2 * sd)[band], (mean + 2 * sd)[band],
                        color=_CB[0], alpha=0.10, label="$\\pm 2$ SD")
        ax.fill_between(steps[band], (mean - sd)[band], (mean + sd)[band],
                        color=_CB[0], alpha=0.22, label="$\\pm 1$ SD")
    if logy:
        ax.set_yscale("log")
    n_seeds = None
    if n is not None and np.any(np.isfinite(n)):
        n_seeds = int(np.nanmax(np.asarray(n, float)))
    ylab  = ylabel if ylabel is not None else pub_label(label)
    title = "across-seed mean $\\pm 1$ SD" + (f" ($n={n_seeds}$)" if n_seeds else "")
    ax.set(xlabel="training step", ylabel=ylab, title=title)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_curve_band_grid(
    curves,                              # list of {steps, mean, sd, title, logy?}

    *,
    ncols: int = 3,
    path:  Optional[str] = None,
):
    r"""Multi-panel overview: one across-seed mean +/-1 SD ribbon per curve, shared step axis."""
    curves = list(curves)
    m = len(curves)
    ncols = max(1, min(ncols, m)) if m else 1
    nrows = int(np.ceil(m / ncols)) if m else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
    for ax in axes.flat[m:]:
        ax.axis("off")
    for ax, c in zip(axes.flat, curves):
        steps = np.asarray(c["steps"], float)
        mean  = np.asarray(c["mean"], float)
        sd    = np.asarray(c["sd"], float)
        finite = np.isfinite(mean)
        ax.plot(steps[finite], mean[finite], color=_CB[0], lw=1.4)
        band = finite & np.isfinite(sd)
        if band.any():
            ax.fill_between(steps[band], (mean - sd)[band], (mean + sd)[band], color=_CB[0], alpha=0.22)
        if c.get("logy"):
            ax.set_yscale("log")
        ax.set_title(pub_label(c.get("title", "")), fontsize=9)
        ax.tick_params(labelsize=7)
    fig.supxlabel("training step", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_scalar_cv_summary(
    aggs,                                # {name: {mean, sd, cv, values}}

    *,
    path: Optional[str] = None,
):
    r"""Per-scalar across-seed coefficient of variation (SD / |mean|, %) as a sorted horizontal bar,
    with each seed's absolute normalized deviation (|value/mean - 1|) overlaid -- the seed-stability
    ranking a referee reads to know which headline numbers are robust and which ride seed noise."""
    items = [(k, v) for k, v in aggs.items()
             if v.get("cv") is not None and np.isfinite(float(v.get("cv", np.nan)))]
    items.sort(key=lambda kv: kv[1]["cv"])
    names = [k for k, _ in items]
    cvs   = [100.0 * float(v["cv"]) for _, v in items]
    fig, ax = plt.subplots(figsize=(7.4, max(2.4, 0.45 * len(names) + 1.0)))
    ys = np.arange(len(names))
    ax.barh(ys, cvs, color=_CB[5], alpha=0.7, zorder=2, label="across-seed CV")
    for y, (_, v) in zip(ys, items):
        mean = float(v.get("mean", np.nan))
        if np.isfinite(mean) and mean != 0.0:
            dev = [abs(100.0 * (float(val) / mean - 1.0)) for val in v.get("values", [])]
            ax.scatter(dev, [y] * len(dev), color=_CB[1], s=14, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([pub_label(n) for n in names], fontsize=8)
    ax.set(xlabel="across-seed CV (% of mean); dots = |per-seed deviation|",
           title="Seed stability of headline scalars")
    fig.tight_layout()
    return _save(fig, path)


def plot_per_layer_band(
    per_layer,                           # {layer: {metric: {mean, sd, n, values}}}
    metric: str,

    *,
    path: Optional[str] = None,
):
    r"""Across-seed per-layer bars (mean) with +/-1 SD error bars for one ``metric``."""
    layers = sorted(per_layer.keys())
    means = [float(per_layer[l].get(metric, {}).get("mean", np.nan)) for l in layers]
    sds   = [float(per_layer[l].get(metric, {}).get("sd", np.nan)) for l in layers]
    sds   = [0.0 if not np.isfinite(s) else s for s in sds]
    fig, ax = plt.subplots(figsize=(max(4.0, 1.1 * len(layers) + 2.0), 4.0))
    xs = np.arange(len(layers))
    ax.bar(xs, means, yerr=sds, capsize=4, color=_CB[2], alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"layer {l}" for l in layers])
    ax.set(ylabel=pub_label(metric), title="per-layer mean across seeds ($\\pm 1$ SD)")
    fig.tight_layout()
    return _save(fig, path)


def _as_f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


@register_figure("pos_extrapolation")
def plot_pos_extrapolation(
    arms,                                # dict {label: [{n, ce}...]} OR list of {"label", "curve"}

    *,
    train_n: Optional[float] = None,
    path:    Optional[str]   = None,
):
    r"""H1/EXP-13: held-out CE vs eval sequence length N, one line per positional scheme.

    Offset attention priors (alibi / t5, functions of |i-j|) stay flat past the trained length; the
    absolute schemes (learned pos_phi table, RoPE) rise -- the extrapolation contrast. The train
    length is marked; points beyond it are pure extrapolation."""
    if isinstance(arms, dict):
        items = [(str(k), v) for k, v in arms.items()]
    else:
        items = [(str(a.get("label", i)), a.get("curve", [])) for i, a in enumerate(arms)]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for j, (lab, curve) in enumerate(items):
        pts = sorted([(float(p["n"]), float(p["ce"])) for p in curve
                      if np.isfinite(_as_f(p.get("ce")))], key=lambda t: t[0])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=_CB[j % len(_CB)], lw=1.6, ms=4, label=lab)
    if train_n is not None and np.isfinite(_as_f(train_n)):
        ax.axvline(float(train_n), color=_CB[7], ls="--", lw=1.0, alpha=0.6, label="train length")
    ax.set(xlabel="eval sequence length N", ylabel="held-out CE (nats)")
    ax.set_title("Positional extrapolation: offset vs absolute (CE vs N)")
    if items:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("renyi_saturation")
def plot_renyi_saturation(
    cells,                               # list of {alpha, attn_entropy, energy_klmax_frac}

    *,
    path: Optional[str] = None,
):
    r"""B2/EXP-12: attention entropy H(beta) and the kl_max energy-saturation fraction vs Renyi order.

    alpha<1 is mass-covering, alpha>1 mode-seeking; for alpha>1 the non-PD blend saturates the pairwise
    energy E_ij to kl_max with zero gradient (rising saturation fraction, right panel), which can drive
    a NON-MONOTONE H(beta)-vs-alpha tail (left panel) -- the saturation diagnostic explaining the
    entropy curve. alpha=1 (KL) is marked."""
    cells = sorted(cells, key=lambda c: float(c["alpha"]))
    a = [float(c["alpha"]) for c in cells]
    h = [float(c.get("attn_entropy", float("nan"))) for c in cells]
    s = [float(c.get("energy_klmax_frac", float("nan"))) for c in cells]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax1.plot(a, h, "o-", color=_CB[0], lw=1.8)
    ax1.axvline(1.0, color=_CB[7], ls="--", lw=1.0, alpha=0.5, label=r"$\alpha=1$ (KL)")
    ax1.set(xlabel=r"Rényi order $\alpha$", ylabel=r"attention entropy H($\beta$)")
    ax1.set_title("Attention diffuseness vs α")
    ax1.legend(fontsize=8, frameon=False)
    ax2.plot(a, s, "s-", color=_CB[1], lw=1.8)
    ax2.axvline(1.0, color=_CB[7], ls="--", lw=1.0, alpha=0.5)
    ax2.set(xlabel=r"Rényi order $\alpha$", ylabel="energy kl_max saturation fraction")
    ax2.set_title("Non-PD saturation vs α (the α>1 tail)")
    fig.suptitle("Rényi α-attention: entropy + saturation diagnostic (B2/EXP-12)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("mu_precond")
def plot_mu_precond(
    cells,                               # list of {precond in {fisher,raw}, n_e_steps, ppl}

    *,
    path: Optional[str] = None,
):
    r"""B3/EXP-14: validation PPL vs n_e_steps, Fisher natural-gradient vs raw-Euclidean mean step.

    The mu-arm ablation: nat_mu = Sigma*grad_mu (Fisher) vs the raw grad_mu, sigma retraction held
    fixed. One line per preconditioner; a gap that grows with n_e_steps means the mean-sector Fisher
    metric is load-bearing for the converged belief."""
    cells = list(cells)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for j, pre in enumerate(("fisher", "raw")):
        pts = sorted([(float(c["n_e_steps"]), float(c["ppl"])) for c in cells
                      if str(c["precond"]) == pre], key=lambda t: t[0])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=_CB[j], lw=1.8, label=f"{pre} mean step")
    ax.set(xlabel="E-step iterations (n_e_steps)", ylabel="validation PPL")
    ax.set_title("Fisher nat-grad vs raw Euclidean E-step mean")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("holonomy_trainability")
def plot_holonomy_trainability(
    arms,                                # list of {label, step, connection_norm, holonomy}

    *,
    path: Optional[str] = None,
):
    r"""A4/EXP-15: belief-transport holonomy vs the learned gauge-connection norm over training.

    The trainability curve for the Regime-II connection: as CE training drives ||connection_W|| up
    from 0, does the holonomy ||H-I||_F track it? Each point is one eval, colored by training step;
    a monotone climb is the open empirical contribution (the telescoping/Route-asymmetry half is
    test-pinned)."""
    arms = [a for a in arms if len(a.get("connection_norm", [])) >= 1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sc = None
    for j, a in enumerate(arms):
        x = np.asarray(a["connection_norm"], float)
        y = np.asarray(a["holonomy"], float)
        s = np.asarray(a.get("step", np.arange(x.size)), float)
        sc = ax.scatter(x, y, c=s, cmap="viridis", s=40, edgecolor="k", linewidth=0.3,
                        label=str(a.get("label", j)))
    if sc is not None:
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="training step")
    ax.set(xlabel=r"gauge connection norm $\|connection_W\|$",
           ylabel=r"holonomy deviation $\|H-I\|_F$")
    ax.set_title("Regime-II connection trainability: holonomy vs ‖connection‖")
    if arms:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)
