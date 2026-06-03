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
) -> Dict[str, float]:
    """Unsupervised cluster quality of ``features`` under ``labels``: silhouette + CH index."""
    from sklearn.metrics import calinski_harabasz_score, silhouette_score
    X = _np(features); y = _np(labels)
    if len(set(y.tolist())) < 2:
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan")}
    return {
        "silhouette":        float(silhouette_score(X, y)),
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


def plot_attention_heatmap(
    beta,                                # (N, N)

    *,
    path: Optional[str] = None,
):
    """Heatmap of the attention matrix (rows = queries, cols = keys)."""
    B = _np(beta)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(B, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\beta_{ij}$")
    ax.set(title="Attention", xlabel="key j", ylabel="query i")
    return _save(fig, path)


def plot_attention_map(
    beta,                                # (N, N) attention weights for ONE layer/head

    *,
    floor: float          = -6.0,        # log10 floor: active cells below clip here; masked beta=0 -> NaN
    title: str            = "Attention",
    path:  Optional[str]  = None,
):
    r"""Single-panel ``log10`` heatmap of one attention map (rows = query i, cols = key j).

    ``beta`` is a softmax row distribution, so ``beta in (0, 1]`` and ``log10(beta) <= 0``; the
    colour range is pinned to ``[floor, 0]`` so every per-(layer, head) image shares one scale and
    a sharp head stays visually comparable to a diffuse one. Causally-masked keys (``beta == 0``,
    upper triangle under the causal prior) become NaN and render in the colormap's "bad" colour
    (grey); active cells below ``floor`` clip to ``floor``. The log scale is what makes the small
    off-diagonal structure legible -- on a linear ``[0, 1]`` scale only the diagonal shows.
    """
    B = _np(beta).astype(np.float64)
    with np.errstate(divide="ignore"):
        logB = np.log10(B)                           # (N, N); beta=0 -> -inf
    logB[~np.isfinite(logB)] = np.nan                # masked / zero -> NaN (grey via set_bad)
    logB = np.clip(logB, floor, 0.0)                 # np.clip preserves NaN; active floor at `floor`
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#d9d9d9")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(logB, cmap=cmap, aspect="auto", vmin=floor, vmax=0.0)
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\log_{10}\beta_{ij}$")
    ax.set(title=title, xlabel="key $j$", ylabel="query $i$")
    return _save(fig, path)


def plot_attention_grid(
    maps,                                # (L, H, N, N) per-layer per-head attention (or (H,N,N) / (N,N))

    *,
    title: str           = "Attention",
    path:  Optional[str] = None,
):
    """Grid of attention heatmaps: rows = layers, cols = heads (rows query i, cols key j).

    Accepts a per-layer/per-head stack ``(L, H, N, N)`` (as :meth:`VFEModel.attention_maps`
    returns), a single layer ``(H, N, N)``, or a single map ``(N, N)``. A shared colour scale
    across all panels makes heads/layers comparable; ``squeeze=False`` keeps the L==1 / H==1
    axes array 2-D so indexing is uniform.
    """
    M = _np(maps)
    if M.ndim == 2:                      # (N, N) -> one layer, one head
        M = M[None, None]
    elif M.ndim == 3:                    # (H, N, N) -> one layer
        M = M[None]
    L, H = M.shape[0], M.shape[1]
    fig, axes = plt.subplots(L, H, figsize=(2.6 * H + 1.0, 2.6 * L + 0.6), squeeze=False)
    vmax = float(M.max()) if M.size else 1.0
    im = None
    for li in range(L):
        for hi in range(H):
            ax = axes[li][hi]
            im = ax.imshow(M[li, hi], cmap="magma", aspect="auto", vmin=0.0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if li == 0:
                ax.set_title(f"head {hi}")
            if hi == 0:
                ax.set_ylabel(f"layer {li}\nquery $i$")
            if li == L - 1:
                ax.set_xlabel("key $j$")
    if im is not None:
        fig.colorbar(im, ax=list(axes.ravel()), shrink=0.85, label=r"$\beta_{ij}$")
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


def plot_trajectory(
    values,                              # (T,) a scalar series (loss / free energy)

    *,
    ylabel: str = "loss",
    title:  str = "Training trajectory",
    path:   Optional[str] = None,
):
    """Line plot of a scalar trajectory over steps."""
    v = _np(values).reshape(-1)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(np.arange(len(v)), v, marker="o", ms=3, lw=1.5, color="#C44E52")
    ax.set(title=title, xlabel="step", ylabel=ylabel)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Figure registry: name -> generator. New figures slot in by name.
# ---------------------------------------------------------------------------
_FIGURES: Dict[str, Callable] = {
    "embedding":           plot_embedding,
    "attention_graph":     plot_attention_graph,
    "attention_heatmap":   plot_attention_heatmap,
    "attention_map":       plot_attention_map,
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
