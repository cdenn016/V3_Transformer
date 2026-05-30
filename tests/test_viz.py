import numpy as np
import pytest
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vfe3.viz.figures import (
    attention_graph,
    clustering_metrics,
    get_figure,
    plot_attention_graph,
    plot_attention_heatmap,
    plot_covariance_ellipses,
    plot_embedding,
    plot_trajectory,
    set_publication_style,
    umap_embed,
)


def _saved_nonempty(path):
    return path.exists() and path.stat().st_size > 0


def test_set_publication_style_runs():
    set_publication_style()
    assert plt.rcParams["savefig.dpi"] == 300


def test_clustering_metrics_separated_blobs():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(30, 4, generator=g)
    b = torch.randn(30, 4, generator=g) + 8.0                  # well separated
    X = torch.cat([a, b]); y = torch.cat([torch.zeros(30), torch.ones(30)])
    m = clustering_metrics(X, y)
    assert m["silhouette"] > 0.5                                # clean separation
    assert m["calinski_harabasz"] > 1.0


def test_umap_embed_shape():
    X = torch.randn(30, 8)
    coords = umap_embed(X, n_neighbors=5, seed=0)
    assert coords.shape == (30, 2)


def test_attention_graph_structure():
    beta = torch.softmax(torch.randn(5, 5), dim=-1)
    G = attention_graph(beta, threshold=0.0)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() <= 5 * 4                        # no self-loops


def test_plot_embedding_saves(tmp_path):
    coords = np.random.randn(20, 2)
    p = tmp_path / "emb.png"
    fig = plot_embedding(coords, labels=np.arange(20), path=str(p))
    plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_attention_graph_and_heatmap_save(tmp_path):
    beta = torch.softmax(torch.randn(6, 6), dim=-1)
    pg = tmp_path / "g.png"; ph = tmp_path / "h.png"
    fig1 = plot_attention_graph(beta, path=str(pg)); plt.close(fig1)
    fig2 = plot_attention_heatmap(beta, path=str(ph)); plt.close(fig2)
    assert _saved_nonempty(pg) and _saved_nonempty(ph)


def test_plot_covariance_ellipses_and_trajectory_save(tmp_path):
    mu = torch.randn(8, 3); sigma = torch.rand(8, 3) + 0.5
    pe = tmp_path / "ell.png"; pt = tmp_path / "traj.png"
    fig1 = plot_covariance_ellipses(mu, sigma, path=str(pe)); plt.close(fig1)
    fig2 = plot_trajectory([3.0, 2.1, 1.7, 1.5], ylabel="CE", path=str(pt)); plt.close(fig2)
    assert _saved_nonempty(pe) and _saved_nonempty(pt)


def test_figure_registry():
    assert callable(get_figure("attention_heatmap"))
    with pytest.raises(KeyError):
        get_figure("not_a_figure")
