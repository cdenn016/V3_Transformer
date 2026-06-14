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
    plot_ablation_forest,
    plot_attention_graph,
    plot_attention_grid,
    plot_attention_heatmap,
    plot_attention_structure,
    plot_belief_category_separation,
    plot_belief_spectrum,
    plot_belief_trajectories,
    plot_belief_umap,
    plot_capacity_scaling,
    plot_covariance_ellipses,
    plot_embedding,
    plot_estep_capacity,
    plot_estep_convergence,
    plot_free_energy_codescent,
    plot_free_energy_decomposition,
    plot_free_energy_descent,
    plot_gauge_equivariance,
    plot_gauge_head_specialization,
    plot_holonomy_curvature,
    plot_ln3_symmetry_breaking,
    plot_lr_grid_heatmap,
    plot_numerical_trust,
    plot_pareto_frontier,
    plot_spd_ellipses,
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
    try:
        coords = umap_embed(X, n_neighbors=5, seed=0)
    except (ImportError, OSError) as exc:
        # umap-learn relies on numba/llvmlite native code; on some platforms (e.g. very new Python
        # where numba lags) that native layer raises OSError/access-violation. umap_embed itself is
        # correct (lazy import, clear error); skip where the native dependency is non-functional.
        pytest.skip(f"umap-learn native layer unavailable on this platform: {exc}")
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


def test_plot_trajectory_options_save(tmp_path):
    # Real step x-axis + log y + smoothing + annotations on a long heavy-tailed series.
    steps = np.arange(100, 100 * 401, 100)                         # 400 points, real "step" values
    vals = 300.0 * np.exp(-steps / 4e4) + np.random.default_rng(0).random(steps.size)
    p_log = tmp_path / "logy.png"; p_med = tmp_path / "median.png"
    fig = plot_trajectory(vals, steps, ylabel="ppl", logy=True, smooth=15, annotate="min",
                          annotate_final=True, path=str(p_log)); plt.close(fig)
    # median_line + 'max' annotate on a positive series (the holonomy treatment).
    fig = plot_trajectory(np.abs(vals) + 1e-4, steps, logy=True, median_line=True, annotate="max",
                          path=str(p_med)); plt.close(fig)
    assert _saved_nonempty(p_log) and _saved_nonempty(p_med)


def test_plot_attention_grid_saves_4d_and_degenerate(tmp_path):
    # (L, H, N, N) grid plus the L==1/H==1 degenerate inputs the squeeze=False guard must handle.
    maps = torch.softmax(torch.randn(2, 3, 5, 5), dim=-1)          # 2 layers x 3 heads
    p4 = tmp_path / "grid.png"; p3 = tmp_path / "g3.png"; p2 = tmp_path / "g2.png"
    fig = plot_attention_grid(maps, path=str(p4)); plt.close(fig)
    fig = plot_attention_grid(maps[0], path=str(p3)); plt.close(fig)   # (H, N, N) -> one layer
    fig = plot_attention_grid(maps[0, 0], path=str(p2)); plt.close(fig)  # (N, N) -> one panel
    assert _saved_nonempty(p4) and _saved_nonempty(p3) and _saved_nonempty(p2)


def test_figure_registry():
    assert callable(get_figure("attention_heatmap"))
    assert callable(get_figure("attention_grid"))
    with pytest.raises(KeyError):
        get_figure("not_a_figure")


# --- publication figures (smoke: build synthetic inputs, save a nonempty PNG) ---

def _T():
    return 5, 6, 4, 2          # T, N, K, H


def test_plot_free_energy_descent_saves(tmp_path):
    s = np.arange(10)
    hist = {"step": s, "self_coupling": np.linspace(20, 5, 10), "belief_coupling": np.linspace(40, 30, 10),
            "attention_entropy": np.linspace(10, 8, 10), "val_ce": np.linspace(6, 4, 10),
            "free_energy_total": np.linspace(120, 80, 10)}
    p = tmp_path / "f1.png"
    fig = plot_free_energy_descent(hist, lambda_beta=1.0, self_div=torch.rand(20), path=str(p))
    plt.close(fig)
    assert _saved_nonempty(p)


def _fe_hist(n=30):
    s = np.arange(1, n + 1) * 100
    return {"step": s, "self_coupling": np.full(n, 140.0), "belief_coupling": np.linspace(30, 20, n),
            "attention_entropy": np.linspace(34, 13, n), "val_ce": np.linspace(5.7, 4.27, n)}


def test_plot_free_energy_decomposition_saves(tmp_path):
    p = tmp_path / "decomp.png"
    fig = plot_free_energy_decomposition(_fe_hist(), lambda_beta=1.0, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_free_energy_codescent_saves(tmp_path):
    p = tmp_path / "codescent.png"
    # per-row learned lambda_beta vector is accepted as well as a scalar
    fig = plot_free_energy_codescent(_fe_hist(), lambda_beta=np.ones(30), path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_estep_convergence_saves(tmp_path):
    T, N, K, _ = _T()
    trace = {"mu": torch.randn(T, N, K).cumsum(0), "sigma": torch.rand(T, N, K) + 0.3,
             "phi": torch.randn(T, N, 2), "free_energy": torch.linspace(50, 30, T)}
    p = tmp_path / "f2.png"
    fig = plot_estep_convergence(trace, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_ln3_symmetry_breaking_saves(tmp_path):
    _, N, K, H = _T()
    def arm():
        return {"step": np.arange(8), "val_ce": np.linspace(1.5, 1.1, 8),
                "omega": torch.randn(N, N, K, K), "beta": torch.softmax(torch.randn(H, N, N), dim=-1)}
    p = tmp_path / "f3.png"
    fig = plot_ln3_symmetry_breaking(arm(), arm(), period=3, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_belief_trajectories_saves(tmp_path):
    T, N, K, _ = _T()
    trace = {"mu": torch.randn(T, N, K).cumsum(0), "sigma": torch.rand(T, N, K) + 0.3, "phi": torch.randn(T, N, 2)}
    layer = {"d_ai": torch.tensor([0.0, 0.5, 0.9]), "effective_rank": torch.tensor([3.0, 2.5, 2.0])}
    p = tmp_path / "f4.png"
    fig = plot_belief_trajectories(trace, layer, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def _category_bank(M=60, K=4):
    bank = {"mu": torch.randn(M, K), "sigma": torch.rand(M, K) + 0.3, "phi": torch.randn(M, 2),
            "token_ids": torch.randint(0, 5, (M,)), "seq_idx": torch.zeros(M)}
    fake = {0: " the", 1: ",", 2: " cat", 3: "ing", 4: " 42"}    # function/punct/content/subword/number
    return bank, (lambda ids: fake.get(int(ids[0]), " x"))


def test_plot_belief_umap_per_channel_categories(tmp_path):
    bank, decode = _category_bank()
    p = tmp_path / "f5.png"
    try:
        fig = plot_belief_umap(bank, "mu", decode=decode, seed=0, path=str(p)); plt.close(fig)
    except (ImportError, OSError) as exc:
        pytest.skip(f"umap-learn native layer unavailable: {exc}")
    assert _saved_nonempty(p)


def test_plot_belief_umap_fallback_no_decode(tmp_path):
    bank, _ = _category_bank(M=40)
    p = tmp_path / "f5b.png"
    try:
        fig = plot_belief_umap(bank, "phi", decode=None, seed=0, path=str(p)); plt.close(fig)
    except (ImportError, OSError) as exc:
        pytest.skip(f"umap-learn native layer unavailable: {exc}")
    assert _saved_nonempty(p)


def test_plot_belief_category_separation_saves(tmp_path):
    bank, decode = _category_bank()
    p = tmp_path / "f5c.png"
    fig = plot_belief_category_separation(bank, decode=decode, sil_sample=50, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_token_category_helpers():
    from vfe3.viz.figures import _bpe_category, _funccontent_category
    assert _bpe_category(",") == 0          # punctuation
    assert _bpe_category(" 2014") == 1      # number (leading space, all-digit core)
    assert _bpe_category(" cat") == 2       # word-start lowercase
    assert _bpe_category(" Cat") == 3       # word-start Capitalized
    assert _bpe_category("ing") == 4        # continuation subword
    assert _bpe_category(" ") == 5          # whitespace/other
    assert _funccontent_category(" the") == 2   # function word
    assert _funccontent_category(" cat") == 3   # content word
    assert _funccontent_category(".") == 0      # punctuation
    assert _funccontent_category("42") == 1     # number


def test_cluster_embedding_and_lift_labels():
    # the belief-UMAP redesign clusters the embedding and labels each cluster by DISTINCTIVE (lift)
    # tokens, not raw frequency: a global stopword present in every cluster must not dominate the labels.
    import numpy as np
    from vfe3.viz.figures import _cluster_embedding, _cluster_lift_labels
    rng = np.random.default_rng(0)
    c0 = rng.normal([0, 0], 0.2, (80, 2)); c1 = rng.normal([10, 10], 0.2, (80, 2))
    coords = np.vstack([c0, c1])
    # cluster 0 = mostly token 5 ('apple') + stopword 0 ('the'); cluster 1 = mostly token 9 ('zebra') + 'the'
    tids = np.concatenate([np.array([5] * 60 + [0] * 20), np.array([9] * 60 + [0] * 20)])
    labels = _cluster_embedding(coords, seed=0)
    assert set(labels.tolist()) - {-1}                          # at least one real cluster found
    names = {5: "apple", 9: "zebra", 0: "the"}
    # top-1 by lift: the distinctive token (apple lift 2.0, zebra lift 2.0) outranks the stopword
    # 'the' (present in both clusters -> lift ~1.0), so the per-cluster top label is never the stopword.
    lab = _cluster_lift_labels(tids, labels, decode=lambda l: names[int(l[0])], k=1)
    joined = " ".join(lab.values())
    assert "apple" in joined and "zebra" in joined             # distinctive tokens surface
    assert "the" not in joined                                  # the in-every-cluster stopword does not


def test_plot_gauge_equivariance_saves(tmp_path):
    resid = {"energy_in_group": torch.rand(50) * 1e-6, "energy_out_group": torch.rand(50) + 0.5,
             "beta_in_group": torch.rand(50) * 1e-6, "beta_out_group": torch.rand(50) * 0.3}
    p = tmp_path / "f6.png"
    fig = plot_gauge_equivariance(resid, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_gauge_head_specialization_saves(tmp_path):
    per_head = {"logdet": torch.randn(40, 2), "anisotropy": torch.rand(40, 2) + 1.0}
    p1 = tmp_path / "f7a.png"; p2 = tmp_path / "f7b.png"
    fig = plot_gauge_head_specialization(per_head, path=str(p1)); plt.close(fig)
    fig = plot_gauge_head_specialization(per_head, head_entropy=torch.rand(2), path=str(p2)); plt.close(fig)
    assert _saved_nonempty(p1) and _saved_nonempty(p2)


def test_plot_attention_structure_saves(tmp_path):
    _, N, _, H = _T()
    beta = torch.softmax(torch.randn(2, H, N, N), dim=-1)         # (L, H, N, N)
    p = tmp_path / "f8.png"
    fig = plot_attention_structure(beta, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_belief_spectrum_and_ellipses_save(tmp_path):
    _, N, K, _ = _T()
    sigma = torch.rand(N, K) + 0.3
    p1 = tmp_path / "f9a.png"; p2 = tmp_path / "f9b.png"
    fig = plot_belief_spectrum(sigma, eps=1e-6, sigma_max=5.0, path=str(p1)); plt.close(fig)
    full = torch.diag_embed(sigma)
    fig = plot_spd_ellipses(torch.randn(N, K), full, dims=(0, 1), path=str(p2)); plt.close(fig)
    assert _saved_nonempty(p1) and _saved_nonempty(p2)


def test_plot_holonomy_curvature_saves(tmp_path):
    flat = {"per_triple": torch.rand(80) * 1e-6, "span": torch.randint(1, 6, (80,)).float()}
    regime = {"per_triple": torch.rand(80) + 0.1, "span": torch.randint(1, 6, (80,)).float()}
    p = tmp_path / "f10.png"
    fig = plot_holonomy_curvature(flat, regime, curvature=torch.rand(6, 6), path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_capacity_scaling_and_estep_capacity_save(tmp_path):
    scaling = {"embed_dim": {"x": np.array([20, 40, 64]), "bpc": np.array([7.5, 7.0, 6.8]),
                             "lo": np.array([7.4, 6.9, 6.7]), "hi": np.array([7.6, 7.1, 6.9])},
               "n_layers": {"x": np.array([1, 2, 3]), "bpc": np.array([7.5, 7.1, 7.0])}}
    p1 = tmp_path / "f11a.png"; p2 = tmp_path / "f11b.png"
    fig = plot_capacity_scaling(scaling, path=str(p1)); plt.close(fig)
    fig = plot_estep_capacity(np.array([1, 2, 4]), np.array([7.5, 7.1, 6.9]), np.array([100., 90., 85.]),
                              n_params=12345, wall_time=np.array([10., 18., 33.]), path=str(p2)); plt.close(fig)
    assert _saved_nonempty(p1) and _saved_nonempty(p2)


def test_plot_pareto_frontier_saves(tmp_path):
    points = {"bpc": np.array([7.5, 7.0, 6.8, 7.2]), "n_params": np.array([1e4, 4e4, 9e4, 2e4]),
              "wall_time": np.array([10., 30., 60., 18.])}
    p = tmp_path / "f11c.png"
    fig = plot_pareto_frontier(points, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_ablation_forest_and_lr_grid_save(tmp_path):
    rows = [{"label": "frozen gauge", "delta": 0.4, "lo": 0.3, "hi": 0.5},
            {"label": "surrogate F", "delta": 0.1, "lo": 0.05, "hi": 0.15},
            {"label": "uniform prior", "delta": 0.25, "lo": 0.2, "hi": 0.3}]
    grid = {"x": np.array([0.01, 0.02, 0.03]), "y": np.array([0.001, 0.002]),
            "z": np.array([[190., 185., 188.], [186., 180., 184.]]),
            "xlabel": "m_mu_lr", "ylabel": "m_sigma_lr", "baseline": (0.02, 0.002)}
    p1 = tmp_path / "f12a.png"; p2 = tmp_path / "f12b.png"
    fig = plot_ablation_forest(rows, path=str(p1)); plt.close(fig)
    fig = plot_lr_grid_heatmap(grid, path=str(p2)); plt.close(fig)
    assert _saved_nonempty(p1) and _saved_nonempty(p2)


def test_plot_numerical_trust_saves(tmp_path):
    guard = {"sigma_floor_frac": 0.0, "sigma_ceil_frac": 0.0, "energy_klmax_frac": 0.0, "selfdiv_klmax_frac": 0.0}
    health = {"nan_mu": 0.0, "nan_sigma": 0.0, "nan_phi": 0.0, "nan_energy": 0.0, "nan_beta": 0.0, "max_condition": 12.0}
    causal = {"future_leakage": torch.zeros(2), "row_sum_error": torch.zeros(2), "active_set_slope": torch.ones(2)}
    p = tmp_path / "f13.png"
    fig = plot_numerical_trust(guard, health, causal, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)
