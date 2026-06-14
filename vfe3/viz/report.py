r"""Single-run publication-figure DRIVER for VFE_3.0.

The figure generators in :mod:`vfe3.viz.figures` and the model-replay runners in
:mod:`vfe3.viz.extract` are pure libraries: nothing outside the tests drove them, so the
publication figure set was never produced from a real run. This module is the missing glue.
``generate_figures(run_dir)`` rebuilds the trained model from a run directory
(``config.json`` + ``best_model.pt``), runs the extract runners and the
:mod:`vfe3.metrics` measurements they feed, and writes the single-run figure set into
``run_dir/figures/``.

This is OPT-IN and OFF the training hot path: it is a separate click-to-run step
(``make_figures.py``), never invoked by ``train``/``finalize_run``, because the runners are
expensive (UMAP embedding, E-step replay, holonomy sampling, a belief bank over many
sequences). Each figure is best-effort -- a plotting / dependency / shape error is logged and
skipped so one bad figure never aborts the rest -- mirroring ``RunArtifacts._save_figures``.

The SWEEP-level figures (capacity_scaling, estep_capacity, pareto_frontier, ablation_forest,
lr_grid_heatmap) need multi-run data and belong to the ablation runner, not a single run; the
two-arm ln3_symmetry_breaking figure needs a frozen-gauge AND a learned-gauge run (an
experiment to set up). Neither is produced here.
"""

import json
import logging
from pathlib import Path
from typing import Callable, List, Optional

import torch

from vfe3 import metrics
from vfe3.config import VFE3Config
from vfe3.viz import extract
from vfe3.viz import figures as figs


def _load_config(run_dir: Path) -> 'tuple[VFE3Config, str]':
    r"""Rebuild ``(cfg, dataset)`` from ``run_dir/config.json`` (the RunArtifacts metadata)."""
    data = json.loads((run_dir / "config.json").read_text())
    return VFE3Config(**data["config"]), data.get("dataset", "")


def _build_loader(dataset: str, cfg: VFE3Config, split: str):
    r"""A stable (unshuffled) loader for ``dataset``/``split``. Raises ``FileNotFoundError`` if the
    cache is absent: the figure driver never substitutes synthetic data for a real corpus (that
    would silently drive the publication figures off a toy stream). Pass ``loader=`` to
    :func:`generate_figures` to drive a custom stream instead."""
    from vfe3.data.datasets import make_dataloader
    return make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size,
                           shuffle=False, drop_last=False)


def _collect_token_batches(
    loader,
    device:   torch.device,
    n_batches: int,
) -> List[torch.Tensor]:
    r"""Up to ``n_batches`` (B, N) token-id batches off the loader (drops the target tensor)."""
    out: List[torch.Tensor] = []
    for batch in loader:
        tokens = batch[0] if isinstance(batch, (tuple, list)) else batch
        out.append(tokens.to(device))
        if len(out) >= n_batches:
            break
    return out


def generate_figures(
    run_dir:       'str | Path',

    *,
    model:         Optional[torch.nn.Module]   = None,   # skip the reload; drive this live model
    loader:        Optional[object]            = None,   # skip the default loader build
    device:        Optional[torch.device]      = None,
    split:         str                         = "validation",
    max_sequences: int                         = 64,
    n_e_steps:     Optional[int]               = None,
    logger:        Optional[logging.Logger]    = None,
) -> List[Path]:
    r"""Drive the model and write the single-run publication figures into ``run_dir/figures/``.

    With ``model=None`` the trained model is rebuilt from ``run_dir/config.json`` and
    ``run_dir/best_model.pt``; pass ``model`` to drive a live in-memory model instead (the test
    path, and any post-train call that still holds the weights). ``loader`` defaults to a stable
    unshuffled loader for the run's dataset (raises if the cache is absent; pass ``loader`` to override).
    ``max_sequences`` caps the belief bank that feeds the UMAP triptych; ``n_e_steps`` overrides
    the E-step trace length (default: the trained ``cfg.n_e_steps``). Returns the figure paths
    actually written (best-effort: a failed figure is logged and omitted).
    """
    run_dir = Path(run_dir)
    figdir = run_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    logger = logger or logging.getLogger(__name__)

    if model is None:
        from vfe3.model.model import VFEModel
        cfg, dataset = _load_config(run_dir)
        model = VFEModel(cfg)
        best = run_dir / "best_model.pt"
        if not best.exists():
            raise FileNotFoundError(f"no best_model.pt in {run_dir}; train with RunArtifacts first")
        model.load_state_dict(torch.load(best, map_location=device or "cpu", weights_only=True))
    else:
        cfg = model.cfg
        dataset = ""
        cfgj = run_dir / "config.json"
        if cfgj.exists():
            dataset = json.loads(cfgj.read_text()).get("dataset", "")

    device = device or next(model.parameters()).device
    model = model.to(device)
    model.eval()
    figs.set_publication_style()

    if loader is None:
        loader = _build_loader(dataset, cfg, split)
    n_batches = max(2, -(-max_sequences // max(cfg.batch_size, 1)))    # ceil-div, >= 2 for the bank
    token_batches = _collect_token_batches(loader, device, n_batches)
    if not token_batches:
        raise RuntimeError(f"loader for {dataset!r}/{split!r} yielded no batches")
    tok = token_batches[0][:1]                                         # one sequence for the single-seq figures

    def _safe(fn: Callable, label: str):
        r"""Run a model-replay extractor, logging+swallowing a failure so the rest proceed."""
        try:
            return fn()
        except Exception as exc:
            logger.warning("input %r failed (%s); dependent figures skipped", label, exc)
            return None

    # ---- expensive model-replay inputs, each guarded (a failure skips only its figures) ----
    trace       = _safe(lambda: extract.e_step_belief_trace(model, tok, n_iter=n_e_steps), "e_step_belief_trace")
    layer_trace = _safe(lambda: extract.across_layer_belief_trace(model, tok), "across_layer_belief_trace")
    bank        = _safe(lambda: extract.belief_bank(model, token_batches, max_sequences=max_sequences), "belief_bank")
    cstate      = _safe(lambda: extract.converged_state(model, tok), "converged_state")
    amaps       = _safe(lambda: model.attention_maps(tok), "attention_maps")
    health      = _safe(lambda: extract.numerical_health(model, tok), "numerical_health")
    s_channel   = _safe(lambda: extract.s_channel_refinement(model, tok), "s_channel_refinement")
    mc_belief   = _safe(lambda: extract.model_channel_belief(model, tok), "model_channel_belief")
    r_centroid  = _safe(lambda: extract.hyper_prior_centroid(model, tok), "hyper_prior_centroid")
    h_coupling  = _safe(lambda: extract.hyper_prior_coupling(model, tok), "hyper_prior_coupling")
    gamma_attn  = _safe(lambda: extract.gamma_attention(model, tok), "gamma_attention")
    mc_bank     = _safe(lambda: extract.model_channel_bank(model, token_batches, max_sequences=max_sequences),
                        "model_channel_bank")

    # gpt2/cl100k decoder for the belief-UMAP linguistic-category colouring + token labels (None when
    # tiktoken is absent or the dataset has no real tokenizer -> the UMAP greys out and labels by id).
    decode = None
    try:
        from vfe3.data.datasets import get_tiktoken_decoder
        decode = get_tiktoken_decoder(dataset)
    except Exception as exc:
        logger.warning("token decoder unavailable (%s); belief UMAP will grey out", exc)

    written: List[Path] = []

    def _emit(name: str, thunk: Callable[[str], object], available: bool) -> None:
        r"""Write ``figures/<name>.png`` from ``thunk(path)``; skip when an input is missing."""
        if not available:
            logger.info("figure %r skipped (input unavailable)", name)
            return
        try:
            path = figdir / f"{name}.png"
            fig = thunk(str(path))
            figs.plt.close(fig)
            written.append(path)
            logger.info("figure -> %s", path)
        except Exception as exc:
            logger.warning("figure %r failed (%s); continuing", name, exc)

    _emit("estep_convergence",
          lambda p: figs.plot_estep_convergence(trace, path=p),
          trace is not None)
    _emit("belief_trajectories",
          lambda p: figs.plot_belief_trajectories(trace, layer_trace, path=p),
          trace is not None)
    for ch in ("mu", "sigma", "phi"):                                 # one UMAP file per belief channel
        _emit(f"belief_umap_{ch}",
              lambda p, ch=ch: figs.plot_belief_umap(bank, ch, decode=decode, path=p),
              bank is not None)
    _emit("belief_category_separation",
          lambda p: figs.plot_belief_category_separation(bank, decode=decode, path=p),
          bank is not None)
    _emit("attention_structure",
          lambda p: figs.plot_attention_structure(amaps, path=p),
          amaps is not None)
    # Per-head belief-beta attention heatmaps (magma), one file per (layer, head) -- the per-head
    # detail behind attention_structure; matched by the model-channel gamma maps (viridis) below.
    if amaps is not None:
        bm = amaps if amaps.dim() == 4 else amaps[None]              # (L, H, N, N)
        for li in range(bm.shape[0]):
            for hi in range(bm.shape[1]):
                bname = (f"attention_beta_head{hi}" if bm.shape[0] == 1
                         else f"attention_beta_layer{li}_head{hi}")
                _emit(bname,
                      lambda p, li=li, hi=hi: figs.plot_attention_heatmap(
                          bm[li, hi], cmap="magma", symbol=r"\beta",
                          title=f"Belief attention - layer {li} head {hi}", path=p),
                      True)
    _emit("gauge_equivariance",
          lambda p: figs.plot_gauge_equivariance(metrics.gauge_equivariance_residual(
              cstate["mu"], cstate["sigma"], cstate["omega"], model.group,
              kappa=cfg.kappa_beta, renyi_order=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
              divergence_family=cfg.divergence_family), path=p),
          cstate is not None)
    _emit("gauge_head_specialization",
          lambda p: figs.plot_gauge_head_specialization(
              metrics.per_head_gauge_invariants(cstate["exp_phi"], model.group.irrep_dims),
              head_entropy=(metrics.attention_entropy_rows(amaps).mean(dim=(0, 2)) if amaps is not None else None),
              path=p),
          cstate is not None)
    _emit("belief_spectrum",
          lambda p: figs.plot_belief_spectrum(cstate["sigma"], eps=cfg.eps, sigma_max=cfg.sigma_max, path=p),
          cstate is not None)
    _emit("spd_ellipses",
          lambda p: figs.plot_spd_ellipses(cstate["mu"], cstate["sigma"], path=p),
          cstate is not None)
    _emit("holonomy_curvature",
          lambda p: figs.plot_holonomy_curvature(
              metrics.holonomy_deviation_sampled(cstate["omega"]),
              curvature=metrics.curvature_field(cstate["omega"]), path=p),   # Panel C: spatial curvature field
          cstate is not None)
    _emit("numerical_trust",
          lambda p: figs.plot_numerical_trust(
              metrics.guard_saturation(cstate["sigma"], cstate["energy"], cstate["self_div"],
                                       eps=cfg.eps, sigma_max=cfg.sigma_max, kl_max=cfg.kl_max),
              health if health is not None else {},
              causal=(metrics.causal_sanity(amaps) if amaps is not None else None), path=p),
          cstate is not None)
    _emit("s_channel_refinement",                                 # only when s_e_step=True (else None)
          lambda p: figs.plot_s_channel_refinement(s_channel, path=p),
          s_channel is not None)
    # Model-channel (s / r / h) and gamma_ij figures: present whenever the model channel is active
    # (s table exists); the r/h figures additionally require the centroid r (lambda_h>0 OR s_e_step),
    # so their extractors return None and _emit skips them when r is absent.
    _emit("model_channel_belief",                                 # the s figure
          lambda p: figs.plot_model_channel_belief(mc_belief, path=p),
          mc_belief is not None)
    _emit("hyper_prior_centroid",                                 # the r figure
          lambda p: figs.plot_hyper_prior_centroid(r_centroid, path=p),
          r_centroid is not None)
    _emit("hyper_prior_coupling",                                 # the h figure (lambda_h block)
          lambda p: figs.plot_hyper_prior_coupling(h_coupling, path=p),
          h_coupling is not None)
    # Per-head model-coupling (gamma) attention heatmaps (viridis), one file per head -- the
    # s-channel sibling of the belief beta maps, in a distinct colour family.
    if gamma_attn is not None:
        gm = gamma_attn["gamma"]                                  # (H, N, N)
        gm = gm if gm.dim() == 3 else gm[None]
        for hi in range(gm.shape[0]):
            _emit(f"attention_gamma_head{hi}",
                  lambda p, hi=hi: figs.plot_attention_heatmap(
                      gm[hi], cmap="viridis", symbol=r"\gamma",
                      title=f"Model-coupling attention - head {hi}", path=p),
                  True)
    for ch in ("mu", "sigma"):                                    # model-channel s UMAP (no phi: shares belief gauge)
        _emit(f"belief_umap_s_{ch}",
              lambda p, ch=ch: figs.plot_belief_umap(mc_bank, ch, decode=decode, path=p),
              mc_bank is not None)

    logger.info("wrote %d single-run figures to %s", len(written), figdir)
    return written
