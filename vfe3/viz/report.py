r"""Single-run publication-figure DRIVER for VFE_3.0.

The figure generators in :mod:`vfe3.viz.figures` and the model-replay runners in
:mod:`vfe3.viz.extract` are pure libraries: nothing outside the tests drove them, so the
publication figure set was never produced from a real run. This module is the missing glue.
``generate_figures(run_dir)`` rebuilds the trained model from a run directory
(``config.json`` + ``best_model.pt``), runs the extract runners and the
:mod:`vfe3.metrics` measurements they feed, and writes the single-run figure set into
``run_dir/figures/``.

This is OFF the training hot path but IS auto-run at the end of training: ``finalize_run`` calls
it on the reloaded best-val model unless ``cfg.generate_figures=False`` (see ``run_artifacts.py``),
and ``make_figures.py`` re-runs the same set on demand for an already-trained run. The runners are
expensive (UMAP embedding, E-step replay, holonomy sampling, a belief bank over many sequences),
so each figure is best-effort -- a plotting / dependency / shape error is logged and skipped so one
bad figure never aborts the rest -- mirroring ``RunArtifacts._save_figures``.

The SWEEP-level figures (capacity_scaling, estep_capacity, pareto_frontier, ablation_forest,
lr_grid_heatmap) need multi-run data and belong to the ablation runner, not a single run; the
two-arm ln3_symmetry_breaking figure needs a frozen-gauge AND a learned-gauge run (an
experiment to set up). Neither is produced here.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Mapping, Optional

import torch

from vfe3 import metrics
from vfe3.config import VFE3Config, config_from_serialized
from vfe3.run_artifacts import semantic_config_fingerprint
from vfe3.viz import extract
from vfe3.viz import figures as figs


def _load_config(run_dir: Path) -> 'tuple[VFE3Config, str]':
    r"""Rebuild ``(cfg, dataset)`` from ``run_dir/config.json`` (the RunArtifacts metadata)."""
    data = json.loads((run_dir / "config.json").read_text())
    if not isinstance(data, Mapping) or not isinstance(data.get("config"), Mapping):
        raise ValueError(f"run metadata {run_dir / 'config.json'} has no config mapping")
    cfg = config_from_serialized(data["config"], source=str(run_dir / "config.json"))
    return cfg, data.get("dataset", "")


def _load_best_model_state(
    path: Path,
    cfg:  VFE3Config,

    *,
    map_location: object,
) -> Mapping[str, torch.Tensor]:
    """Validate and unwrap a self-bound ``best_model.pt`` for strict model loading."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    required = {"model_state", "config", "config_fingerprint"}
    if not isinstance(payload, Mapping) or not payload or not required.issubset(payload):
        raise ValueError(f"best checkpoint {path} is not a self-bound model/config bundle")
    embedded = payload["config"]
    if not isinstance(embedded, Mapping) or not embedded:
        raise ValueError(f"best checkpoint {path} has no embedded config mapping")
    raw_fingerprint = semantic_config_fingerprint(embedded)
    if payload["config_fingerprint"] != raw_fingerprint:
        raise ValueError(f"best checkpoint {path} has a config fingerprint mismatch")
    embedded_cfg = config_from_serialized(embedded, source=f"{path} embedded config")
    if asdict(embedded_cfg) != asdict(cfg):
        raise ValueError(f"best checkpoint {path} has a semantic config mismatch with config.json")
    model_state = payload["model_state"]
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError(f"best checkpoint {path} must contain a nonempty model_state mapping")
    return model_state


def _build_loader(dataset: str, cfg: VFE3Config, split: str):
    r"""A stable (unshuffled) loader for ``dataset``/``split``. Raises ``FileNotFoundError`` if the
    cache is absent: the figure driver never substitutes synthetic data for a real corpus (that
    would silently drive the publication figures off a toy stream). Pass ``loader=`` to
    :func:`generate_figures` to drive a custom stream instead."""
    from vfe3.data.datasets import make_dataloader
    return make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size,
                           shuffle=False, drop_last=False, vocab_size=cfg.vocab_size)


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
    split:         str                         = "validation",
    max_sequences: int                         = 64,
    allow_large:   bool                        = False,
    model:         Optional[torch.nn.Module]   = None,   # skip the reload; drive this live model
    loader:        Optional[object]            = None,   # skip the default loader build
    device:        Optional[torch.device]      = None,
    n_e_steps:     Optional[int]               = None,
    logger:        Optional[logging.Logger]    = None,
) -> List[Path]:
    r"""Drive the model and write the single-run publication figures into ``run_dir/figures/``.

    With ``model=None`` the trained model is rebuilt from ``run_dir/config.json`` and
    ``run_dir/best_model.pt``; pass ``model`` to drive a live in-memory model instead (the test
    path, and any post-train call that still holds the weights). ``loader`` defaults to a stable
    unshuffled loader for the run's dataset (raises if the cache is absent; pass ``loader`` to override).
    ``max_sequences`` caps the belief bank that feeds the UMAP triptych. ``allow_large`` opts into
    the two full-vocabulary extractors when their estimated logits-plus-probabilities peak exceeds
    8 GB; lighter inputs and figures still run when they are skipped. ``n_e_steps`` overrides the
    E-step trace length (default: the trained ``cfg.n_e_steps``). Returns the figure paths actually
    written (best-effort: a failed figure is logged and omitted).
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
        model.load_state_dict(
            _load_best_model_state(best, cfg, map_location=device or "cpu"),
            strict=True,
        )
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

    full_vocab_gb = 8.0 * int(cfg.vocab_size) * int(cfg.max_seq_len) * int(cfg.batch_size) / 1e9
    skip_full_vocab = full_vocab_gb > 8.0 and not allow_large
    if skip_full_vocab:
        logger.warning(
            "full-vocab figure inputs skipped: estimated logits+probabilities peak %.1f GB exceeds "
            "the 8 GB guard; pass allow_large=True to override",
            full_vocab_gb,
        )

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
    ce_bank     = (None if skip_full_vocab else
                   _safe(lambda: extract.belief_ce_bank(model, loader, device=device, max_batches=n_batches),
                         "belief_ce_bank"))   # B1/EXP-3 Sigma_q<->CE join (calibration figures)
    amaps       = _safe(lambda: model.attention_maps(tok), "attention_maps")
    per_layer   = _safe(lambda: model.diagnostics_per_layer(tok), "diagnostics_per_layer")
    health      = _safe(lambda: extract.numerical_health(model, tok), "numerical_health")
    s_channel   = _safe(lambda: extract.s_channel_refinement(model, tok), "s_channel_refinement")
    mc_belief   = _safe(lambda: extract.model_channel_belief(model, tok), "model_channel_belief")
    r_centroid  = _safe(lambda: extract.hyper_prior_centroid(model, tok), "hyper_prior_centroid")
    h_coupling  = _safe(lambda: extract.hyper_prior_coupling(model, tok), "hyper_prior_coupling")
    gamma_attn  = _safe(lambda: extract.gamma_attention(model, tok), "gamma_attention")
    mc_bank     = _safe(lambda: extract.model_channel_bank(model, token_batches, max_sequences=max_sequences),
                        "model_channel_bank")
    vstats      = (None if skip_full_vocab else
                   _safe(lambda: extract.vocab_prediction_stats(model, token_batches),
                         "vocab_prediction_stats"))
    readout     = _safe(lambda: extract.decode_readout(model), "decode_readout")
    run_label   = f"K{cfg.embed_dim}"

    # gpt2/cl100k decoder for the belief-UMAP linguistic-category coloring + token labels (None when
    # tiktoken is absent or the dataset has no real tokenizer -> the UMAP grays out and labels by id).
    decode = None
    try:
        from vfe3.data.datasets import get_tiktoken_decoder
        decode = get_tiktoken_decoder(dataset)
    except Exception as exc:
        logger.warning("token decoder unavailable (%s); belief UMAP will gray out", exc)

    # Decode-calibration reliability bins (conf/acc/frac) the run already wrote to research.json
    # (run_artifacts._calibration_and_strata); the B1/EXP-3 reliability diagram only plots them.
    reliability = None
    rj = run_dir / "research.json"
    if rj.exists():
        try:
            reliability = json.loads(rj.read_text()).get("reliability")
        except Exception as exc:                                  # best-effort, like every figure input
            logger.warning("research.json reliability unreadable (%s); reliability figure skipped", exc)

    written: List[Path] = []

    def _emit(name: str, thunk: Callable[[str], object], available: bool) -> None:
        r"""Write ``figures/<name>.png`` from ``thunk(path)``; skip when an input is missing."""
        if not available:
            logger.info("figure %r skipped (input unavailable)", name)
            return
        _before = set(figs.plt.get_fignums())            # registry snapshot for the leak sweep below
        try:
            path = figdir / f"{name}.png"
            fig = thunk(str(path))
            figs.plt.close(fig)
            written.append(path)
            logger.info("figure -> %s", path)
        except Exception as exc:
            # The thunk can register a pyplot figure and then raise (tight_layout/savefig) before
            # _emit ever receives it; close what it registered (audit 2026-07-01 round-3).
            for num in set(figs.plt.get_fignums()) - _before:
                figs.plt.close(num)
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
    _emit("per_layer_diagnostics",                               # the depth axis the aggregates collapse
          lambda p: figs.plot_per_layer_diagnostics(per_layer, path=p),
          per_layer is not None)
    _emit("gauge_equivariance",
          lambda p: figs.plot_gauge_equivariance(metrics.gauge_equivariance_residual(
              cstate["mu"], cstate["sigma"], cstate["omega"], model.group,
              kappa=cfg.kappa_beta, renyi_order=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
              divergence_family=cfg.divergence_family), path=p),
          cstate is not None)
    _emit("gauge_head_specialization",
          lambda p: figs.plot_gauge_head_specialization(
              metrics.per_head_gauge_invariants(cstate["exp_phi"], model.group.irrep_dims),
              # cstate["exp_phi"] is the FINAL block's gauge frame, so pair it with the FINAL layer's
              # per-head entropy (amaps[-1]) -- NOT the all-layer mean, which mismatched the depths.
              head_entropy=(metrics.attention_entropy_rows(amaps[-1]).mean(dim=-1) if amaps is not None else None),
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
    # s-channel sibling of the belief beta maps, in a distinct color family.
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
        _emit(f"model_umap_{ch}",
              lambda p, ch=ch: figs.plot_belief_umap(mc_bank, ch, kind="Model", decode=decode, path=p),
              mc_bank is not None)
    # Next-token vocabulary-probability figures (single-arm here; the cross-run K70-vs-K120 contrast
    # is the two-arm vocab_comparison_figures driver). vocab_confusion needs the token decoder for its
    # category bucketing; decode_readout is None (skipped) on the use_prior_bank=True KL-decode path.
    _emit("vocab_probability_heatmap",
          lambda p: figs.plot_vocab_probability_heatmap([{**vstats, "label": run_label}], decode=decode, path=p),
          vstats is not None)
    _emit("vocab_calibration",
          lambda p: figs.plot_vocab_calibration([{**vstats, "label": run_label}], decode=decode, path=p),
          vstats is not None)
    _emit("vocab_confusion",
          lambda p: figs.plot_vocab_confusion([{**vstats, "label": run_label}], decode=decode, path=p),
          vstats is not None and decode is not None)
    _emit("decode_readout",
          lambda p: figs.plot_decode_readout([{**readout, "label": run_label}], decode=decode, path=p),
          readout is not None)

    # B1/EXP-3 -- Sigma_q-as-calibrated-uncertainty: the decode-reliability control + the two
    # Sigma_q<->CE figures (stratified error curve, rho scatter) off the belief_ce_bank join.
    _emit("reliability_diagram",
          lambda p: figs.plot_reliability_diagram(reliability, path=p),
          reliability is not None)
    _emit("sigma_stratified_error",
          lambda p: figs.plot_sigma_stratified_error(ce_bank, path=p),
          ce_bank is not None)
    _emit("sigma_ce_scatter",
          lambda p: figs.plot_sigma_ce_scatter(ce_bank, path=p),
          ce_bank is not None)

    # Per-layer metrics CSV (rows = inference depth): the metrics.csv is final-block-only, so this is
    # the only place the per-layer belief-channel free energy / holonomy / gauge / belief geometry is
    # written to disk. Best-effort, like the figures.
    if per_layer is not None:
        try:
            import csv
            keys = list(per_layer)
            n_layers = len(per_layer[keys[0]]) if keys else 0
            csv_path = run_dir / "metrics_per_layer.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["layer", *keys])
                for li in range(n_layers):
                    w.writerow([li, *(float(per_layer[k][li]) for k in keys)])
            logger.info("wrote per-layer metrics -> %s", csv_path)
        except Exception as exc:
            logger.warning("metrics_per_layer.csv failed (%s); continuing", exc)

    logger.info("wrote %d single-run figures to %s", len(written), figdir)
    return written


def vocab_comparison_figures(
    run_dirs:  'list[str | Path]',
    out_dir:   'str | Path',

    *,
    labels:        Optional[List[str]]      = None,
    device:        Optional[torch.device]   = None,
    split:         str                      = "validation",
    max_sequences: int                      = 256,
    taxonomy:      str                      = "function_content",
    logger:        Optional[logging.Logger] = None,
) -> List[Path]:
    r"""Cross-run vocabulary-probability comparison (e.g. K70 vs K120) into ``out_dir``.

    Loads each trained run, runs the single ``vocab_prediction_stats`` pass, and writes the four
    arm-list figures (probability heatmap / calibration / confusion / decode readout) with one
    column per run -- the side-by-side collapse contrast the single-run :func:`generate_figures`
    pipeline cannot make (it sees one model). Every arm must use the same tokenizer tag; mixed
    tokenizers are rejected before any model load or plotting because one shared decoder cannot
    label them honestly. Each figure is best-effort: a failure is logged and skipped. ``labels``
    default to ``K<embed_dim>``; the decoder is taken from the first run's dataset after that identity
    check (the confusion figure is skipped when no tokenizer is available).
    """
    from vfe3.data.datasets import _tokenizer_tag
    from vfe3.model.model import VFEModel

    prepared = []
    for rd in run_dirs:
        run_dir = Path(rd)
        cfg, dataset = _load_config(run_dir)
        prepared.append((run_dir, cfg, dataset, _tokenizer_tag(dataset)))
    tokenizer_tags = {arm[3] for arm in prepared}
    if len(tokenizer_tags) > 1:
        details = ", ".join(f"{run_dir}: {tag}" for run_dir, _, _, tag in prepared)
        raise ValueError(f"mixed tokenizer tags in vocabulary comparison: {details}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = logger or logging.getLogger(__name__)
    figs.set_publication_style()

    arms_pred: List[dict] = []
    arms_readout: List[dict] = []
    decode = None
    for i, (rd, cfg, dataset, _) in enumerate(prepared):
        model = VFEModel(cfg)
        best = rd / "best_model.pt"
        if not best.exists():
            raise FileNotFoundError(f"no best_model.pt in {rd}; train with RunArtifacts first")
        model.load_state_dict(
            _load_best_model_state(best, cfg, map_location=device or "cpu"),
            strict=True,
        )
        dev = device or next(model.parameters()).device
        model = model.to(dev)
        model.eval()
        loader = _build_loader(dataset, cfg, split)
        n_batches = max(2, -(-max_sequences // max(cfg.batch_size, 1)))
        token_batches = _collect_token_batches(loader, dev, n_batches)
        label = labels[i] if labels and i < len(labels) else f"K{cfg.embed_dim}"
        arms_pred.append({**extract.vocab_prediction_stats(model, token_batches), "label": label})
        ro = extract.decode_readout(model)
        if ro is not None:
            arms_readout.append({**ro, "label": label})
        if decode is None:
            try:
                from vfe3.data.datasets import get_tiktoken_decoder
                decode = get_tiktoken_decoder(dataset)
            except Exception as exc:
                logger.warning("token decoder unavailable (%s); confusion figure skipped", exc)

    written: List[Path] = []

    def _emit(name: str, thunk: Callable[[], object], available: bool) -> None:
        if not available:
            logger.info("comparison figure %r skipped (input unavailable)", name)
            return
        before = set(figs.plt.get_fignums())
        fig = None
        try:
            path = out / f"{name}.png"
            fig = thunk()
            fig.savefig(str(path))
            written.append(path)
            logger.info("figure -> %s", path)
        except Exception as exc:
            logger.warning("comparison figure %r failed (%s); continuing", name, exc)
        finally:
            if fig is not None:
                figs.plt.close(fig)
            for num in set(figs.plt.get_fignums()) - before:
                figs.plt.close(num)

    _emit("vocab_probability_heatmap_compare",
          lambda: figs.plot_vocab_probability_heatmap(arms_pred, decode=decode), bool(arms_pred))
    _emit("vocab_calibration_compare",
          lambda: figs.plot_vocab_calibration(arms_pred, decode=decode), bool(arms_pred))
    _emit("vocab_confusion_compare",
          lambda: figs.plot_vocab_confusion(arms_pred, decode=decode, taxonomy=taxonomy),
          bool(arms_pred) and decode is not None)
    _emit("decode_readout_compare",
          lambda: figs.plot_decode_readout(arms_readout, decode=decode), bool(arms_readout))

    logger.info("wrote %d comparison figures to %s", len(written), out)
    return written
