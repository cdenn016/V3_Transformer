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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence

import torch

from vfe3 import metrics
from vfe3.config import VFE3Config, config_from_serialized
from vfe3.run_artifacts import semantic_config_fingerprint
from vfe3.viz import extract
from vfe3.viz import embedding_comparison
from vfe3.viz import figures as figs
from vfe3.viz.text import supports_english_linguistic_taxonomies


CONTROLLED_BANK_TOKENS = 16_384

_SINGLE_RUN_FIGURE_INVENTORY = (
    "estep_convergence.png",
    "belief_trajectories.png",
    "belief_category_separation.png",
    "attention_structure.png",
    "per_layer_diagnostics.png",
    "gauge_equivariance.png",
    "gauge_head_specialization.png",
    "belief_spectrum.png",
    "spd_ellipses.png",
    "holonomy_curvature.png",
    "numerical_trust.png",
    "s_channel_refinement.png",
    "model_channel_belief.png",
    "hyper_prior_centroid.png",
    "hyper_prior_coupling.png",
    "belief_umap_mu.png",
    "belief_umap_sigma.png",
    "belief_umap_phi.png",
    "model_umap_mu.png",
    "model_umap_sigma.png",
    "model_umap_phi.png",
    "vocab_probability_heatmap.png",
    "vocab_calibration.png",
    "vocab_confusion.png",
    "decode_readout.png",
    "reliability_diagram.png",
    "sigma_stratified_error.png",
    "sigma_ce_scatter.png",
)

_ENGLISH_ONLY_SINGLE_RUN_FIGURES = frozenset({
    "belief_category_separation.png",
    "vocab_confusion.png",
})


def plan_single_run_figures(
    dataset:      str,
    availability: Mapping[str, bool],
) -> tuple[str, ...]:
    """Return available single-run figure filenames in stable publication order."""
    english_taxonomies = supports_english_linguistic_taxonomies(dataset)
    return tuple(
        filename
        for filename in _SINGLE_RUN_FIGURE_INVENTORY
        if availability.get(filename.removesuffix(".png"), False)
        and (english_taxonomies or filename not in _ENGLISH_ONLY_SINGLE_RUN_FIGURES)
    )


@dataclass(frozen=True)
class ArtifactPublicationResult:
    """Separately report the newly published comparison sidecar and figure."""

    json_path:   Optional[Path]
    figure_path: Optional[Path]
    outcomes:    Dict[str, Dict[str, object]]

    def __iter__(self) -> Iterator[Optional[Path]]:
        yield self.json_path
        yield self.figure_path


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
    r"""Collect CPU-hosted token batches; consumers move only their current batch to ``device``."""
    del device                                                        # retained private-call compatibility
    out: List[torch.Tensor] = []
    for batch in loader:
        tokens = batch[0] if isinstance(batch, (tuple, list)) else batch
        out.append(tokens.detach().cpu())
        if len(out) >= n_batches:
            break
    return out


def _collect_batches(loader, n_batches: int) -> List[object]:
    r"""Materialize the bounded report population once, retaining targets for shared extraction."""
    out: List[object] = []
    for batch in loader:
        out.append(batch)
        if len(out) >= n_batches:
            break
    return out


def _resolve_bank_budget(
    cfg: VFE3Config,

    *,
    max_tokens:    Optional[int],
    max_sequences: Optional[int],
) -> 'tuple[Optional[int], Optional[int], int]':
    """Resolve the exact bank cap and number of loader batches needed to satisfy it."""
    extract._validate_bank_caps(max_tokens=max_tokens, max_sequences=max_sequences)
    if max_tokens is None and max_sequences is None:
        max_tokens = CONTROLLED_BANK_TOKENS
    if max_tokens is not None:
        tokens_per_batch = max(int(cfg.batch_size) * int(cfg.max_seq_len), 1)
        n_batches = max(1, -(-max_tokens // tokens_per_batch))
    else:
        n_batches = max(1, -(-int(max_sequences) // max(int(cfg.batch_size), 1)))
    return max_tokens, max_sequences, n_batches


def _estimated_full_vocab_bank_bytes(cfg: VFE3Config, n_batches: int) -> int:
    r"""Conservatively budget retained logits plus probability workspaces for every bank batch."""
    if n_batches < 1:
        raise ValueError(f"n_batches must be positive, got {n_batches}")
    elements = (
        int(cfg.vocab_size)
        * int(cfg.max_seq_len)
        * int(cfg.batch_size)
        * int(n_batches)
    )
    return 2 * torch.empty((), dtype=torch.float32).element_size() * elements


def generate_figures(
    run_dir:       'str | Path',

    *,
    split:         str                         = "validation",
    max_tokens:    Optional[int]               = None,
    max_sequences: Optional[int]               = None,
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
    When both population caps are omitted, the belief/model banks use the controlled default of
    exactly 16,384 tokens. Explicit ``max_sequences`` calls preserve the exploratory compatibility
    path; the two caps are mutually exclusive. ``allow_large`` opts into the two full-vocabulary
    extractors when their estimated logits-plus-probabilities peak exceeds 8 GB; lighter inputs and
    figures still run when they are skipped. ``n_e_steps`` overrides the E-step trace length
    (default: the trained ``cfg.n_e_steps``). Returns the figure paths actually written
    (best-effort: a failed figure is logged and omitted).
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
    english_linguistic_diagnostics = supports_english_linguistic_taxonomies(dataset)
    figs.set_publication_style()

    if loader is None:
        loader = _build_loader(dataset, cfg, split)
    max_tokens, max_sequences, n_batches = _resolve_bank_budget(
        cfg,
        max_tokens=max_tokens,
        max_sequences=max_sequences,
    )
    controlled_bank = max_tokens is not None
    report_batches = _collect_batches(loader, n_batches)
    if not report_batches:
        raise RuntimeError(f"loader for {dataset!r}/{split!r} yielded no batches")

    full_vocab_gb = _estimated_full_vocab_bank_bytes(cfg, n_batches) / 1e9
    skip_full_vocab = full_vocab_gb > 8.0 and not allow_large
    if skip_full_vocab:
        logger.warning(
            "full-vocab figure inputs skipped: aggregate bank+workspace estimate %.1f GB exceeds "
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

    inference_bank = _safe(lambda: extract.collect_inference_bank(
        model,
        report_batches,
        max_batches=n_batches,
        device=device,
        return_logits=not skip_full_vocab,
    ), "inference_bank")
    if inference_bank is not None:
        retained_bank_gb = extract.inference_bank_nbytes(inference_bank) / 1e9
        logger.info(
            "report inference bank retains %.3f GB across %d CPU-hosted batches",
            retained_bank_gb,
            len(inference_bank),
        )
        if retained_bank_gb > 8.0 and not allow_large and not skip_full_vocab:
            logger.warning(
                "full-vocab figure inputs skipped: retained inference bank %.1f GB exceeds the "
                "8 GB guard; pass allow_large=True to override",
                retained_bank_gb,
            )
            for record in inference_bank:
                record["logits"] = None
            skip_full_vocab = True
    token_batches = (
        [record["tokens"] for record in inference_bank]
        if inference_bank is not None
        else _collect_token_batches(report_batches, device, n_batches)
    )
    tok = token_batches[0][:1].to(device)                              # one device-resident report sequence

    # ---- expensive model-replay inputs, each guarded (a failure skips only its figures) ----
    # The same-token diagnostics consume one captured forward. If capture itself fails, passing
    # snapshot=None preserves the previous per-extractor best-effort fallback instead of dropping
    # every dependent figure.
    snapshot = _safe(lambda: model.build_diagnostic_snapshot(tok), "diagnostic_snapshot")
    trace_snapshot = snapshot
    if (snapshot is not None and n_e_steps is not None
            and int(n_e_steps) != len(snapshot.trace_states) - 1):
        trace_snapshot = None
    trace       = _safe(lambda: extract.e_step_belief_trace(
        model, tok, n_iter=n_e_steps, snapshot=trace_snapshot), "e_step_belief_trace")
    layer_trace = _safe(lambda: extract.across_layer_belief_trace(
        model, tok, snapshot=snapshot), "across_layer_belief_trace")
    bank        = _safe(lambda: extract.belief_bank(
        model, token_batches, max_tokens=max_tokens, max_sequences=max_sequences,
        inference_bank=inference_bank), "belief_bank")
    cstate      = _safe(lambda: extract.converged_state(
        model, tok, snapshot=snapshot), "converged_state")
    ce_bank     = (None if skip_full_vocab else
                   _safe(lambda: extract.belief_ce_bank(
                       model, report_batches, device=device, max_batches=n_batches,
                       inference_bank=inference_bank),
                         "belief_ce_bank"))   # B1/EXP-3 Sigma_q<->CE join (calibration figures)
    amaps       = _safe(lambda: model.attention_maps(tok, snapshot=snapshot), "attention_maps")
    per_layer   = _safe(lambda: model.diagnostics_per_layer(
        tok, snapshot=snapshot), "diagnostics_per_layer")
    health      = _safe(lambda: extract.numerical_health(
        model, tok, snapshot=snapshot), "numerical_health")
    s_channel   = _safe(lambda: extract.s_channel_refinement(
        model, tok, snapshot=snapshot), "s_channel_refinement")
    mc_belief   = _safe(lambda: extract.model_channel_belief(
        model, tok, snapshot=snapshot), "model_channel_belief")
    r_centroid  = _safe(lambda: extract.hyper_prior_centroid(
        model, tok, snapshot=snapshot), "hyper_prior_centroid")
    h_coupling  = _safe(lambda: extract.hyper_prior_coupling(
        model, tok, snapshot=snapshot), "hyper_prior_coupling")
    mc_bank     = _safe(lambda: extract.model_channel_bank(
        model, token_batches, max_tokens=max_tokens, max_sequences=max_sequences,
        inference_bank=inference_bank),
                        "model_channel_bank")
    vstats      = (None if skip_full_vocab else
                   _safe(lambda: extract.vocab_prediction_stats(
                       model, token_batches, inference_bank=inference_bank),
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
    if not english_linguistic_diagnostics:
        logger.info(
            "English-only belief-category and vocabulary-confusion figures disabled for dataset %r",
            dataset,
        )

    # Decode-calibration reliability bins (conf/acc/frac) the run already wrote to research.json
    # (run_artifacts._calibration_and_strata); the B1/EXP-3 reliability diagram only plots them.
    reliability = None
    rj = run_dir / "research.json"
    if rj.exists():
        try:
            reliability = json.loads(rj.read_text()).get("reliability")
        except Exception as exc:                                  # best-effort, like every figure input
            logger.warning("research.json reliability unreadable (%s); reliability figure skipped", exc)

    model_channels = (("mu", "sigma", "phi")
                      if mc_bank is not None and "phi" in mc_bank else ("mu", "sigma"))
    belief_controlled_ready = (
        not controlled_bank
        or (bank is not None and bank["token_ids"].shape[0] == max_tokens)
    )
    model_controlled_ready = (
        not controlled_bank
        or (mc_bank is not None and mc_bank["token_ids"].shape[0] == max_tokens)
    )
    availability = {
        "estep_convergence":          trace is not None,
        "belief_trajectories":        trace is not None,
        "belief_category_separation": bank is not None,
        "attention_structure":        amaps is not None,
        "per_layer_diagnostics":      per_layer is not None,
        "gauge_equivariance":         cstate is not None,
        "gauge_head_specialization":  cstate is not None,
        "belief_spectrum":            cstate is not None,
        "spd_ellipses":               cstate is not None,
        "holonomy_curvature":         cstate is not None,
        "numerical_trust":            cstate is not None,
        "s_channel_refinement":       s_channel is not None,
        "model_channel_belief":       mc_belief is not None,
        "hyper_prior_centroid":       r_centroid is not None,
        "hyper_prior_coupling":       h_coupling is not None,
        "belief_umap_mu":             bank is not None and belief_controlled_ready,
        "belief_umap_sigma":          bank is not None and belief_controlled_ready,
        "belief_umap_phi":            bank is not None and belief_controlled_ready,
        "model_umap_mu":              mc_bank is not None and model_controlled_ready,
        "model_umap_sigma":           mc_bank is not None and model_controlled_ready,
        "model_umap_phi":             (mc_bank is not None and model_controlled_ready
                                       and "phi" in model_channels),
        "vocab_probability_heatmap":  vstats is not None,
        "vocab_calibration":          vstats is not None,
        "vocab_confusion":            vstats is not None and decode is not None,
        "decode_readout":             readout is not None,
        "reliability_diagram":        reliability is not None,
        "sigma_stratified_error":     ce_bank is not None,
        "sigma_ce_scatter":           ce_bank is not None,
    }
    planned_figures = frozenset(plan_single_run_figures(dataset, availability))

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
            outcomes = getattr(fig, "_vfe3_publication_outcomes", None)
            figs.plt.close(fig)
            written.append(path)
            logger.info("figure -> %s", path)
            if isinstance(outcomes, Mapping):
                sidecar = outcomes.get("sidecar")
                if isinstance(sidecar, Mapping) and sidecar.get("published"):
                    logger.info("sidecar -> %s", sidecar.get("path"))
                elif isinstance(sidecar, Mapping) and sidecar.get("error"):
                    logger.warning("figure %r sidecar failed (%s); figure retained",
                                   name, sidecar.get("error"))
        except Exception as exc:
            # The thunk can register a pyplot figure and then raise (tight_layout/savefig) before
            # _emit ever receives it; close what it registered (audit 2026-07-01 round-3).
            for num in set(figs.plt.get_fignums()) - _before:
                figs.plt.close(num)
            logger.warning("figure %r failed (%s); continuing", name, exc)

    _emit("estep_convergence",
          lambda p: figs.plot_estep_convergence(trace, path=p),
          "estep_convergence.png" in planned_figures)
    _emit("belief_trajectories",
          lambda p: figs.plot_belief_trajectories(trace, layer_trace, path=p),
          "belief_trajectories.png" in planned_figures)
    _emit("belief_category_separation",
          lambda p: figs.plot_belief_category_separation(bank, decode=decode, path=p),
          "belief_category_separation.png" in planned_figures)
    _emit("attention_structure",
          lambda p: figs.plot_attention_structure(amaps, path=p),
          "attention_structure.png" in planned_figures)
    # Per-head beta/gamma attention heatmaps are NOT emitted here: training already writes them at
    # every eval_interval (RunArtifacts.save_attention_maps / save_gamma_attention_maps under
    # <run_dir>/attention/), so end-of-training copies in figures/ were redundant.
    _emit("per_layer_diagnostics",                               # the depth axis the aggregates collapse
          lambda p: figs.plot_per_layer_diagnostics(per_layer, path=p),
          "per_layer_diagnostics.png" in planned_figures)
    _emit("gauge_equivariance",
          lambda p: figs.plot_gauge_equivariance(metrics.gauge_equivariance_residual(
              cstate["mu"], cstate["sigma"], cstate["omega"], model.group,
              kappa=cfg.kappa_beta, renyi_order=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
              diagonal=cfg.diagonal_covariance, divergence_family=cfg.divergence_family), path=p),
          "gauge_equivariance.png" in planned_figures)
    _emit("gauge_head_specialization",
          lambda p: figs.plot_gauge_head_specialization(
              metrics.per_head_gauge_invariants(cstate["exp_phi"], model.group.irrep_dims),
              # cstate["exp_phi"] is the FINAL block's gauge frame, so pair it with the FINAL layer's
              # per-head entropy (amaps[-1]) -- NOT the all-layer mean, which mismatched the depths.
              head_entropy=(metrics.attention_entropy_rows(amaps[-1]).mean(dim=-1) if amaps is not None else None),
              path=p),
          "gauge_head_specialization.png" in planned_figures)
    _emit("belief_spectrum",
          lambda p: figs.plot_belief_spectrum(
              cstate["sigma"], eps=cfg.eps, sigma_max=cfg.sigma_max,
              family=cfg.family, path=p),
          "belief_spectrum.png" in planned_figures)
    _emit("spd_ellipses",
          lambda p: figs.plot_spd_ellipses(
              cstate["mu"], cstate["sigma"], eps=cfg.eps,
              family=cfg.family, path=p),
          "spd_ellipses.png" in planned_figures)
    _emit("holonomy_curvature",
          lambda p: figs.plot_holonomy_curvature(
              metrics.holonomy_deviation_sampled(cstate["omega"]),
              curvature=metrics.curvature_field(cstate["omega"]), path=p),   # Panel C: spatial curvature field
          "holonomy_curvature.png" in planned_figures)
    _emit("numerical_trust",
          lambda p: figs.plot_numerical_trust(
              metrics.guard_saturation(cstate["sigma"], cstate["energy"], cstate["self_div"],
                                       eps=cfg.eps, sigma_max=cfg.sigma_max, kl_max=cfg.kl_max),
              health if health is not None else {},
              causal=(metrics.causal_sanity(amaps) if amaps is not None else None), path=p),
          "numerical_trust.png" in planned_figures)
    _emit("s_channel_refinement",                                 # only when s_e_step=True (else None)
          lambda p: figs.plot_s_channel_refinement(s_channel, path=p),
          "s_channel_refinement.png" in planned_figures)
    # Model-channel (s / r / h) and gamma_ij figures: present whenever the model channel is active
    # (s table exists); the r/h figures additionally require the centroid r (lambda_h>0 OR s_e_step),
    # so their extractors return None and _emit skips them when r is absent.
    _emit("model_channel_belief",                                 # the s figure
          lambda p: figs.plot_model_channel_belief(mc_belief, path=p),
          "model_channel_belief.png" in planned_figures)
    _emit("hyper_prior_centroid",                                 # the r figure
          lambda p: figs.plot_hyper_prior_centroid(r_centroid, path=p),
          "hyper_prior_centroid.png" in planned_figures)
    _emit("hyper_prior_coupling",                                 # the h figure (lambda_h block)
          lambda p: figs.plot_hyper_prior_coupling(h_coupling, path=p),
          "hyper_prior_coupling.png" in planned_figures)
    if controlled_bank and bank is not None and not belief_controlled_ready:
        logger.warning(
            "controlled belief UMAP skipped: requested %d tokens but loader supplied %d",
            max_tokens,
            bank["token_ids"].shape[0],
        )
    if controlled_bank and mc_bank is not None and not model_controlled_ready:
        logger.warning(
            "controlled model UMAP skipped: requested %d tokens but loader supplied %d",
            max_tokens,
            mc_bank["token_ids"].shape[0],
        )
    with figs.UMAPWorker() as umap_worker:
        for ch in ("mu", "sigma", "phi"):                         # one UMAP per belief channel
            _emit(f"belief_umap_{ch}",
                  lambda p, ch=ch: figs.plot_belief_umap(
                      bank, ch, decode=decode, controlled=controlled_bank,
                      english_linguistic_diagnostics=english_linguistic_diagnostics,
                      seeds=(embedding_comparison.CONTROLLED_SEEDS if controlled_bank else None),
                      umap_worker=umap_worker, path=p,
                      sidecar_path=(str(figdir / f"belief_umap_{ch}.json")
                                    if controlled_bank else None)),
                  f"belief_umap_{ch}.png" in planned_figures)
        for ch in model_channels:                                  # phi only for independent phi_tilde
            _emit(f"model_umap_{ch}",
                  lambda p, ch=ch: figs.plot_belief_umap(
                      mc_bank, ch, kind="Model", decode=decode, controlled=controlled_bank,
                      english_linguistic_diagnostics=english_linguistic_diagnostics,
                      seeds=(embedding_comparison.CONTROLLED_SEEDS if controlled_bank else None),
                      umap_worker=umap_worker, path=p,
                      sidecar_path=(str(figdir / f"model_umap_{ch}.json")
                                    if controlled_bank else None)),
                  f"model_umap_{ch}.png" in planned_figures)
    # Next-token vocabulary-probability figures (single-arm here; the cross-run K70-vs-K120 contrast
    # is the two-arm vocab_comparison_figures driver). vocab_confusion needs the token decoder for its
    # category bucketing; decode_readout is None (skipped) on the use_prior_bank=True KL-decode path.
    _emit("vocab_probability_heatmap",
          lambda p: figs.plot_vocab_probability_heatmap([{**vstats, "label": run_label}], decode=decode, path=p),
          "vocab_probability_heatmap.png" in planned_figures)
    _emit("vocab_calibration",
          lambda p: figs.plot_vocab_calibration([{**vstats, "label": run_label}], decode=decode, path=p),
          "vocab_calibration.png" in planned_figures)
    _emit("vocab_confusion",
          lambda p: figs.plot_vocab_confusion([{**vstats, "label": run_label}], decode=decode, path=p),
          "vocab_confusion.png" in planned_figures)
    _emit("decode_readout",
          lambda p: figs.plot_decode_readout([{**readout, "label": run_label}], decode=decode, path=p),
          "decode_readout.png" in planned_figures)

    # B1/EXP-3 -- Sigma_q-as-calibrated-uncertainty: the decode-reliability control + the two
    # Sigma_q<->CE figures (stratified error curve, rho scatter) off the belief_ce_bank join.
    _emit("reliability_diagram",
          lambda p: figs.plot_reliability_diagram(reliability, path=p),
          "reliability_diagram.png" in planned_figures)
    _emit("sigma_stratified_error",
          lambda p: figs.plot_sigma_stratified_error(ce_bank, path=p),
          "sigma_stratified_error.png" in planned_figures)
    _emit("sigma_ce_scatter",
          lambda p: figs.plot_sigma_ce_scatter(ce_bank, path=p),
          "sigma_ce_scatter.png" in planned_figures)

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


def compare_belief_umap_sidecars(
    sidecars: Sequence['str | Path'],
    labels:   Sequence[str],

    *,
    json_path:   'str | Path',
    figure_path: 'str | Path',
) -> ArtifactPublicationResult:
    """Validate controlled sidecars and write a metric-only JSON/PNG cross-run comparison."""
    records = []
    for sidecar in sidecars:
        path = Path(sidecar)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"controlled sidecar {path} is not a JSON object")
        records.append(payload)
    summary = embedding_comparison.comparison_summary(records, labels)
    output_json = Path(json_path)
    output_figure = Path(figure_path)
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    outcomes: Dict[str, Dict[str, object]] = {
        "figure": {"path": str(output_figure), "published": False, "error": None},
        "sidecar": {"path": str(output_json), "published": False, "error": None},
    }
    figure = None
    published_figure: Optional[Path] = None
    published_json: Optional[Path] = None
    try:
        figure = figs.plot_controlled_embedding_comparison(
            summary,
            path=str(output_figure),
        )
        outcomes["figure"]["published"] = True
        published_figure = output_figure
    except Exception as exc:
        outcomes["figure"]["error"] = str(exc)
    finally:
        if figure is not None:
            figs.plt.close(figure)
    try:
        embedding_comparison.write_json_atomic(summary, output_json)
        outcomes["sidecar"]["published"] = True
        published_json = output_json
    except Exception as exc:
        outcomes["sidecar"]["error"] = str(exc)
    return ArtifactPublicationResult(published_json, published_figure, outcomes)


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
    english_linguistic_diagnostics = all(
        supports_english_linguistic_taxonomies(dataset)
        for _, _, dataset, _ in prepared
    )
    if not english_linguistic_diagnostics:
        (logger or logging.getLogger(__name__)).info(
            "English-only vocabulary confusion disabled for non-English comparison datasets"
        )

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
          bool(arms_pred) and decode is not None and english_linguistic_diagnostics)
    _emit("decode_readout_compare",
          lambda: figs.plot_decode_readout(arms_readout, decode=decode), bool(arms_readout))

    logger.info("wrote %d comparison figures to %s", len(written), out)
    return written
