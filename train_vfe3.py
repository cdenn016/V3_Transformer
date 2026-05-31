r"""Click-to-run training entry for the VFE_3.0 transformer.

Mirrors VFE_2.0 ``train_vfe.py``: edit the ``config`` dict below, pick a ``DATASET``,
then run ``python train_vfe3.py``. There is no CLI arg parsing.

The default ``DATASET = "synthetic-period3"`` runs end-to-end with zero external data:
a deterministic period-3 token stream (the cutover anchor used by
``tests/test_train.py``). Selecting a real corpus (e.g. ``"wikitext-2"``) tries the
tokenized cache and falls back to the synthetic stream if the cache is absent, so this
file never crashes for lack of data.
"""

import logging

import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows, make_dataloader
from vfe3.train import evaluate, train


# --- click-to-run knobs -------------------------------------------------------
SEED = 0
DATASET = "synthetic-period3"                       # "synthetic-period3" | "wikitext-2" | ...
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Small period-3-learning defaults (the structured-stream cutover regime).
config = dict(
    vocab_size=6,
    embed_dim=4,
    n_heads=2,
    max_seq_len=8,
    n_layers=1,
    n_e_steps=3,
    e_mu_lr=0.3,
    e_phi_lr=0.3,
    m_mu_lr=0.05,
    m_sigma_lr=0.01,
    m_phi_lr=0.05,
    warmup_steps=5,
    max_steps=200,
    log_interval=20,
    eval_interval=100,
    seed=SEED,
)


def synthetic_period3_loader(period=3, n=600, seq_len=8, batch_size=8, seed=0) -> DataLoader:
    r"""Deterministic period-3 token stream (mirrors tests/test_train.py)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(period).repeat(n // period + 2)
    ds = TokenWindows(base[:n].to(torch.long), seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)


def _banner(model, cfg: VFE3Config, dataset: str, device: str, n_steps: int) -> str:
    n_params = sum(p.numel() for p in model.parameters())
    bar = "=" * 64
    return "\n".join([
        bar,
        f" Gauge VFE Transformer | {n_params} params | {device}",
        bar,
        f" K={cfg.embed_dim}  N={cfg.max_seq_len}  L={cfg.n_layers}  heads={cfg.n_heads}  "
        f"group={cfg.gauge_group}  family={cfg.family}",
        f" steps={n_steps}  batch={cfg.batch_size}  dataset={dataset}",
        f" M-LRs: mu={cfg.m_mu_lr}  sigma={cfg.m_sigma_lr}  phi={cfg.m_phi_lr}",
        f" VFE: alpha={cfg.alpha}  kappa={cfg.kappa}  tau={cfg.tau:.4f}  mass_phi={cfg.mass_phi}",
        f" seed={cfg.seed}",
        bar,
    ])


def _select_loader(dataset: str, cfg: VFE3Config, logger: logging.Logger) -> DataLoader:
    if dataset == "synthetic-period3":
        return synthetic_period3_loader(seq_len=cfg.max_seq_len, batch_size=cfg.batch_size, seed=cfg.seed)
    try:
        return make_dataloader(dataset, "train", cfg.max_seq_len, cfg.batch_size, max_tokens=8192)
    except FileNotFoundError:
        logger.warning("cache for %r absent; falling back to synthetic-period3", dataset)
        return synthetic_period3_loader(seq_len=cfg.max_seq_len, batch_size=cfg.batch_size, seed=cfg.seed)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("train_vfe3")

    from vfe3.model.model import VFEModel

    cfg = VFE3Config(**config)
    torch.manual_seed(cfg.seed)
    model = VFEModel(cfg).to(DEVICE)
    loader = _select_loader(DATASET, cfg, logger)

    logger.info(_banner(model, cfg, DATASET, DEVICE, cfg.max_steps))
    train(
        model, loader, cfg,
        n_steps=cfg.max_steps,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_loader=loader,
        device=torch.device(DEVICE),
        logger=logger,
    )
    m = evaluate(model, loader, device=torch.device(DEVICE))
    logger.info("=" * 64)
    logger.info(
        "Final | Loss: %.4f | CE: %.4f | PPL: %.1f | BPC: %.4f",
        m["ce"], m["ce"], m["ppl"], m["bpc"],
    )


if __name__ == "__main__":
    main()
