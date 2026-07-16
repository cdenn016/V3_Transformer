r"""Click-to-run: side-by-side vocabulary next-token probability figures for two trained runs.

Edit ``CONFIG`` below and run. Loads two run directories written by ``train_vfe3.py``
(``config.json`` + ``best_model.pt``), runs one inference pass per run, and writes the four
two-arm comparison figures to ``out_dir``:

  * ``vocab_probability_heatmap_compare`` -- Seq x top-k ``p(o_{n+1} | o_{<=n})`` per run (green
    box = true next token); horizontal bands at every position are the signature of prior collapse.
  * ``vocab_calibration_compare``         -- mean predicted prob vs empirical unigram (log-log);
    mass on ``y=x`` with context-gain near zero means the model fell back to the marginal.
  * ``vocab_confusion_compare``           -- row-normalized next-token category confusion.
  * ``decode_readout_compare``            -- the linear decode matrix ``W`` (``logits = mu_q @ W^T``).

The default arms are the K70 (good, ppl ~83.6) and K120 (degraded, ppl ~96.8) runs, the collapse
contrast under investigation. This is a SEPARATE, opt-in step from training.
"""

import os
if os.environ.get("VFE3_ALLOW_DUPLICATE_OPENMP") == "1":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging

import torch

from vfe3.viz.report import vocab_comparison_figures

CONFIG = {
    "run_dirs": [
        "vfe3_runs/84.65_wikitext-103_K70_block_glk_linear_mix_s6-3epoch",
        "vfe3_runs/97.09_wikitext-103_K120_block_glk_linear_mix_s6-3epoch",
    ],
    "labels":        ["K70 (ppl 83.6)", "K120 (ppl 96.8)"],
    "out_dir":       "vfe3_runs/_vocab_compare",
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "split":         "validation",
    "max_sequences": 256,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"\nVFE_3.0 vocabulary-probability comparison\n  runs: {CONFIG['run_dirs']}\n"
          f"  device: {CONFIG['device']}")
    paths = vocab_comparison_figures(
        CONFIG["run_dirs"],
        CONFIG["out_dir"],
        labels=CONFIG["labels"],
        device=torch.device(CONFIG["device"]),
        split=CONFIG["split"],
        max_sequences=CONFIG["max_sequences"],
    )
    print(f"\nwrote {len(paths)} comparison figures to {CONFIG['out_dir']}")


if __name__ == "__main__":
    main()
