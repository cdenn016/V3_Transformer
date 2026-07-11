r"""Click-to-run for the tests that NEED the GPU (the CPU audit suite is check_audit_fixes.py).

Two things live here, both requiring CUDA (your RTX 5090):

  * t3  -- the learnability gate ``test_training_decreases_loss_on_structured_stream``. It is kept as
           a NON-strict xfail because on CPU the per-head-tau model clears the ln(3) floor by only a
           thin margin and the LRs were calibrated for the old sqrt(embed_dim) tau. This script RUNS
           the same 3-seed period-3 training on the GPU and REPORTS the median end-CE, the ln(3)-margin
           threshold, and the headroom -- so you can decide whether to promote it to a hard gate (and
           re-tune the LRs via LR_OVERRIDES below).
  * t6  -- the CUDA-only tests that are SKIPPED on CPU: the Laplace CPU<->CUDA agreement test and the
           efe-scorer device-aware regression. Run here via pytest with VFE3_TEST_DEVICE=cuda.

No CLI args -- edit the config, then::

    python check_gpu_tests.py
"""

import os

# ---- config (edit me) ------------------------------------------------------
N_SEEDS      = 3          # t3 averages the median end-CE over this many fixed seeds
MAX_STEPS    = 200        # t3 training steps per seed (the gate's default)
LR_OVERRIDES = {}         # re-tune for the per-head tau, e.g. {"e_phi_lr": 0.4, "m_phi_lr": 0.06}
RUN_T6       = True       # also run the CUDA-only pytest tests (Laplace agreement, efe-scorer device)
# ---------------------------------------------------------------------------

os.environ["VFE3_TEST_DEVICE"] = "cuda"    # the t6 pytest tests read this at import

import math
import sys
from dataclasses import replace

import torch

from check_junit import run_pytest_junit


def _run_t3(dev):
    from tests.test_train import (_MARGINAL_ENTROPY_P3, _CUTOVER_MARGIN,
                                  _median, _periodic_loader, _structured_cfg)
    from vfe3.model.model import VFEModel
    from vfe3.train import train

    threshold = _MARGINAL_ENTROPY_P3 - _CUTOVER_MARGIN
    print("t3 -- learnability gate (period-3 next-token structure, gauge transport ON):")
    ends = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        cfg = _structured_cfg()
        if LR_OVERRIDES:
            cfg = replace(cfg, **LR_OVERRIDES)
        cfg = replace(cfg, max_steps=MAX_STEPS)
        model = VFEModel(cfg).to(dev)
        losses = train(model, _periodic_loader(V=6, period=3, seed=seed),
                       cfg, n_steps=MAX_STEPS, device=dev)
        ends.append(float(losses[-1]))
        print(f"    seed {seed}: end CE = {ends[-1]:.4f}")
    med = _median(ends)
    headroom = threshold - med
    clears = headroom > 0.0
    print(f"    median end CE = {med:.4f}   threshold = ln(3)-{_CUTOVER_MARGIN} = {threshold:.4f}")
    print(f"    headroom = {headroom:+.4f}  ->  {'CLEARS the floor (learns the period)' if clears else 'FAILS (pins at ln 3)'}")
    if clears and headroom < 0.05:
        print("    NOTE: clears, but headroom < 0.05 -- still thin; consider LR_OVERRIDES before promoting to strict.")
    print()
    return clears


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available in this environment -- these tests NEED the GPU.")
        print("Run this on the RTX 5090 (and make sure torch was installed with CUDA support).")
        return 2
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}\n")

    t3_ok = _run_t3(dev)

    t6_code = 0
    t6_counts = None
    if RUN_T6:
        t6 = [
            "tests/test_laplace_family.py::test_laplace_cuda_matches_cpu",          # t6: Laplace CPU<->CUDA agreement
            "tests/test_efe_scorer.py::test_preference_builders_are_device_aware",  # t6: efe-scorer device regression
        ]
        print("t6 -- CUDA-only pytest tests:")
        t6_code, t6_counts = run_pytest_junit(
            t6 + ["-v", "-p", "no:cacheprovider"],
            prefix="vfe3-gpu-t6-",
        )

    bar = "=" * 64
    print("\n" + bar)
    if RUN_T6:
        t6_summary = (
            f"  |  t6 {'GREEN' if t6_code == 0 else f'FAIL (exit {t6_code})'} "
            f"({t6_counts['passes']} passed, {t6_counts['skipped']} skipped, "
            f"{t6_counts['failures']} failed, {t6_counts['errors']} errors)"
        )
    else:
        t6_summary = ""
    print(f"GPU CHECK: t3 gate {'CLEARS' if t3_ok else 'FAILS'} the ln(3) margin" + t6_summary)
    print(bar)
    # exit 0 only if the t3 gate clears AND (t6 ran green or was skipped)
    return 0 if (t3_ok and t6_code == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
