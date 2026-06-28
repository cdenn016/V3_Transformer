r"""Pre-registration-invariant tests for the EFE ring experiment script (efe_ring_experiment.py;
spec Section 4.1). Pins sampler CORRECTNESS -- uniform goal != s0 -- which the audit 2026-06-28 found
unpinned (a green suite survived a biased sampler, finding F2/F6).

Config TOGGLES (steps, seeds, batch_size, ...) are intentionally NOT pinned here: they are the user's
live pre-registration surface, edited between runs, and any deviation is already logged in the run
output. Only behavioral contracts of the harness functions are pinned.
"""
import torch

import efe_ring_experiment as exp
from vfe3.inference import ring_task as rt


def test_sample_episodes_excludes_start_and_is_uniform():
    n = 200_000
    goals, s0 = exp.sample_episodes(n, seed=0, device=torch.device("cpu"))
    assert goals.shape == (n,) and s0.shape == (n,)
    assert bool((goals != s0).all())                          # g != s0 always (spec Section 4.1)
    assert int(goals.min()) >= 0 and int(goals.max()) < rt.M  # on the ring
    # the ring offset (goal - s0) mod M must be uniform over the M-1 nonzero values, none ~2x another.
    offset = (goals - s0) % rt.M
    counts = torch.bincount(offset, minlength=rt.M).float()
    assert float(counts[0]) == 0.0                            # never the zero offset
    expected = n / (rt.M - 1)
    # pre-fix the clockwise neighbor (offset 1) carried ~2x the mass; require all within 10% of uniform.
    assert float((counts[1:] - expected).abs().max()) < 0.1 * expected


def test_sample_episodes_respects_device():
    goals, s0 = exp.sample_episodes(8, seed=1, device=torch.device("cpu"))
    assert goals.device.type == "cpu" and s0.device.type == "cpu"
    assert goals.shape == (8,) and s0.shape == (8,)
