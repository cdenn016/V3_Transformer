r"""Run-folder naming for the click-to-run entry (``train_vfe3``): the descriptive label, the
timestamped in-progress directory, and the finalize-time rename to ``<test_ppl>_<label>`` (no
timestamp) the user organizes runs by. Pure filesystem/string logic -- no model is built."""
import logging
import math

import pytest

import train_vfe3
from vfe3.config import VFE3Config

_LOG = logging.getLogger("test_run_naming")


def _cfg(**kw) -> VFE3Config:
    base = dict(vocab_size=6, embed_dim=20, n_heads=2, max_seq_len=8, n_layers=1,
                gauge_group="block_glk", use_prior_bank=False, use_head_mixer=True, seed=0)
    base.update(kw)
    return VFE3Config(**base)


def test_run_label_is_descriptive_with_tags_and_no_timestamp():
    assert train_vfe3._run_label(_cfg(), "wikitext-103") == "wikitext-103_K20_block_glk_linear_mix_s0"


def test_run_label_drops_tags_when_off():
    # use_prior_bank=True drops _linear; use_head_mixer=False drops _mix
    label = train_vfe3._run_label(_cfg(use_prior_bank=True, use_head_mixer=False), "wikitext-103")
    assert label == "wikitext-103_K20_block_glk_s0"


def test_run_label_seed_suffix_distinguishes_seeds():
    # the _s<seed> suffix keeps a multi-seed launch's run folders distinct (and seed-identifiable)
    assert train_vfe3._run_label(_cfg(seed=3), "wikitext-103").endswith("_s3")
    assert train_vfe3._run_label(_cfg(seed=64), "wikitext-103").endswith("_s64")


def test_run_dir_prefixes_label_with_timestamp():
    rd = train_vfe3._run_dir(_cfg(), "wikitext-103")
    name = rd.replace("\\", "/").split("/")[-1]
    stamp, _, rest = name.partition("_")
    assert "-" in stamp and stamp.replace("-", "").isdigit()        # YYYYMMDD-HHMMSS
    assert rest == "wikitext-103_K20_block_glk_linear_mix_s0"


def test_rename_uses_test_ppl_and_drops_timestamp(tmp_path):
    label = "wikitext-103_K20_block_glk_linear_mix"
    src = tmp_path / f"20260607-124645_{label}"
    src.mkdir()
    (src / "summary.json").write_text("{}")                          # a real artifact must move with it
    out = train_vfe3._rename_run_by_ppl(str(src), label, 154.293, _LOG)
    assert out.replace("\\", "/").split("/")[-1] == f"154.29_{label}"
    assert not src.exists()
    assert (tmp_path / f"154.29_{label}" / "summary.json").exists()


def test_rename_collision_gets_numeric_suffix(tmp_path):
    label = "wikitext-103_K20_block_glk_linear_mix"
    (tmp_path / f"154.29_{label}").mkdir()                           # an existing run at the same PPL
    src = tmp_path / f"20260607-124645_{label}"
    src.mkdir()
    out = train_vfe3._rename_run_by_ppl(str(src), label, 154.29, _LOG)
    assert out.replace("\\", "/").split("/")[-1] == f"154.29_{label}_2"
    assert not src.exists()


def test_rename_skips_when_ppl_missing_or_nonfinite(tmp_path):
    label = "wikitext-103_K20_block_glk_linear_mix"
    src = tmp_path / f"20260607-124645_{label}"
    src.mkdir()
    for ppl in (None, float("inf"), float("nan")):
        out = train_vfe3._rename_run_by_ppl(str(src), label, ppl, _LOG)
        assert out == str(src) and src.exists()                      # kept under the timestamped name


def test_rename_skips_when_dir_absent(tmp_path):
    missing = str(tmp_path / "20260607-124645_nope")
    assert train_vfe3._rename_run_by_ppl(missing, "nope", 100.0, _LOG) == missing


def test_single_run_rejects_seed_precedence_mismatch():
    with pytest.raises(ValueError, match="SEEDS.*config.*seed"):
        train_vfe3._resolve_seeds({"seed": 6}, seeds=(54,), num_runs=1)
