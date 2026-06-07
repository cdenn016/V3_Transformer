import math

import torch
from torch.utils.data import DataLoader

import vfe3.data.datasets as datasets_mod
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows, tokens_per_char
from vfe3.model.model import VFEModel
from vfe3.train import evaluate


def test_evaluate_applies_tokens_per_char_to_bpc_only():
    # bpc scales linearly with tokens_per_char (bits/token at 1.0); ce and ppl are unaffected.
    cfg = VFE3Config(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1,
                     gauge_group="block_glk", pos_phi="none")
    torch.manual_seed(0)
    model = VFEModel(cfg)
    ds = TokenWindows(torch.arange(8).repeat(20), 6)
    loader = DataLoader(ds, batch_size=4, drop_last=True)
    m1 = evaluate(model, loader, tokens_per_char=1.0)
    m3 = evaluate(model, loader, tokens_per_char=3.0)
    assert abs(m1["bpc"] - m1["ce"] / math.log(2.0)) < 1e-9        # default 1.0 = bits-per-token
    assert abs(m3["bpc"] - 3.0 * m1["bpc"]) < 1e-9                 # linear in tokens_per_char
    assert abs(m3["ce"] - m1["ce"]) < 1e-12                        # ce untouched
    assert abs(m3["ppl"] - m1["ppl"]) < 1e-9                       # ppl untouched


def test_tokens_per_char_counts_unicode_codepoints(monkeypatch):
    # 5 tokens decoding to a 2-codepoint string -> tokens_per_char = 5/2 = 2.5 (codepoints, not bytes).
    monkeypatch.setattr(datasets_mod, "get_tiktoken_decoder", lambda ds: (lambda ids: "ab"))
    monkeypatch.setattr(datasets_mod, "load_cached_tokens",
                        lambda ds, split, cache_dir=None: torch.arange(5))
    datasets_mod._TOKENS_PER_CHAR_CACHE.clear()
    assert abs(tokens_per_char("wikitext-103", "test") - 2.5) < 1e-9


def test_tokens_per_char_none_for_synthetic_so_caller_keeps_bits_per_token():
    # No real tokenizer -> None -> the entry point leaves tokens_per_char=1.0 (honest bits/token).
    assert tokens_per_char("synthetic-period3", "validation") is None
