"""Tests for VFEModel.generate -- the additive autoregressive sampler.

generate reuses VFEModel.forward (targets=None -> logits (B, N, V)), takes the
last-position logits, turns them into a next token (greedy or sampled), appends,
and repeats. It never touches the training/loss branch, so it cannot corrupt
training; that training-isolation is the safety oracle below.

Forward-comparison oracles (greedy == forward+argmax, top_k membership) hold only
for the FIRST generated token: generate's step-1 internal forward sees exactly the
prompt, so logits[:, -1, :] there equals forward(prompt)[:, -1, :]. Step 2+
conditions on a longer sequence and has no relation to forward(prompt). Those
oracles therefore use max_new_tokens=1.
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _tiny_model(seed: int = 0, **overrides) -> VFEModel:
    """A tiny fixed-seed VFEModel. vocab_size is kept comfortably larger than any
    top_k used below so torch.topk never sees k > vocab_size."""
    base = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0, seed=seed)
    base.update(overrides)
    cfg = VFE3Config(**base)
    torch.manual_seed(seed)
    return VFEModel(cfg)


def test_shape_in_vocab_and_prompt_preserved():
    model = _tiny_model()
    V = model.cfg.vocab_size
    B, N0 = 2, 3
    prompt = torch.randint(0, V, (B, N0))
    out = model.generate(prompt, max_new_tokens=5)
    assert out.shape == (B, N0 + 5)
    assert (out >= 0).all() and (out < V).all()            # every id in [0, V)
    assert torch.equal(out[:, :N0], prompt)                # prompt preserved


def test_greedy_is_deterministic():
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    a = model.generate(prompt, max_new_tokens=4, greedy=True)
    b = model.generate(prompt, max_new_tokens=4, greedy=True)
    assert torch.equal(a, b)


def test_greedy_equals_forward_argmax_first_token():
    # The pin: generate's first greedy token == argmax of the last-position logits
    # from a direct forward on the SAME prompt (no hidden divergence). First token
    # only -- step 2+ conditions on a longer sequence. Prompt <= max_seq_len so the
    # internal forward and this comparison forward see identical input.
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    logits = model(prompt)                                 # (B, N0, V), targets=None
    expected = logits[:, -1, :].argmax(dim=-1)             # (B,)
    out = model.generate(prompt, max_new_tokens=1, greedy=True)
    assert torch.equal(out[:, -1], expected)


def test_greedy_ignores_temperature_topk_topp():
    # Greedy returns before any temperature/top_k/top_p logic: a wild temperature
    # and aggressive top_k/top_p alongside greedy=True must not change the result.
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    plain = model.generate(prompt, max_new_tokens=4, greedy=True)
    wild = model.generate(prompt, max_new_tokens=4, greedy=True,
                          temperature=137.0, top_k=1, top_p=0.01)
    assert torch.equal(plain, wild)


def test_top_k_one_is_argmax_first_token():
    # top_k=1 (not greedy) leaves a single survivor -> softmax mass 1.0 -> multinomial
    # returns it deterministically; it equals the argmax token. No seed needed.
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    logits = model(prompt)
    expected = logits[:, -1, :].argmax(dim=-1)
    out = model.generate(prompt, max_new_tokens=1, greedy=False, top_k=1)
    assert torch.equal(out[:, -1], expected)


def test_top_k_membership_first_token():
    # The first sampled token must lie among the k largest of the last-position logits.
    model = _tiny_model()
    V = model.cfg.vocab_size
    k = 3
    prompt = torch.randint(0, V, (2, 3))
    logits = model(prompt)
    topk_ids = logits[:, -1, :].topk(k, dim=-1).indices     # (B, k)
    torch.manual_seed(0)
    out = model.generate(prompt, max_new_tokens=1, greedy=False, top_k=k)
    chosen = out[:, -1]                                     # (B,)
    assert (chosen.unsqueeze(-1) == topk_ids).any(dim=-1).all()


def test_top_p_and_temperature_paths_run_in_vocab():
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    torch.manual_seed(0)
    out_p = model.generate(prompt, max_new_tokens=3, greedy=False, top_p=0.9)
    out_t = model.generate(prompt, max_new_tokens=3, greedy=False, temperature=0.7)
    for out in (out_p, out_t):
        assert out.shape == (2, 3 + 3)
        assert (out >= 0).all() and (out < V).all()


def test_prompt_longer_than_max_seq_len_does_not_error():
    # A prompt longer than max_seq_len must not error: the loop truncates to the last
    # max_seq_len tokens before each forward. The returned sequence keeps the FULL prompt.
    model = _tiny_model()
    V = model.cfg.vocab_size
    L = model.cfg.max_seq_len
    long_prompt = torch.randint(0, V, (2, L + 4))           # longer than max_seq_len
    out = model.generate(long_prompt, max_new_tokens=2, greedy=True)
    assert out.shape == (2, (L + 4) + 2)
    assert torch.equal(out[:, : L + 4], long_prompt)        # full prompt preserved
    assert (out >= 0).all() and (out < V).all()


def test_generate_is_training_isolated():
    # The safety oracle: generate changes no parameter and does not break training.
    model = _tiny_model()
    V = model.cfg.vocab_size
    before = model.prior_bank.mu_embed.detach().clone()
    prompt = torch.randint(0, V, (2, 3))
    torch.manual_seed(0)
    model.generate(prompt, max_new_tokens=4, greedy=False, top_k=2, top_p=0.9)
    after = model.prior_bank.mu_embed.detach().clone()
    assert torch.equal(before, after)                       # no parameter changed
    # training forward still produces a finite loss after a generate call
    tokens = torch.randint(0, V, (2, 4)); targets = torch.randint(0, V, (2, 4))
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)


def test_policy_mode_rejects_call_time_sampler_knobs():
    # audit F9 (2026-06-28): under a policy scorer, generate() routes through _policy_select, which uses
    # policy_top_k / policy_precision and does NOT consume call-time temperature/top_k/top_p. Those must
    # be rejected at the call rather than silently dropped; 'greedy' stays honored. ('flat' is the only
    # generic-safe preference -- see test_generic_policy_rejects_context_requiring_preference.)
    model = _tiny_model(policy_mode="efe_one_step", policy_preference="flat")
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    for kwargs in ({"temperature": 0.7}, {"top_k": 3}, {"top_p": 0.9}):
        with pytest.raises(ValueError):
            model.generate(prompt, max_new_tokens=1, **kwargs)
    # the call-time defaults (temperature=1.0, no top_k/top_p) are accepted; greedy is honored
    out = model.generate(prompt, max_new_tokens=2, greedy=True)
    assert out.shape == (2, 3 + 2) and (out >= 0).all() and (out < V).all()
