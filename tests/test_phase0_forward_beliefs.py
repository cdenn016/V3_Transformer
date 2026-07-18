r"""Regression gate for the shared ``forward_beliefs`` extraction.

The refactor factors the inline belief pipeline of ``VFEModel.forward`` (encode -> pos_phi -> optional
s-refine -> precision-bias fold -> vfe_stack -> final_norm) into the shared seam ``forward_beliefs``,
It must leave ``forward`` and ``generate`` byte-identical to the pre-remediation baseline.

Two complementary guards, mirroring tests/test_perf_equivalence.py's "freeze the pre-refactor numerics"
discipline:
  (1) scalar checksums rebaselined at K=4 from unmodified `origin/main` 1b58d4f before the
      2026-07-13 remediation,
      across the configs that exercise every touched branch: (a) inference logits, (b) dense training,
      (c) fused-chunked training, (d) mstep_self_coupling_weight>0, (e) mass_phi>0, (f) linear decode.
  (2) a within-version seam invariant asserted bit-exactly with torch.equal (machine-independent):
      the return_logits=True branch equals forward(targets=None).
"""
import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

# --- frozen origin/main 1b58d4f K=4 checksums (seed 0, the configs/tokens below) ---
_A_LOGITS_SUM = -0.3824235201        # (a) inference logits, KL-to-prior decode
_B_LOGITS_SUM = -0.3824235201        # (b) dense-train logits (== inference: same decode)
_B_LOSS = 2.7726001740               # (b) dense training loss == ce (no mass_phi / mstep)
_C_LOSS = 2.7726004124               # (c) fused-chunked CE (matches dense to ~1e-6, distinct kernel)
_D_LOSS = 2.7726044655               # (d) loss WITH mstep self-coupling (> ce)
_D_CE = 2.7726001740                 # (d) ce unchanged by the mstep term
_E_LOSS = 2.7726650238               # (e) loss WITH mass_phi penalty (> ce)
_E_CE = 2.7726004124                 # (e) ce from the mass-penalty path
_F_LOGITS_SUM = -0.1151125083        # (f) linear-decode inference logits
_F_LOSS = 2.7732081413               # (f) linear-decode training loss == ce
_GEN_GREEDY = [[1, 2, 3, 3, 3, 3, 3, 3]]   # greedy generate (argmax) HEAD pin

_TOK = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 0]])
_TGT = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 0, 1]])
_PROMPT = torch.tensor([[1, 2, 3]])
_ATOL = 1e-5


def _base(**kw) -> VFE3Config:
    d = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=2,
             n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.02, use_prior_bank=True, seed=0)
    d.update(kw)
    return VFE3Config(**d)


def _build(cfg: VFE3Config) -> VFEModel:
    torch.manual_seed(0)
    return VFEModel(cfg)


def test_inference_logits_match_head():
    m = _build(_base())
    with torch.no_grad():
        logits = m(_TOK)
    assert logits.shape == (2, 8, 16)
    assert abs(float(logits.double().sum()) - _A_LOGITS_SUM) < _ATOL


def test_dense_training_matches_head():
    m = _build(_base(decode_mode="diagonal"))
    with torch.no_grad():
        logits, loss, ce = m(_TOK, _TGT)
    assert abs(float(logits.double().sum()) - _B_LOGITS_SUM) < _ATOL
    assert abs(float(loss) - _B_LOSS) < _ATOL
    assert abs(float(ce) - _B_LOSS) < _ATOL          # loss == ce when no mass_phi / mstep


def test_fused_chunked_training_matches_head():
    m = _build(_base(decode_mode="diagonal_chunked"))
    with torch.no_grad():
        logits, loss, ce = m(_TOK, _TGT)
    assert logits is None                            # the fused path forms no (B,N,V) tensor
    assert abs(float(loss) - _C_LOSS) < _ATOL
    assert abs(float(ce) - _C_LOSS) < _ATOL


def test_mstep_self_coupling_matches_head():
    m = _build(_base(mstep_self_coupling_weight=0.5))
    with torch.no_grad():
        _, loss, ce = m(_TOK, _TGT)
    assert abs(float(loss) - _D_LOSS) < _ATOL        # the mstep term reads cap['prior']/['out']/['converged']
    assert abs(float(ce) - _D_CE) < _ATOL
    assert float(loss) > float(ce)                   # the term strictly adds to the loss


def test_mass_phi_matches_head():
    m = _build(_base(mass_phi=0.3))
    with torch.no_grad():
        _, loss, ce = m(_TOK, _TGT)
    assert abs(float(loss) - _E_LOSS) < _ATOL        # mass_phi reads belief.phi (== out.phi)
    assert abs(float(ce) - _E_CE) < _ATOL
    assert float(loss) > float(ce)


def test_linear_decode_ablation_matches_head():
    m = _build(_base(use_prior_bank=False))
    with torch.no_grad():
        logits = m(_TOK)
    assert abs(float(logits.double().sum()) - _F_LOGITS_SUM) < _ATOL
    m = _build(_base(use_prior_bank=False))
    with torch.no_grad():
        _, loss, ce = m(_TOK, _TGT)
    assert abs(float(loss) - _F_LOSS) < _ATOL
    assert abs(float(ce) - _F_LOSS) < _ATOL


def test_generate_greedy_matches_head():
    m = _build(_base())
    with torch.no_grad():
        seq = m.generate(_PROMPT, 5, greedy=True)
    assert seq.tolist() == _GEN_GREEDY


def test_generate_sampled_is_reproducible():
    # generate() is unchanged in Phase 0; sampled output is RNG-sensitive (multinomial) and so not
    # pinned cross-machine, but it must be reproducible given the same seed within a run.
    m = _build(_base())
    torch.manual_seed(123)
    with torch.no_grad():
        s1 = m.generate(_PROMPT, 5, greedy=False, temperature=1.0, top_k=5)
    torch.manual_seed(123)
    with torch.no_grad():
        s2 = m.generate(_PROMPT, 5, greedy=False, temperature=1.0, top_k=5)
    assert torch.equal(s1, s2)


def test_seam_forward_beliefs_equals_forward_inference():
    # The load-bearing invariant: the return_logits=True branch is bit-identical to forward(targets=None).
    m = _build(_base())
    with torch.no_grad():
        belief, fb_logits = m.forward_beliefs(_TOK, return_logits=True)
        fwd_logits = m(_TOK)
    assert isinstance(belief, BeliefState)
    assert belief.mu.shape == (2, 8, 4) and (belief.sigma > 0).all()
    assert torch.equal(fb_logits, fwd_logits)
    assert m.forward_beliefs(_TOK, return_logits=False)[1] is None


def test_capture_out_param_enriched_on_mstep_path():
    # The mstep self-coupling out-param carries the three pre-transform intermediates forward needs:
    # the converged q* (from vfe_stack), the encode-time prior, and the raw pre-final_norm output.
    m = _build(_base(mstep_self_coupling_weight=0.5))
    cap: dict = {}
    with torch.no_grad():
        belief, _ = m.forward_beliefs(_TOK, return_logits=False, capture=cap)
    assert set(cap) >= {"converged", "prior", "out"}
    assert isinstance(cap["prior"], BeliefState) and isinstance(cap["out"], BeliefState)
    # belief.mu is post-final_norm; cap['out'].mu is the raw stack output it is derived from.
    assert cap["out"].mu.shape == belief.mu.shape
    assert torch.equal(belief.sigma, cap["out"].sigma)   # final_norm transforms only the mean
