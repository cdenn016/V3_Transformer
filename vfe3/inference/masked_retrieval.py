r"""The partially observed masked key-value retrieval task: the H>=2 epistemic task for the EFE
policy experiment (spec docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md,
Section 5 Phase 3; design doc docs/research/active-inference/2026-06-29-masked-retrieval-task-design.md).

Each episode presents a table of key->value pairs with the TARGET key's value masked, then the query
``QUERY k_t``. The agent emits compound action tokens: ``PROBE_k_i`` makes the environment append the
revealed value token v_{vals[i]} (a new observation), and ``ANSWER_v_j`` commits and terminates
(correct iff j == vals[target]). The only reliable route is the two-step ``PROBE_k_t ; ANSWER_v_{vals[t]}``,
so the task turns on an information-seeking action and is realizable only at H>=2 -- the non-degenerate
home for the matched-compute beam / best-of-N baselines (which reduce to existing arms at H=1).

Training is teacher-forced on RANDOM-action trajectories (a probe of a RANDOM key, its revealed value,
then the answer of that revealed value): the model learns the env MECHANICS (probe -> reveal a known
value; answer the just-revealed value) but NOT the probe-choice POLICY, so greedy / log-prob decoding
carries no learned preference over which key to probe (mirroring the ring's random-transition training).
Whether an arm SOLVES the task therefore measures whether its scorer chooses to probe the target -- the
load-bearing epistemic decision. The prediction (design doc): the EFE rollout fails, because its
information-gain term is identically zero (sigma-gate FAILED, point belief) and cannot value the probe.

Pure environment + data + a closed-loop runner with the arms; the orchestration lives in the experiment
script. Decode is the linear path (use_prior_bank=False), matching the ring operating point.
"""
import time
from typing import Dict, Optional, Tuple

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

# ---- sealed vocabulary layout (design doc) ----
N_KEYS = 4                                  # table keys k_0..k_{N_KEYS-1}
N_VALS = 8                                  # value symbols v_0..v_{N_VALS-1}
VAL0   = 0                                  # values occupy ids 0..N_VALS-1
KEY0   = N_VALS                             # keys occupy ids N_VALS..N_VALS+N_KEYS-1
MASK   = N_VALS + N_KEYS                    # masked-value placeholder
SEP    = MASK + 1
QUERY  = SEP + 1
EOS    = QUERY + 1
PROBE0 = EOS + 1                            # PROBE_k_i = PROBE0 + i  (i = 0..N_KEYS-1)
ANSWER0 = PROBE0 + N_KEYS                   # ANSWER_v_j = ANSWER0 + j (j = 0..N_VALS-1)
V       = ANSWER0 + N_VALS                  # vocabulary size (= 2*N_VALS + 2*N_KEYS + 4)

CTX_LEN = 3 * N_KEYS + 2                    # "k v SEP" per key, then "QUERY k_t"
MAX_LEN = CTX_LEN + 8                       # context + a few action/reveal tokens (budget headroom)


def val_token(j: 'int | torch.Tensor') -> 'int | torch.Tensor':
    return VAL0 + j


def key_token(i: 'int | torch.Tensor') -> 'int | torch.Tensor':
    return KEY0 + i


def probe_token(i: 'int | torch.Tensor') -> 'int | torch.Tensor':
    return PROBE0 + i


def answer_token(j: 'int | torch.Tensor') -> 'int | torch.Tensor':
    return ANSWER0 + j


def probe_actions(device: Optional[torch.device] = None) -> torch.Tensor:
    """(N_KEYS,) the PROBE_k_i action token ids."""
    return torch.arange(PROBE0, PROBE0 + N_KEYS, device=device)


def answer_actions(device: Optional[torch.device] = None) -> torch.Tensor:
    """(N_VALS,) the ANSWER_v_j action token ids."""
    return torch.arange(ANSWER0, ANSWER0 + N_VALS, device=device)


def all_actions(device: Optional[torch.device] = None) -> torch.Tensor:
    """(N_KEYS + N_VALS,) the full single-token action menu (probes then answers)."""
    return torch.arange(PROBE0, V, device=device)


def render_context(
    vals:    torch.Tensor,           # (B, N_KEYS) value index per key
    target:  torch.Tensor,           # (B,) target key index

    *,
    reveal_target: bool = False,     # False -> target value MASKed (the task); True -> shown (diagnostics)
) -> torch.Tensor:                   # (B, CTX_LEN) "k_0 v SEP ... k_{nk-1} v QUERY k_t"
    r"""Render the table (target value masked) followed by the query. The non-target keys show their
    value; the target key shows MASK, so vals[target] is unknowable from the context."""
    B = vals.shape[0]
    device = vals.device
    col = lambda v: torch.full((B,), v, dtype=torch.long, device=device)
    parts = []
    for k in range(N_KEYS):
        parts.append(col(key_token(k)))                              # k_i
        v_k = val_token(vals[:, k])                                  # its value token
        if not reveal_target:
            masked = torch.where(target == k, torch.full_like(v_k, MASK), v_k)
            parts.append(masked)
        else:
            parts.append(v_k)
        parts.append(col(SEP))
    parts.append(col(QUERY))
    parts.append(key_token(target))                                 # QUERY k_t
    return torch.stack(parts, dim=1)                                # (B, 3*N_KEYS + 2)


def sample_episode(
    batch_size: int,

    *,
    generator:  Optional[torch.Generator] = None,
    device:     Optional[torch.device]    = None,
) -> 'Tuple[torch.Tensor, torch.Tensor]':
    r"""Sample an episode batch: ``vals`` (B, N_KEYS) the value of each key (values may repeat) and
    ``target`` (B,) the queried key. The answer to beat is ``vals[b, target[b]]``."""
    vals = torch.randint(0, N_VALS, (batch_size, N_KEYS), generator=generator, device=device)
    target = torch.randint(0, N_KEYS, (batch_size,), generator=generator, device=device)
    return vals, target


def target_value(vals: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B,) the value index of the target key, vals[b, target[b]]."""
    return vals.gather(1, target.unsqueeze(1)).squeeze(1)


def sample_batch(
    batch_size: int,

    *,
    generator:  Optional[torch.Generator] = None,
    device:     Optional[torch.device]    = None,
) -> 'Tuple[torch.Tensor, torch.Tensor]':
    r"""A teacher-forcing training batch on RANDOM-action trajectories
    ``[context] PROBE_k_r v_{vals[r]} ANSWER_v_{vals[r]} EOS`` with r drawn uniformly (NOT the target):
    the model learns the mechanics (probe a key -> its value is revealed; answer the just-revealed
    value) without a probe-choice policy. The targets at the probe position teach the copy of a KNOWN
    value (r != target) and stay unknowable for r == target (masked); the load-bearing target is at the
    revealed-value position (answer the revealed value). Returns (tokens, targets) of shape (B, CTX_LEN+4)."""
    vals, target = sample_episode(batch_size, generator=generator, device=device)
    r = torch.randint(0, N_KEYS, (batch_size,), generator=generator, device=device)   # random probed key
    v_r = vals.gather(1, r.unsqueeze(1)).squeeze(1)                                    # its value index
    ctx = render_context(vals, target)                                                # (B, CTX_LEN)
    traj = torch.stack([probe_token(r), val_token(v_r), answer_token(v_r),
                        torch.full_like(r, EOS)], dim=1)                               # (B, 4)
    tokens = torch.cat([ctx, traj], dim=1)                                            # (B, CTX_LEN+4)
    targets = torch.empty_like(tokens)
    targets[:, :-1] = tokens[:, 1:]
    targets[:, -1] = -100                                                             # no target after EOS
    return tokens, targets


@torch.no_grad()
def predictive_adequacy(
    model:      VFEModel,

    *,
    n:          int = 4096,
    generator:  Optional[torch.Generator] = None,
) -> float:
    r"""Answer-mechanics accuracy: given a probed key and its revealed value teacher-forced, does the
    model answer that value? Fraction with argmax p(o | context PROBE_k_r v_{vals[r]}) == ANSWER_v_{vals[r]}.
    This admits a model that has learned the answer protocol; whether the SCORER chooses to probe the
    target is the open question the experiment tests, NOT this gate."""
    device = next(model.parameters()).device
    vals, target = sample_episode(n, generator=generator)
    r = torch.randint(0, N_KEYS, (n,), generator=generator)
    vals, target, r = vals.to(device), target.to(device), r.to(device)
    v_r = vals.gather(1, r.unsqueeze(1)).squeeze(1)
    ctx = render_context(vals, target)
    prefix = torch.cat([ctx, probe_token(r).unsqueeze(1), val_token(v_r).unsqueeze(1)], dim=1)
    logits = model(prefix)                                          # (B, L, V)
    pred = logits[:, -1, :].argmax(dim=-1)                          # predicted token after the revealed value
    return float((pred == answer_token(v_r)).float().mean())


def train_checkpoint(
    *,
    seed:        int,
    steps:       int   = 6000,
    batch_size:  int   = 256,
    lr:          float = 3e-3,
    embed_dim:   int   = 20,
    n_heads:     int   = 2,
    n_layers:    int   = 1,
    n_e_steps:   int   = 1,
    log_every:   int   = 0,
    device:      Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
) -> 'Tuple[VFEModel, float]':
    r"""Train one masked-retrieval checkpoint (linear decode use_prior_bank=False, cache-supported
    defaults n_layers=1/n_e_steps=1 so the H>=2 EFE rollout uses the belief cache) for a fixed step
    budget; returns the model and its answer-mechanics adequacy. Plain AdamW next-token training on the
    random-action trajectories; runs on CUDA when available."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    cfg_kw = dict(
        vocab_size=V, embed_dim=embed_dim, n_heads=n_heads, max_seq_len=MAX_LEN,
        n_layers=n_layers, n_e_steps=n_e_steps, e_q_mu_lr=0.05, e_phi_lr=0.0,
        use_prior_bank=False, use_head_mixer=False,
    )
    if cfg_overrides:
        cfg_kw.update(cfg_overrides)
    model = VFEModel(VFE3Config(**cfg_kw)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed + 10_000)             # CPU generator -> device-independent data
    model.train()
    t_start = time.time()
    for step in range(steps):
        tokens, targets = sample_batch(batch_size, generator=gen)
        _, loss, _ = model(tokens.to(device), targets.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            elapsed = max(time.time() - t_start, 1e-9)
            rate = (step + 1) / elapsed
            print(f"  [seed {seed}] step {step + 1}/{steps}  loss={float(loss.detach()):.4f}  "
                  f"{rate:.1f} steps/s  ETA {(steps - step - 1) / rate:5.0f}s", flush=True)
    model.eval()
    adequacy = predictive_adequacy(model, generator=torch.Generator().manual_seed(seed + 20_000))
    return model, adequacy
