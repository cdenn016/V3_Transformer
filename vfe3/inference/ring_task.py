r"""The controlled closed-loop ring goal-steering ("cursor control") synthetic task for the v1 EFE
policy experiment (spec docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md,
Section 4.1).

A fully observed, deterministic environment whose pragmatic payoff is realizable at a one-step
horizon. The vocabulary (V = 32) partitions into m = 16 ring state symbols q_0..q_15 (ids 0..15),
three action tokens DEC / STAY / INC (16/17/18) with deltas -1 / 0 / +1, four control tokens
GOAL / SEP / CUR / EOS (19..22), and reserved filler (23..31). An episode draws an initial state s_0
and a goal g != s_0 uniformly on the ring and presents the context ``GOAL q_g SEP CUR q_{s_t}``. The
agent emits one action; the environment applies s_{t+1} = (s_t + delta(a)) mod m and re-renders. An
episode is correct iff the state equals the goal at the budget's end (T_ep = 10).

Training renders the exact prefix the experiment scores: ``GOAL q_g SEP CUR q_s a q_{s'}`` (length 7),
so teacher-forced next-token prediction at the action position learns the transition q_{s'} | (s, a),
the precondition any model-based planner needs (predictive-adequacy gate, Section 4.5).

This module is pure environment + data + a batched closed-loop runner; the orchestration (three-seed
training, the arm matrix, paired statistics, go/no-go) lives in the experiment script.
"""
import math
import time
from typing import Dict, Optional, Tuple

import torch

from vfe3.config import VFE3Config
from vfe3.inference.policy import get_policy, get_preference
from vfe3.model.model import VFEModel

# ---- sealed vocabulary layout (spec Section 4.1) ----
M       = 16                    # ring size (state symbols q_0..q_{M-1} are token ids 0..M-1)
V       = 32                    # vocabulary
DEC, STAY, INC = 16, 17, 18     # action tokens; deltas -1, 0, +1
GOAL, SEP, CUR, EOS = 19, 20, 21, 22
SEQ_LEN = 7                     # GOAL q_g SEP CUR q_s a q_s'
ACTION_TOKENS = (DEC, STAY, INC)


def action_delta_table() -> torch.Tensor:
    """(V,) ring delta per token: -1 for DEC, +1 for INC, 0 otherwise (non-action -> wasted STAY)."""
    d = torch.zeros(V, dtype=torch.long)
    d[DEC] = -1
    d[INC] = 1
    return d


def state_support() -> torch.Tensor:
    """(M,) the state token ids 0..M-1; the support of the task preference p_task."""
    return torch.arange(M)


def ring_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cyclic distance min(|a-b|, M-|a-b|) on the ring (token ids in 0..M-1)."""
    d = (a - b).abs()
    return torch.minimum(d, M - d)


def ring_preference(
    goals:  torch.Tensor,            # (B,) goal state ids

    *,
    beta_C:        float           = 5.0,  # preference precision (sealed constant)
    support_floor: Optional[float] = None, # None -> hard -inf; finite -> explicit soft floor
) -> torch.Tensor:                   # (B, V) log p_task(o)
    r"""The ring's instantiation of the spec's task preference p(o|C) = softmax(beta_C U_C) with the
    DISTANCE-GRADED utility U_C(o) = -ring_distance(o, g) on the M state tokens. ``support_floor=None``
    retains exact hard support (-inf on non-state tokens); a supplied finite value is explicitly
    reported as a finite-floor preference and normalized together with the state utilities.

    Correction to the spec's literal wording (Section 4.1): a pure peak on the goal symbol (uniform
    mass over all other states) carries NO one-step gradient at ring-distance > 1 -- both neighboring
    actions then land on non-goal states with equal preference mass, so greedy H=1 cannot tell which
    reduces the distance. The spec's own solvability argument ("the reachable state nearest the goal
    is closest to the goal-peaked preference") requires a distance-graded utility, which this provides;
    the sealed beta_C=5.0 is kept (its 0.90 goal-mass figure was the uniform-off-goal reading)."""
    if support_floor is not None and not math.isfinite(support_floor):
        raise ValueError(f"support_floor must be finite or None, got {support_floor}")
    states = state_support().to(goals.device)                        # (M,)
    dist = ring_distance(states.unsqueeze(0), goals.unsqueeze(1))     # (B, M)
    floor = float("-inf") if support_floor is None else support_floor
    U = torch.full((goals.shape[0], V), floor, device=goals.device)
    U[:, :M] = -beta_C * dist.float()
    return torch.log_softmax(U, dim=-1)


def transition(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """s' = (s + delta(a)) mod M, vectorized. ``actions`` are token ids; non-action tokens give delta 0."""
    delta = action_delta_table().to(states.device)[actions]
    return (states + delta) % M


def render_context(
    goals:  torch.Tensor,            # (B,) goal state ids
    states: torch.Tensor,            # (B,) current state ids
) -> torch.Tensor:                   # (B, 5) "GOAL q_g SEP CUR q_s"
    """Render the per-step context the experiment conditions on."""
    B = goals.shape[0]
    col = lambda v: torch.full((B,), v, dtype=torch.long, device=goals.device)
    return torch.stack([col(GOAL), goals, col(SEP), col(CUR), states], dim=1)


def sample_batch(
    batch_size: int,

    *,
    generator:  Optional[torch.Generator] = None,
    device:     Optional[torch.device]    = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""A training batch of rendered transitions ``GOAL q_g SEP CUR q_s a q_s'`` (B, 7) with the
    next-token targets (B, 7) (last position ignored, -100). g, s, a are drawn uniformly and
    independently, so the model learns the full transition table q_{s'} | (s, a) and learns that g
    does not affect the transition. The load-bearing target is at the action position (index 5)."""
    g = torch.randint(0, M, (batch_size,), generator=generator, device=device)
    s = torch.randint(0, M, (batch_size,), generator=generator, device=device)
    a_idx = torch.randint(0, 3, (batch_size,), generator=generator, device=device)
    a = torch.tensor(ACTION_TOKENS, device=device)[a_idx]
    s_next = transition(s, a)
    col = lambda v: torch.full((batch_size,), v, dtype=torch.long, device=device)
    tokens = torch.stack([col(GOAL), g, col(SEP), col(CUR), s, a, s_next], dim=1)  # (B, 7)
    targets = torch.empty_like(tokens)
    targets[:, :-1] = tokens[:, 1:]
    targets[:, -1] = -100                                   # no target after the final state token
    return tokens, targets


@torch.no_grad()
def predictive_adequacy(
    model:      VFEModel,

    *,
    n:          int = 4096,
    generator:  Optional[torch.Generator] = None,
) -> float:
    r"""Teacher-forced next-observation (transition) accuracy on held-out random transitions: the
    fraction for which argmax p(o | GOAL q_g SEP CUR q_s a) == q_{s'} (spec Section 4.5; gate >= 0.98).
    Read at the action position (index 5), exactly where the experiment reads q(o|pi)."""
    device = next(model.parameters()).device
    tokens, _ = sample_batch(n, generator=generator)        # CPU-reproducible data
    tokens = tokens.to(device)
    logits = model(tokens[:, :SEQ_LEN - 1])                 # prefix through the action token (B, 6, V)
    pred = logits[:, -1, :].argmax(dim=-1)                  # predicted next state at the action position
    truth = tokens[:, -1]                                   # the true q_{s'}
    return float((pred == truth).float().mean())


def train_ring_checkpoint(
    *,
    seed:        int,
    steps:       int   = 15000,
    batch_size:  int   = 256,
    lr:          float = 3e-3,
    embed_dim:   int   = 20,
    n_heads:     int   = 2,
    n_layers:    int   = 2,
    n_e_steps:   int   = 2,
    log_every:   int   = 0,                      # >0 -> print step/loss/rate/ETA every log_every steps
    device:      Optional[str] = None,           # None -> cuda if available else cpu (use the 5090)
    cfg_overrides: Optional[dict] = None,
) -> Tuple[VFEModel, float]:
    r"""Train one ring checkpoint at the operating-point architecture (embed_dim=20, linear decode
    use_prior_bank=False, use_head_mixer=False; spec Section 4.5) for a fixed step budget, returning
    the model and its final predictive adequacy. Plain AdamW next-token training on rendered
    transitions; no dev-selection knob (final checkpoint taken). Runs on CUDA when available (the
    iterative E-step is slow on CPU; train this on the GPU)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    cfg_kw = dict(
        vocab_size=V, embed_dim=embed_dim, n_heads=n_heads, max_seq_len=SEQ_LEN,
        n_layers=n_layers, n_e_steps=n_e_steps, e_q_mu_lr=0.05, e_phi_lr=0.02,
        use_prior_bank=False, use_head_mixer=False,
    )
    if cfg_overrides:
        cfg_kw.update(cfg_overrides)
    model = VFEModel(VFE3Config(**cfg_kw)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed + 10_000)      # CPU generator -> device-independent data
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
            eta = (steps - step - 1) / rate
            print(f"  [seed {seed}] step {step + 1}/{steps}  loss={float(loss.detach()):.4f}  "
                  f"{rate:.1f} steps/s  ETA {eta:5.0f}s", flush=True)
    model.eval()
    adequacy = predictive_adequacy(model, generator=torch.Generator().manual_seed(seed + 20_000))
    return model, adequacy


# ---- Phase 2 decoding baselines and lesion checks (spec Sections 4.3, 4.6) -------------------------
# v1-honest matrix: the matched-compute beam / best-of-N baselines and the length-normalized /
# argmax-confidence near-competitors reduce to existing arms at the H=1 single-decision horizon (no
# multi-step search to run; a single emitted token makes length-normalization a no-op; argmax-confidence
# is the greedy logprob arm), so they are deferred to the horizon phase (Phase 3) rather than run as
# vacuous duplicates -- see docs/research/active-inference/2026-06-28-phase2-scope-note.md. The
# genuinely v1-distinct standard decoders are the goal-free sampling strategies over the candidate menu
# below; none reads the goal preference, so on the ring all collapse to ~chance and are reported (not
# gated) to show that no standard decoder steers.
SAMPLING_BASELINES = ("temp_sample", "nucleus", "typical")


def closed_loop_causality_holds() -> bool:
    r"""The v1 closed-loop causality lesion check (spec Section 4.6): the committed action must
    measurably change the next observation. The deterministic ring transition guarantees it -- from
    every state the three actions DEC/STAY/INC land on three DISTINCT next states (s-1, s, s+1 mod M
    are pairwise distinct for M > 2) -- and this confirms it by construction over all M states."""
    states = state_support()
    nxt = torch.stack([transition(states, torch.full_like(states, a)) for a in ACTION_TOKENS], dim=1)
    return bool((nxt[:, 0] != nxt[:, 1]).all() and (nxt[:, 1] != nxt[:, 2]).all()
                and (nxt[:, 0] != nxt[:, 2]).all())


def _decode_menu(
    menu_logits: torch.Tensor,       # (B, |menu|) base logits over the candidate menu

    mode:        str,                 # 'temp_sample' | 'nucleus' | 'typical'

    *,
    temperature: float = 1.0,        # temp_sample: softmax(logits / T)
    top_p:       float = 0.9,        # nucleus / typical: retained probability mass
    generator:   Optional[torch.Generator] = None,
) -> torch.Tensor:                   # (B, 1) sampled menu index
    r"""Goal-free standard decoding strategies over the candidate menu (spec Section 4.3). Temperature
    sampling, nucleus (top-p), and locally-typical sampling (Meister et al. 2022). The draw is taken on
    CPU with the optional generator for device-independent reproducibility (mirrors the 'random' arm)."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature}")
    if not math.isfinite(top_p) or not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be finite and in (0, 1], got {top_p}")
    if not bool(torch.isfinite(menu_logits).all()):
        raise ValueError("menu_logits must contain only finite values")
    if mode == "temp_sample":
        probs = torch.softmax(menu_logits / temperature, dim=-1)
    elif mode in ("nucleus", "typical"):
        logp = torch.log_softmax(menu_logits, dim=-1)
        p = logp.exp()
        if mode == "nucleus":
            order = p.sort(dim=-1, descending=True)                       # most probable first
            sorted_p, sorted_idx = order.values, order.indices
        else:                                                            # locally-typical
            p_logp = torch.where(p > 0, p * logp, torch.zeros_like(p))
            H = -p_logp.sum(dim=-1, keepdim=True)                         # (B,1) entropy
            shift = (-logp - H).abs()                                     # deviation from expected info
            order = shift.sort(dim=-1)                                    # most typical (smallest) first
            sorted_idx = order.indices
            sorted_p = p.gather(-1, sorted_idx)
        cum = sorted_p.cumsum(dim=-1)
        keep = (cum - sorted_p) < top_p                                  # retain up to the mass threshold
        keep[..., 0] = True                                             # always keep the first
        kept = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
        probs = torch.zeros_like(p).scatter(-1, sorted_idx, kept)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown sampling baseline {mode!r}")
    row_mass = probs.sum(dim=-1)
    if not bool(torch.isfinite(probs).all()) or not bool((row_mass > 0).all()):
        raise ValueError("sampling probabilities must be finite with positive mass in every row")
    idx = torch.multinomial(probs.cpu(), 1, generator=generator)         # CPU draw (generator-safe)
    return idx.to(menu_logits.device)


@torch.no_grad()
def run_episodes(
    model:      VFEModel,
    goals:      torch.Tensor,        # (B,) goal state ids
    states:     torch.Tensor,        # (B,) initial state ids s_0

    policy_mode: str,                 # scorer: efe_one_step|logprob_control ; baseline: random|greedy_ref|temp_sample|nucleus|typical

    *,
    preference_key: str             = "task",   # p(o|C) registry key for the EFE arm
    score_terms:    Tuple[str, ...] = ("risk", "ambiguity"),
    gamma:          float           = 1.0,
    temperature:    float           = 1.0,      # temp_sample decoder temperature (Phase 2 baseline)
    top_p:          float           = 0.9,      # nucleus / typical retained mass (Phase 2 baselines)
    candidate_mode: str             = "actions",  # "actions" (the 3 control actions) | "top_k" (top-Kp tokens)
    top_k:          int             = 8,
    beta_C:         float           = 5.0,
    horizon:        int             = 1,
    budget:         int             = 10,       # T_ep
    generator:      Optional[torch.Generator] = None,   # for the stochastic arms (random / sampling)
) -> Dict[str, torch.Tensor]:
    r"""Run B closed-loop ring episodes in parallel under the given arm and return per-episode
    outcomes. Each step renders ``GOAL q_g SEP CUR q_s``, forms the candidate menu from the base logits
    (the pre-registered generator E, candidate prior = base softmax over the menu), selects an action
    through ``policy_mode``, and applies the DETERMINISTIC environment transition (never the model's
    predicted outcome; spec Section 2.2). Arms: ``efe_one_step``/``logprob_control`` are the scorer arms
    (argmax of the policy posterior); ``random`` is the uniform placebo; ``greedy_ref`` is the
    unmodified ``generate`` reference (argmax over the full vocab); ``temp_sample``/``nucleus``/
    ``typical`` are the goal-free sampling baselines (spec Section 4.3). Returns ``correct`` (B,) bool,
    ``steps_to_goal`` (B,), ``frac_at_goal`` (B,), and scalar ``mean_risk``/``mean_ambiguity``
    diagnostics for the scorer arms (0 elsewhere; spec Section 4.4 component attribution)."""
    device = next(model.parameters()).device                          # run on the model's device
    goals = goals.to(device)
    state = states.to(device)
    delta_tab = action_delta_table().to(device)
    B = states.shape[0]
    reached = torch.full((B,), budget, dtype=torch.long, device=device)   # step index of first arrival
    at_goal_steps = torch.zeros(B, dtype=torch.long, device=device)
    risk_sum = torch.zeros((), device=device)                         # scorer-arm component diagnostics
    amb_sum = torch.zeros((), device=device)
    n_dec = 0
    for t in range(budget):
        context = render_context(goals, state)                            # (B, 5)
        base_logits = model.forward(context)[:, -1, :]                    # (B, V)
        if policy_mode == "greedy_ref":
            action = base_logits.argmax(dim=-1)                           # unmodified generate (full vocab)
            state = (state + delta_tab[action]) % M
            now_at = state == goals
            reached = torch.where(now_at & (reached == budget),
                                  torch.full_like(reached, t + 1), reached)
            at_goal_steps = at_goal_steps + now_at.long()
            continue
        if candidate_mode == "actions":
            # The control-task policy space IS the action set. The top-Kp token generator admits
            # non-action tokens (e.g. CUR) whose diffuse OOD prediction scores LOWER than a genuine
            # step toward a far goal, so the policy would pick a junk token that maps to a wasted STAY.
            topk = torch.tensor(ACTION_TOKENS, device=device).unsqueeze(0).expand(B, -1)  # (B, 3)
        else:
            topk = base_logits.topk(top_k, dim=-1).indices               # (B, Kp) top-Kp tokens
        menu_logits = torch.gather(base_logits, 1, topk)                  # (B, |menu|)
        if policy_mode == "random":
            kp = topk.shape[1]                                            # actual menu width (not cfg top_k)
            rand = (torch.rand(B, kp, generator=generator) if generator is not None
                    else torch.rand(B, kp))                              # CPU draw (generator-safe)
            idx = rand.to(device).argmax(dim=-1, keepdim=True)           # uniform pick over the menu
        elif policy_mode in SAMPLING_BASELINES:
            idx = _decode_menu(menu_logits, policy_mode, temperature=temperature,
                               top_p=top_p, generator=generator)         # goal-free standard decoders
        else:
            candidates = topk.unsqueeze(-1)                              # (B, |menu|, 1)
            log_prior = torch.log_softmax(menu_logits, dim=-1)          # (B, |menu|) candidate prior E
            if preference_key == "task":
                support_floor = -beta_C * (M / 2 + 1)                     # explicit finite-floor arm
                pref = ring_preference(
                    goals, beta_C=beta_C, support_floor=support_floor)     # distance-graded p_task (B, V)
            elif preference_key == "flat":
                pref = get_preference("flat")(model.prior_bank, device=device)
            elif preference_key == "held_out_predictive":
                # the ring next-observation marginal is uniform over the M state tokens (the data
                # distribution): carries no per-episode goal, so it is the control arm (spec 2.3/4.3).
                # Finite floor on non-state tokens (not -inf) so the forward KL stays finite.
                U = torch.full((V,), -30.0, device=device)
                U[:M] = 0.0
                pref = torch.log_softmax(U, dim=-1)
            else:
                raise ValueError(f"unsupported preference_key for the ring task: {preference_key!r}")
            out = get_policy(policy_mode)(
                context, candidates, pref, model,
                gamma=gamma, horizon=horizon, score_terms=score_terms,
                log_prior=log_prior, base_logits=base_logits,
            )
            idx = out.policy_posterior.argmax(dim=-1, keepdim=True)       # greedy (deterministic eval)
            risk_sum = risk_sum + torch.gather(out.risk, 1, idx).sum()    # committed-action diagnostics
            amb_sum = amb_sum + torch.gather(out.ambiguity, 1, idx).sum()
            n_dec += B
        action = torch.gather(topk, 1, idx).squeeze(1)                   # (B,) committed action token
        state = (state + delta_tab[action]) % M                          # deterministic environment step
        now_at = state == goals
        reached = torch.where(now_at & (reached == budget),
                              torch.full_like(reached, t + 1), reached)
        at_goal_steps = at_goal_steps + now_at.long()
    correct = state == goals
    return {
        "correct": correct,
        "steps_to_goal": reached,
        "frac_at_goal": at_goal_steps.float() / budget,
        "mean_risk": (risk_sum / n_dec) if n_dec else torch.zeros((), device=device),
        "mean_ambiguity": (amb_sum / n_dec) if n_dec else torch.zeros((), device=device),
    }
