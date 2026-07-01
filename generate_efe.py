r"""Click-to-run: generate from a trained checkpoint with the EFE policy scorer on/off.

Edit CONFIG below, then run (no CLI args). It loads one of YOUR checkpoints, encodes a prompt with the
matching tokenizer (gpt2 for wikitext-103), and autoregressively generates -- once with the base model
(policy_mode='none') and, if you set a policy_mode, once more with the EFE reranker -- and prints both
so you can see the effect directly.

WHAT THIS CAN AND CANNOT DO ON WIKITEXT (read before expecting a win):
  * The scorer is INFERENCE-TIME, no-grad, default-off. It NEVER touches training; your production
    training runs are unaffected and need no change. This script only changes DECODING.
  * Through generate() the only allowed preference is 'flat' (open LM has no per-episode goal). With the
    default score_terms ('risk','ambiguity') the flat score is the constant log V, so efe_one_step
    falls back to BASE GREEDY -- a no-op. The one non-trivial generate-time use is the confidence
    reranker: policy_score_terms=('ambiguity',), which prefers tokens whose continuation the model is
    most confident about. It is a decoding heuristic, not epistemic active inference; effect on quality
    is unvalidated (see docs/research/active-inference/2026-06-29-v3-active-inference-closeout.md).
  * Goal-steering (policy_preference='task') is NOT available here -- config rejects it in generate()
    because it needs a goal token; drive the scorer directly through a harness (cf. ring_task.py).
  * efe_rollout (horizon>1) is NOT wired into generate() either (its candidate menu is single-token);
    use a harness. This script supports policy_mode in {none, efe_one_step, logprob_control}.

Run on the GPU (the iterative E-step is slow on CPU); it auto-uses CUDA when available.
"""
from dataclasses import fields

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

# ---------------------------------- edit me ----------------------------------
CONFIG = dict(
    # --- checkpoint (a resumable step_<N>.pt embeds its config and is self-contained) ---
    checkpoint   = "vfe3_runs/162.08_wikitext-103_K20_block_glk_s54/checkpoints/step_15000.pt",
    config_from  = None,        # if `checkpoint` is a pure state_dict (best_model.pt), point this at a
                                # sibling step_<N>.pt to borrow its config; else leave None
    dataset      = "wikitext-103",   # selects the tokenizer (gpt2); used to decode ids -> text

    # --- prompt + generation ---
    prompt          = "The history of the city",
    max_new_tokens  = 60,
    greedy          = True,     # True -> argmax / argmax-of-policy-posterior; deterministic

    # --- EFE policy scorer (the "active inference" knobs) ---
    policy_mode        = "efe_one_step",        # none | efe_one_step | logprob_control
    policy_preference  = "flat",                # generate() allows only 'flat'
    policy_score_terms = ("ambiguity",),        # ('risk','ambiguity') = no-op on LM; ('ambiguity',) = confidence reranker
    policy_top_k       = 8,                     # candidate menu width Kp
    policy_precision   = 1.0,                   # gamma in softmax(-gamma * G)
    policy_horizon     = 1,                     # must be 1 for efe_one_step in generate()

    device       = None,        # None -> cuda if available else cpu
)
# -----------------------------------------------------------------------------

_POLICY_FIELDS = ("policy_mode", "policy_preference", "policy_score_terms",
                  "policy_top_k", "policy_precision", "policy_horizon")


def _load_checkpoint(cfg: dict):
    """Return (config_dict, state_dict) from a step_<N>.pt (has 'config'+'model_state') or a pure
    state_dict (best_model.pt) plus an optional `config_from` step checkpoint for the config."""
    obj = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)  # your own trusted ckpt
    if isinstance(obj, dict) and "model_state" in obj:                 # resumable step_<N>.pt
        return dict(obj["config"]), obj["model_state"]
    # pure state_dict (best_model.pt): borrow the config from a sibling step checkpoint
    if cfg["config_from"] is None:
        raise ValueError(
            f"{cfg['checkpoint']} looks like a pure state_dict with no embedded config; set "
            f"CONFIG['config_from'] to a step_<N>.pt from the same run to supply the architecture.")
    src = torch.load(cfg["config_from"], map_location="cpu", weights_only=False)
    return dict(src["config"]), obj


def _build_model(config_dict: dict, state_dict, *, policy_overrides: dict, device: str) -> VFEModel:
    """Rebuild the model at the checkpoint's architecture with the policy fields overridden, then load
    the weights. The scorer adds no parameters, so the state_dict matches regardless of policy_mode."""
    valid = {f.name for f in fields(VFE3Config)}
    # audit F12 (2026-07-01): dropping genuine legacy fields is the intended migration, but do it
    # LOUDLY -- a renamed/removed field silently reverting to the current default could change the
    # reconstructed architecture with no notice. Warn, never raise (older checkpoints must load).
    import warnings
    dropped = sorted(set(config_dict) - valid)
    if dropped:
        warnings.warn(
            f"generate_efe: checkpoint config has {len(dropped)} field(s) unknown to the current "
            f"VFE3Config, dropping them (behavior falls back to defaults): {dropped}",
            UserWarning, stacklevel=2,
        )
    cfg_dict = {k: v for k, v in config_dict.items() if k in valid}    # drop any stale/unknown keys
    cfg_dict.update(policy_overrides)                                  # VFE3Config.__post_init__ validates the combo
    model = VFEModel(VFE3Config(**cfg_dict)).to(device)
    model.load_state_dict(state_dict)                                  # strict: scorer is param-free
    model.eval()
    return model


def _generate(model: VFEModel, prompt_ids: torch.Tensor, cfg: dict) -> torch.Tensor:
    return model.generate(prompt_ids, cfg["max_new_tokens"], greedy=cfg["greedy"])


def main():
    cfg = CONFIG
    device = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: running on CPU; the iterative E-step makes generation very slow. Use the GPU.")

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")                                # wikitext-103 tokenizer
    prompt_ids = torch.tensor([enc.encode(cfg["prompt"])], dtype=torch.long, device=device)

    config_dict, state_dict = _load_checkpoint(cfg)
    print(f"checkpoint: {cfg['checkpoint']}")
    print(f"arch: embed_dim={config_dict.get('embed_dim')} n_layers={config_dict.get('n_layers')} "
          f"n_e_steps={config_dict.get('n_e_steps')} use_prior_bank={config_dict.get('use_prior_bank')}")
    print(f"prompt: {cfg['prompt']!r}  ({prompt_ids.shape[1]} tokens)\n")

    # 1) base model (policy off) -- always shown as the reference
    base = _build_model(config_dict, state_dict, policy_overrides={"policy_mode": "none"}, device=device)
    base_out = _generate(base, prompt_ids, cfg)
    print("=== BASE (policy_mode='none') ===")
    print(enc.decode([int(t) for t in base_out[0].tolist()]), "\n")

    # 2) policy-reranked, only if you turned it on
    if cfg["policy_mode"] != "none":
        overrides = {k: cfg[k] for k in _POLICY_FIELDS}
        pol = _build_model(config_dict, state_dict, policy_overrides=overrides, device=device)
        pol_out = _generate(pol, prompt_ids, cfg)
        print(f"=== POLICY (policy_mode={cfg['policy_mode']!r}, "
              f"score_terms={cfg['policy_score_terms']}, top_k={cfg['policy_top_k']}, "
              f"gamma={cfg['policy_precision']}) ===")
        print(enc.decode([int(t) for t in pol_out[0].tolist()]), "\n")
        same = torch.equal(base_out, pol_out)
        print(f"identical to base? {same}"
              + ("  (expected for flat + default score_terms: the score is constant -> base greedy)"
                 if same else "  (the reranker changed the continuation)"))


if __name__ == "__main__":
    main()
