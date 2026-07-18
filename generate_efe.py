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
  * efe_rollout (horizon>1) IS reachable through generate() (audit PB-05): it builds a bounded H-step
    beam candidate menu and commits the first action of the selected policy. It REQUIRES a
    cache-supported checkpoint config (vfe3/inference/belief_cache.py::cache_supported) and
    policy_horizon>1; on an unsupported config the scorer fails closed rather than paying the dishonest
    full recompute. This script supports policy_mode in {none, efe_one_step, logprob_control, efe_rollout}.
  * policy_ambiguity_mode selects the EFE ambiguity estimator (registry key, exposed in CONFIG below).
    The default 'likelihood_entropy' is the sigma-free arm actually used in production. 'sigma_mc' has
    an executable antithetic-Monte-Carlo estimator (audit PB-06) but stays GATE-CLOSED: setting
    policy_sigma_ambiguity_validated / policy_sigma_gate_artifact / policy_sigma_mc_samples does NOT by
    itself unlock it -- construction still fails closed because the shipped sigma-gate preregistration
    resolves the live specification identity to FAIL (no matching empirical PASS record exists yet).

Run on the GPU (the iterative E-step is slow on CPU); it auto-uses CUDA when available.
"""
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch

from vfe3.config import VFE3Config, config_from_serialized, migrate_serialized_config
from vfe3.data.datasets import tiktoken_encoding_name
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import _write_json_atomic, semantic_config_fingerprint
from vfe3.runtime import deterministic_state

# ---------------------------------- edit me ----------------------------------
CONFIG = dict(
    # --- checkpoint (current best_model.pt and resumable step_<N>.pt files are self-contained) ---
    checkpoint   = "",          # required: set this to the checkpoint you intend to generate from
    config_from  = None,        # legacy raw state_dict only: must bind the identical weights + config
    dataset      = "wikitext-103",   # selects the tokenizer (gpt2); used to decode ids -> text

    # --- prompt + generation ---
    prompt          = "A man sat",
    max_new_tokens  = 60,
    greedy          = False,     # True -> argmax / argmax-of-policy-posterior; deterministic
    generation_seed = 6,         # explicit stochastic-decoding seed (also persisted below)
    output_path     = "efe_generation.json",

    # --- EFE policy scorer (the "active inference" knobs) ---
    policy_mode        = "efe_one_step",        # none | efe_one_step | logprob_control | efe_rollout (cache-supported cfg, horizon>1)
    policy_preference  = "flat",                # generate() allows only 'flat'
    policy_score_terms = ("ambiguity",),        # ('risk','ambiguity') = no-op on LM; ('ambiguity',) = confidence reranker
    policy_top_k       = 8,                     # candidate menu width Kp
    policy_precision   = 1,                   # gamma in softmax(-gamma * G)
    policy_horizon     = 1,                     # 1 for efe_one_step; >1 for efe_rollout (needs a cache-supported cfg)
    policy_ambiguity_mode            = "likelihood_entropy",  # ambiguity registry key; 'sigma_mc' is gate-closed (PB-06)
    policy_sigma_mc_samples          = 16,                    # sealed MC sample count for 'sigma_mc'; inert at the default ambiguity
    policy_sigma_ambiguity_validated = False,                 # PB-06 precondition flag; True alone never unlocks 'sigma_mc'
    policy_sigma_gate_artifact       = None,                  # path to a PASS sigma-gate record; required (not sufficient) for 'sigma_mc'

    device       = None,        # None -> cuda if available else cpu
)
# -----------------------------------------------------------------------------

_POLICY_FIELDS = ("policy_mode", "policy_preference", "policy_score_terms",
                  "policy_top_k", "policy_precision", "policy_horizon",
                  "policy_ambiguity_mode", "policy_sigma_mc_samples",
                  "policy_sigma_ambiguity_validated", "policy_sigma_gate_artifact")


def _validated_generation_paths(
    cfg: Mapping[str, Any],
) -> Tuple[Path, Optional[Path], Path]:
    """Resolve input/output paths and reject any destructive publication alias."""
    checkpoint = cfg.get("checkpoint")
    if not isinstance(checkpoint, (str, Path)) or not str(checkpoint):
        raise ValueError("set CONFIG['checkpoint'] to an existing checkpoint before generation")
    output = cfg.get("output_path")
    if not isinstance(output, str) or not output:
        raise ValueError("output_path must be a non-empty path string")
    config_from = cfg.get("config_from")
    if config_from is not None and (
        not isinstance(config_from, (str, Path)) or not str(config_from)
    ):
        raise ValueError("config_from must be None or a non-empty path")

    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=False)
    config_from_path = (
        Path(config_from).expanduser().resolve(strict=False)
        if config_from is not None else None
    )
    output_path = Path(output).expanduser().resolve(strict=False)
    input_paths = {checkpoint_path}
    if config_from_path is not None:
        input_paths.add(config_from_path)
    if output_path in input_paths:
        raise ValueError("output_path must not alias checkpoint or config_from input")
    return checkpoint_path, config_from_path, output_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _generation_code_identity(root: Optional[Path] = None) -> str:
    """Hash the generation entry point and every executable package Python source."""
    base = Path(__file__).resolve().parent if root is None else Path(root).resolve()
    sources = [base / "generate_efe.py"]
    sources.extend(sorted(
        path for path in (base / "vfe3").rglob("*.py")
        if "__pycache__" not in path.parts
    ))
    digest = hashlib.sha256()
    for path in sources:
        relative = path.relative_to(base).as_posix().encode("utf-8")
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        for payload in (relative, content):
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def _runtime_identity(device: 'str | torch.device') -> dict:
    """Return framework, determinism, and canonical execution-device provenance."""
    resolved = torch.device(device)
    index = resolved.index
    name = None
    if resolved.type == "cuda":
        index = torch.cuda.current_device() if index is None else index
        name = torch.cuda.get_device_name(index)
    return {
        "determinism": deterministic_state(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device": {"type": resolved.type, "index": index, "name": name},
    }


def _model_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, schemas, and bytes in a stable key order."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"model_state entry {name!r} is not a tensor")
        owned = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(owned.dtype), "shape": list(owned.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(owned.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _bound_config(
    payload: Mapping[str, Any],

    *,
    source: object,
) -> Tuple[dict, str]:
    """Return a mapping config and its verified or computed semantic fingerprint."""
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"checkpoint {source} has no embedded config mapping")
    config_dict = dict(config)
    computed = semantic_config_fingerprint(config_dict)
    stored = payload.get("config_fingerprint")
    if stored is not None and stored != computed:
        raise ValueError(f"checkpoint {source} has a config fingerprint mismatch")
    migration = migrate_serialized_config(
        config_dict,
        source=f"checkpoint {source}",
        strict_unknown=True,
    )
    return asdict(migration.config), computed


def _state_dicts_equal(
    left:  Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Whether two loaded state dictionaries contain exactly equal tensor values."""
    if set(left) != set(right):
        return False
    return all(
        isinstance(left[key], torch.Tensor)
        and isinstance(right[key], torch.Tensor)
        and torch.equal(left[key], right[key])
        for key in left
    )


def _load_checkpoint(
    cfg: Mapping[str, Any],

    *,
    checkpoint_snapshot: Optional[bytes] = None,
    config_from_snapshot: Optional[bytes] = None,
) -> Tuple[dict, Mapping[str, torch.Tensor]]:
    """Load a self-bound checkpoint or a legacy state explicitly bound to identical weights."""
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("set CONFIG['checkpoint'] to an existing checkpoint before generation")
    checkpoint_source = (
        io.BytesIO(checkpoint_snapshot) if checkpoint_snapshot is not None else checkpoint)
    obj = torch.load(checkpoint_source, map_location="cpu", weights_only=True)
    if not isinstance(obj, Mapping) or not obj:
        raise ValueError(f"checkpoint {checkpoint} is empty or malformed")

    if "model_state" in obj:
        config_dict, fingerprint = _bound_config(obj, source=checkpoint)
        config_from = cfg.get("config_from")
        if config_from:
            config_source = (
                io.BytesIO(config_from_snapshot)
                if config_from_snapshot is not None else config_from)
            source_obj = torch.load(config_source, map_location="cpu", weights_only=True)
            if not isinstance(source_obj, Mapping):
                raise ValueError(f"config_from checkpoint {config_from} is malformed")
            _, source_fingerprint = _bound_config(source_obj, source=config_from)
            if source_fingerprint != fingerprint:
                raise ValueError(
                    f"semantic config mismatch between {checkpoint} and {config_from}")
        state_dict = obj["model_state"]
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError(
                f"checkpoint {checkpoint} must contain a nonempty model_state mapping")
        return config_dict, state_dict

    config_from = cfg.get("config_from")
    if not config_from:
        raise ValueError(
            f"{checkpoint} is a legacy pure state_dict with no bound config; set "
            "CONFIG['config_from'] to a checkpoint containing the identical model_state and config.")
    config_source = (
        io.BytesIO(config_from_snapshot) if config_from_snapshot is not None else config_from)
    source_obj = torch.load(config_source, map_location="cpu", weights_only=True)
    if not isinstance(source_obj, Mapping):
        raise ValueError(f"config_from checkpoint {config_from} is malformed")
    config_dict, _ = _bound_config(source_obj, source=config_from)
    source_state = source_obj.get("model_state")
    if not isinstance(source_state, Mapping) or not _state_dicts_equal(obj, source_state):
        raise ValueError(
            f"cannot bind legacy state_dict {checkpoint}: its weights do not exactly match "
            f"config_from checkpoint {config_from}")
    return config_dict, obj


def _tokenizer_for_dataset(
    dataset: str,

    *,
    vocab_size: int,
) -> Any:
    """Load the cache-compatible tokenizer and require its vocabulary to match the model."""
    import tiktoken
    encoding_name = tiktoken_encoding_name(dataset)
    enc = tiktoken.get_encoding(encoding_name)
    if enc.n_vocab != vocab_size:
        raise ValueError(
            f"dataset {dataset!r} tokenizer vocabulary has {enc.n_vocab} ids, but the "
            f"checkpoint config has vocab_size={vocab_size}")
    return enc


def _build_model(
    config_dict: Mapping[str, Any],
    state_dict:  Mapping[str, torch.Tensor],

    *,
    policy_overrides: Mapping[str, Any],
    device:           str,
) -> VFEModel:
    """Rebuild the model at the checkpoint's architecture with the policy fields overridden, then load
    the weights. The scorer adds no parameters, so the state_dict matches regardless of policy_mode."""
    cfg_dict = dict(config_dict)
    cfg_dict.update(policy_overrides)
    cfg = config_from_serialized(cfg_dict, source="generate_efe checkpoint config")
    model = VFEModel(cfg).to(device)
    model.load_state_dict(state_dict, strict=True)                      # scorer is parameter-free
    model.eval()
    return model


def _generate(
    prompt_ids: torch.Tensor,
    model:      VFEModel,
    cfg:        Mapping[str, Any],
) -> torch.Tensor:
    return model.generate(prompt_ids, cfg["max_new_tokens"], greedy=cfg["greedy"])


def _run_generation_arms(
    prompt_ids:  torch.Tensor,
    config_dict: Mapping[str, Any],
    state_dict:  Mapping[str, torch.Tensor],
    cfg:         Mapping[str, Any],

    *,
    device: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build both arms, then generate from paired CPU and all-CUDA RNG states."""
    base_model = _build_model(
        config_dict, state_dict, policy_overrides={"policy_mode": "none"}, device=device,
    )
    policy_model = None
    if cfg["policy_mode"] != "none":
        overrides = {key: cfg[key] for key in _POLICY_FIELDS}
        policy_model = _build_model(
            config_dict, state_dict, policy_overrides=overrides, device=device,
        )

    generation_seed = cfg.get("generation_seed")
    if (isinstance(generation_seed, bool) or not isinstance(generation_seed, int)
            or generation_seed < 0):
        raise ValueError(
            f"generation_seed must be a non-negative integer, got {generation_seed!r}"
        )
    # torch.manual_seed sets the CPU generator and every CUDA generator.  Apply it only after both
    # models exist so model-construction RNG consumption cannot move either generation arm.
    torch.manual_seed(generation_seed)
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    base_out = _generate(prompt_ids, base_model, cfg)
    if policy_model is None:
        return base_out, None
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    policy_out = _generate(prompt_ids, policy_model, cfg)
    return base_out, policy_out


def main() -> None:
    cfg = CONFIG
    checkpoint_path, config_from_path, output_path = _validated_generation_paths(cfg)
    checkpoint_snapshot = checkpoint_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_snapshot).hexdigest()
    config_from_snapshot = (
        config_from_path.read_bytes() if config_from_path is not None else None)
    config_from_sha256 = (
        hashlib.sha256(config_from_snapshot).hexdigest()
        if config_from_snapshot is not None else None)
    code_identity_sha256 = _generation_code_identity()
    device = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: running on CPU; the iterative E-step makes generation very slow. Use the GPU.")

    config_dict, state_dict = _load_checkpoint(
        cfg,
        checkpoint_snapshot=checkpoint_snapshot,
        config_from_snapshot=config_from_snapshot,
    )
    if (_sha256_file(checkpoint_path) != checkpoint_sha256
            or (config_from_path is not None
                and _sha256_file(config_from_path) != config_from_sha256)):
        raise RuntimeError("generation input changed during checkpoint loading")
    enc = _tokenizer_for_dataset(cfg["dataset"], vocab_size=config_dict.get("vocab_size"))
    prompt_ids = torch.tensor([enc.encode(cfg["prompt"])], dtype=torch.long, device=device)
    print(f"checkpoint: {cfg['checkpoint']}")
    print(f"arch: embed_dim={config_dict.get('embed_dim')} n_layers={config_dict.get('n_layers')} "
          f"n_e_steps={config_dict.get('n_e_steps')} use_prior_bank={config_dict.get('use_prior_bank')}")
    print(f"prompt: {cfg['prompt']!r}  ({prompt_ids.shape[1]} tokens)\n")

    base_out, pol_out = _run_generation_arms(
        prompt_ids, config_dict, state_dict, cfg, device=device,
    )
    if (_sha256_file(checkpoint_path) != checkpoint_sha256
            or (config_from_path is not None
                and _sha256_file(config_from_path) != config_from_sha256)):
        raise RuntimeError("generation input changed during generation; refusing publication")
    print("=== BASE (policy_mode='none') ===")
    print(enc.decode([int(t) for t in base_out[0].tolist()]), "\n")

    # 2) policy-reranked, only if you turned it on
    if pol_out is not None:
        print(f"=== POLICY (policy_mode={cfg['policy_mode']!r}, "
              f"score_terms={cfg['policy_score_terms']}, top_k={cfg['policy_top_k']}, "
              f"gamma={cfg['policy_precision']}) ===")
        print(enc.decode([int(t) for t in pol_out[0].tolist()]), "\n")
        same = torch.equal(base_out, pol_out)
        print(f"identical to base? {same}"
              + ("  (expected for flat + default score_terms: the score is constant -> base greedy)"
                 if same else "  (the reranker changed the continuation)"))

    policy_contract = {
        key: (list(cfg[key]) if key == "policy_score_terms" else cfg[key])
        for key in _POLICY_FIELDS
    }
    record = {
        "schema_version": 3,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "config_from": str(config_from_path) if config_from_path is not None else None,
        "config_from_sha256": config_from_sha256,
        "model_state_sha256": _model_state_sha256(state_dict),
        "config_fingerprint": semantic_config_fingerprint(config_dict),
        "code_identity_sha256": code_identity_sha256,
        "runtime_state": _runtime_identity(device),
        "generation_seed": int(cfg["generation_seed"]),
        "greedy": bool(cfg["greedy"]),
        "policy_mode": str(cfg["policy_mode"]),
        "policy_score_terms": list(cfg["policy_score_terms"]),
        "generation_contract": {
            "dataset": str(cfg["dataset"]),
            "prompt": str(cfg["prompt"]),
            "prompt_token_ids": prompt_ids.detach().cpu().tolist(),
            "max_new_tokens": int(cfg["max_new_tokens"]),
            "generation_seed": int(cfg["generation_seed"]),
            "greedy": bool(cfg["greedy"]),
            "device": str(device),
            "policy": policy_contract,
        },
        "outputs": {
            "base_token_ids": base_out.detach().cpu().tolist(),
            "policy_token_ids": (
                pol_out.detach().cpu().tolist() if pol_out is not None else None
            ),
        },
    }
    _write_json_atomic(output_path, record)
    print(f"generation record: {output_path}")


if __name__ == "__main__":
    main()
