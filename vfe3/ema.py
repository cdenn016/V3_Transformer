r"""EMA / Polyak weight averaging for the learned tables (default-OFF; toggle ``use_ema``).

A passive exponential moving average of the model's trainable parameters (the prior/decode tables
and the few learned scalars). After each optimizer step the shadow is updated

    s_t = decay * s_{t-1} + (1 - decay) * theta_t,

i.e. the standard Polyak tail-average. The averaged point sits in a flatter region of the loss
surface and typically generalizes a little better than the last SGD iterate, at no change to the
free energy: EMA reads parameters, draws no RNG, and never touches ``.grad`` or the optimizer state,
so the SGD trajectory is identical whether the toggle is on or off.

Usage in the training loop: ``update`` after each optimizer step; ``store`` then ``copy_to`` to swap
the averaged weights in for evaluation/checkpointing, then ``restore`` to put the live SGD weights
back before the next step; a final ``copy_to`` leaves the trained model holding the averaged weights.
``state_dict``/``load_state_dict`` persist the shadow across a resume so the average is not silently
re-seeded from the resumed iterate.
"""

import warnings
import weakref
from typing import Dict, Set

import torch


class EMA:
    r"""Shadow average of a module's ``requires_grad`` parameters."""

    def __init__(
        self,
        model: torch.nn.Module,

        *,
        decay: float = 0.999,
    ) -> None:
        self.decay = float(decay)
        self._model_ref = weakref.ref(model)
        # Track only trainable params: frozen tensors (e.g. r_mu with learnable_r=False) never move,
        # so averaging them is a wasteful no-op. Keyed by the stable parameter name.
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _derived_parameter_names(model: torch.nn.Module) -> Set[str]:
        r"""Non-gradient parameters deterministically derived from averaged trainable tables."""
        cfg = getattr(model, "cfg", None)
        if (cfg is None or not getattr(cfg, "learnable_r", False)
                or getattr(cfg, "r_update_mode", "gradient") != "barycenter"):
            return set()
        return {name for name, _ in model.named_parameters()
                if name.startswith("prior_bank.r_")}

    @staticmethod
    @torch.no_grad()
    def _refresh_derived(model: torch.nn.Module) -> None:
        r"""Recompute the barycenter centroid after averaged model-channel tables are installed."""
        cfg = getattr(model, "cfg", None)
        prior_bank = getattr(model, "prior_bank", None)
        if (cfg is not None and prior_bank is not None
                and getattr(cfg, "learnable_r", False)
                and getattr(cfg, "r_update_mode", "gradient") == "barycenter"):
            prior_bank.barycenter_r_()

    @torch.no_grad()
    def reset(self, model: torch.nn.Module) -> None:
        r"""Reseed the shadow from the model's current params (e.g. after loading resumed weights).

        Used by ``load_checkpoint`` when a bundle carries no ``ema_state`` (a ``use_ema=False`` or
        legacy checkpoint): the shadow constructed at ``__init__`` clones the PRE-load fresh init,
        so without this reseed the running average would blend real weights into random-init noise
        (audit 2026-07-01 C3). Same ``requires_grad`` filter as ``__init__`` so frozen params (e.g.
        ``r_mu`` under ``learnable_r=False``) stay excluded consistently."""
        self._model_ref = weakref.ref(model)
        self.shadow = {name: param.detach().clone()
                       for name, param in model.named_parameters()
                       if param.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        r"""Blend the live parameters into the shadow: ``s <- decay*s + (1-decay)*theta``.

        A non-finite live parameter is SKIPPED (not blended): ``mul_(decay)`` of a NaN stays NaN
        forever, so a single transient NaN/Inf would permanently poison the running average and the
        final ``copy_to`` would write it into the evaluated/checkpointed model (audit 2026-06-17 r2 id19).
        """
        tracked = [
            (name, param.detach())
            for name, param in model.named_parameters()
            if name in self.shadow
        ]
        if not tracked:
            return
        finite = torch.stack([
            torch.isfinite(param).all()
            for _, param in tracked
        ]).cpu().tolist()
        for (name, param), is_finite in zip(tracked, finite):
            if is_finite:
                self.shadow[name].mul_(self.decay).add_(param, alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> None:
        r"""Stash the current live parameters so ``restore`` can put them back after a ``copy_to``."""
        derived = self._derived_parameter_names(model)
        self._backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if name in self.shadow or name in derived
        }

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        r"""Write the shadow (averaged) weights into the model parameters in place."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name])
        self._refresh_derived(model)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        r"""Write the stashed live parameters back into the model and drop the stash."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> Dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.decay = float(state["decay"])
        saved_shadow = state["shadow"]
        if not isinstance(saved_shadow, dict):
            raise TypeError(f"EMA shadow must be a dict, got {type(saved_shadow).__name__}")
        model = self._model_ref()
        current = ({name: param for name, param in model.named_parameters() if param.requires_grad}
                   if model is not None else self.shadow)
        loaded: Dict[str, torch.Tensor] = {}
        missing = []
        incompatible = []
        for name, param in current.items():
            saved = saved_shadow.get(name)
            if saved is None:
                missing.append(name)
                loaded[name] = param.detach().clone()
            elif not isinstance(saved, torch.Tensor) or saved.shape != param.shape:
                incompatible.append(name)
                loaded[name] = param.detach().clone()
            else:
                loaded[name] = saved.detach().to(device=param.device, dtype=param.dtype).clone()
        stale = sorted(set(saved_shadow) - set(current))
        self.shadow = loaded
        if missing or incompatible or stale:
            warnings.warn(
                "EMA shadow keys differed from the live model's current trainable parameters; "
                f"reseeded missing={sorted(missing)}, incompatible={sorted(incompatible)}, "
                f"dropped stale={stale} from the loaded model state.",
                UserWarning,
                stacklevel=2,
            )
