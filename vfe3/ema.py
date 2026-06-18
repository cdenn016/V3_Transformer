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

from typing import Dict

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
        # Track only trainable params: frozen tensors (e.g. r_mu with learnable_r=False) never move,
        # so averaging them is a wasteful no-op. Keyed by the stable parameter name.
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        r"""Blend the live parameters into the shadow: ``s <- decay*s + (1-decay)*theta``."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> None:
        r"""Stash the current live parameters so ``restore`` can put them back after a ``copy_to``."""
        self._backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if name in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        r"""Write the shadow (averaged) weights into the model parameters in place."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name])

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
        self.shadow = {name: tensor.clone() for name, tensor in state["shadow"].items()}
