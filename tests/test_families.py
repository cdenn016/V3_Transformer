import math

import pytest
import torch


def test_family_registry_register_get_and_cov_kind():
    from vfe3.families.base import (
        BeliefParams, register_family, get_family, family_cov_kind, divergence_families,
    )

    class _ToyParams(BeliefParams):
        cov_kind = "diagonal"
        def __init__(self, x): self.x = x
        def coordinate_dim(self): return self.x.shape[-1]
        def block(self, start, end): return _ToyParams(self.x[..., start:end])
        def broadcast_over_keys(self): return _ToyParams(self.x.unsqueeze(-2))
        def natural(self): return (self.x,)
        @classmethod
        def log_partition_at(cls, theta): return theta[0].sum(dim=-1)
        def entropy(self): return self.x.sum(dim=-1) * 0.0

    register_family("toy_reg_test")(_ToyParams)
    try:
        assert get_family("toy_reg_test") is _ToyParams
        assert family_cov_kind("toy_reg_test") == "diagonal"
        assert "toy_reg_test" in divergence_families()
    finally:
        from vfe3.families.base import _FAMILIES
        _FAMILIES.pop("toy_reg_test", None)


def test_family_cov_kind_unregistered_raises():
    from vfe3.families.base import family_cov_kind
    with pytest.raises(KeyError):
        family_cov_kind("no_such_family")
