import torch

from vfe3.config import VFE3Config


def test_deterministic_config_field_default_and_settable():
    assert VFE3Config().deterministic is False                    # default: today's byte-identical path
    assert VFE3Config(deterministic=True).deterministic is True


def test_seed_everything_default_leaves_global_determinism_untouched():
    import ablation
    was = torch.are_deterministic_algorithms_enabled()
    ablation._seed_everything(0)                                   # default deterministic=False
    assert torch.are_deterministic_algorithms_enabled() == was    # default path must not flip global state


def test_seed_everything_deterministic_toggle_sets_flags():
    from vfe3.runtime import seed_everything
    was = torch.are_deterministic_algorithms_enabled()
    was_cudnn = torch.backends.cudnn.deterministic
    try:
        seed_everything(0, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:                                                       # restore so this does not leak into other tests
        torch.use_deterministic_algorithms(was)
        torch.backends.cudnn.deterministic = was_cudnn


def test_seed_everything_true_then_false_is_reversible():
    from vfe3.runtime import seed_everything
    seed_everything(1, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    seed_everything(1, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
