import pytest
import torch

from vfe3.geometry import retraction as retraction_module
from vfe3.geometry import transport as transport_module


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_public_spd_certificates_use_float64_for_lower_precision_representations(
    dtype:       torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    certificate_dtypes: list[torch.dtype] = []
    original_cholesky_ex = torch.linalg.cholesky_ex

    def recording_cholesky_ex(
        matrix:       torch.Tensor,
        *,
        upper:        bool = False,
        check_errors: bool = False,
    ) -> 'torch.return_types.linalg_cholesky_ex':
        certificate_dtypes.append(matrix.dtype)
        return original_cholesky_ex(matrix, upper=upper, check_errors=check_errors)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", recording_cholesky_ex)
    boundary = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=dtype)

    repaired = retraction_module._certify_public_spd(
        boundary,
        eps=0.125,
        sigma_max=2.0,
    )
    production_certificate_dtypes = tuple(certificate_dtypes)

    assert repaired.dtype == dtype
    assert production_certificate_dtypes
    assert all(item == torch.float64 for item in production_certificate_dtypes)
    represented = repaired.double()
    identity = torch.eye(2, dtype=torch.float64)
    assert original_cholesky_ex(
        represented - 0.125 * identity,
        check_errors=False,
    ).info.item() == 0
    assert original_cholesky_ex(
        2.0 * identity - represented,
        check_errors=False,
    ).info.item() == 0


@pytest.mark.parametrize(
    ("matrix_data", "skew_symmetric"),
    (
        ([[0.30, 1.10], [-0.20, -0.40]], False),
        ([[0.00, 1.00], [-1.00, 0.00]], True),
    ),
)
def test_dense_exp_pair_inverts_the_represented_forward_factor(
    matrix_data:    list[list[float]],
    skew_symmetric: bool,
    monkeypatch:    pytest.MonkeyPatch,
) -> None:
    matrix = torch.tensor(matrix_data, dtype=torch.float32, requires_grad=True)
    inverse_inputs: list[torch.Tensor] = []
    original_inverse = transport_module._checked_group_inverse

    def recording_inverse(value: torch.Tensor) -> torch.Tensor:
        inverse_inputs.append(value)
        return original_inverse(value)

    monkeypatch.setattr(transport_module, "_checked_group_inverse", recording_inverse)
    exp_pos, exp_neg = transport_module.stable_matrix_exp_pair(
        matrix,
        skew_symmetric=skew_symmetric,
        max_norm=float("inf"),
    )

    assert exp_neg is not None
    assert len(inverse_inputs) == 1
    assert torch.equal(inverse_inputs[0], exp_pos)
    expected_inverse = torch.linalg.inv(exp_pos.double()).to(exp_pos.dtype)
    assert torch.equal(exp_neg, expected_inverse)
    (exp_pos.square().sum() + exp_neg.square().sum()).backward()
    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()


def test_dense_block_exp_pair_inverts_each_represented_forward_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = torch.tensor(
        [
            [0.20, 0.70, 0.00, 0.00],
            [-0.10, -0.30, 0.00, 0.00],
            [0.00, 0.00, -0.40, 0.60],
            [0.00, 0.00, 0.20, 0.10],
        ],
        requires_grad=True,
    )
    inverse_inputs: list[torch.Tensor] = []
    original_inverse = transport_module._checked_group_inverse

    def recording_inverse(value: torch.Tensor) -> torch.Tensor:
        inverse_inputs.append(value)
        return original_inverse(value)

    monkeypatch.setattr(transport_module, "_checked_group_inverse", recording_inverse)
    exp_pos, exp_neg = transport_module.stable_matrix_exp_pair(matrix, block_dims=[2, 2])

    assert exp_neg is not None
    assert inverse_inputs
    represented_blocks = torch.stack((exp_pos[:2, :2], exp_pos[2:, 2:]))
    expected_blocks = torch.linalg.inv(represented_blocks.double()).to(exp_pos.dtype)
    for block_index, start in enumerate((0, 2)):
        end = start + 2
        assert torch.equal(exp_neg[start:end, start:end], expected_blocks[block_index])
    assert torch.count_nonzero(exp_neg[:2, 2:]) == 0
    assert torch.count_nonzero(exp_neg[2:, :2]) == 0
    exp_neg.square().sum().backward()
    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()


def test_compact_exp_pair_inverts_the_represented_forward_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocks = torch.tensor(
        [
            [[0.20, 0.70], [-0.10, -0.30]],
            [[-0.40, 0.60], [0.20, 0.10]],
        ],
        requires_grad=True,
    )
    inverse_inputs: list[torch.Tensor] = []
    original_inverse = transport_module._checked_group_inverse

    def recording_inverse(value: torch.Tensor) -> torch.Tensor:
        inverse_inputs.append(value)
        return original_inverse(value)

    monkeypatch.setattr(transport_module, "_checked_group_inverse", recording_inverse)
    exp_pos, exp_neg = transport_module._stable_compact_glk_exp_pair(blocks)

    assert len(inverse_inputs) == 1
    assert torch.equal(inverse_inputs[0], exp_pos)
    expected_inverse = torch.linalg.inv(exp_pos.double()).to(exp_pos.dtype)
    assert torch.equal(exp_neg, expected_inverse)
    exp_neg.square().sum().backward()
    assert blocks.grad is not None
    assert torch.isfinite(blocks.grad).all()


def test_forward_only_exp_pair_does_not_build_an_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_inverse(_value: torch.Tensor) -> torch.Tensor:
        raise AssertionError("only_forward=True attempted to build an inverse")

    monkeypatch.setattr(transport_module, "_checked_group_inverse", unexpected_inverse)
    exp_pos, exp_neg = transport_module.stable_matrix_exp_pair(
        torch.tensor([[0.20, 0.10], [-0.30, 0.40]]),
        only_forward=True,
    )

    assert torch.isfinite(exp_pos).all()
    assert exp_neg is None
