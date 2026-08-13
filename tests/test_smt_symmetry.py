import math

import torch

from utils.symmetry import augment_continuous_symmetry, continuous_axis_error_deg


def test_z_symmetry_keeps_normal_and_changes_yaw():
    rotations = torch.eye(3).unsqueeze(0)
    sym_info = torch.tensor([[0, 0, 0, 1]], dtype=torch.int8)
    augmented = augment_continuous_symmetry(
        rotations, sym_info, angles=torch.tensor([math.pi / 2])
    )
    assert torch.allclose(augmented[0, :, 2], rotations[0, :, 2])
    assert torch.allclose(
        augmented[0],
        torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-6,
    )


def test_non_symmetric_rotation_is_unchanged():
    rotations = torch.eye(3).repeat(2, 1, 1)
    sym_info = torch.zeros((2, 4), dtype=torch.int8)
    augmented = augment_continuous_symmetry(
        rotations, sym_info, angles=torch.tensor([0.3, 1.2])
    )
    assert torch.equal(augmented, rotations)


def test_axis_error_ignores_yaw_but_not_normal_flip():
    gt = torch.eye(3).repeat(2, 1, 1)
    pred = gt.clone()
    pred[0] = augment_continuous_symmetry(
        gt[:1], torch.tensor([[0, 0, 0, 1]], dtype=torch.int8), torch.tensor([1.1])
    )[0]
    pred[1] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    sym_info = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=torch.int8)
    error = continuous_axis_error_deg(pred, gt, sym_info)
    assert torch.allclose(error, torch.tensor([0.0, 180.0]), atol=1e-3)
