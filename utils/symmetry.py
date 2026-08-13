import math

import torch


_ANY_TAG = 1


def _axis_angle_rotation(axis, angles, dtype, device):
    """Return batched local-axis rotation matrices."""
    count = angles.shape[0]
    result = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(count, 1, 1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    if axis == 0:
        result[:, 1, 1] = cos
        result[:, 1, 2] = -sin
        result[:, 2, 1] = sin
        result[:, 2, 2] = cos
    elif axis == 1:
        result[:, 0, 0] = cos
        result[:, 0, 2] = sin
        result[:, 2, 0] = -sin
        result[:, 2, 2] = cos
    elif axis == 2:
        result[:, 0, 0] = cos
        result[:, 0, 1] = -sin
        result[:, 1, 0] = sin
        result[:, 1, 1] = cos
    else:
        raise ValueError(f'invalid symmetry axis: {axis}')
    return result


def augment_continuous_symmetry(rotations, sym_info, angles=None):
    """Sample equivalent rotations for objects with one continuous symmetry axis.

    ``sym_info`` follows GenPose2/cutoop's ``[any, x, y, z]`` encoding, where
    tag value 1 means arbitrary rotation around the corresponding local axis.
    The sampled group element is right-multiplied because symmetry is defined
    in the object-local frame. Non-symmetric rows are returned unchanged.
    """
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError('rotations must have shape [B, 3, 3]')
    if sym_info.ndim != 2 or sym_info.shape != (rotations.shape[0], 4):
        raise ValueError('sym_info must have shape [B, 4]')

    result = rotations.clone()
    if angles is None:
        angles = torch.rand(
            rotations.shape[0], dtype=rotations.dtype, device=rotations.device
        ) * (2.0 * math.pi)
    else:
        angles = torch.as_tensor(angles, dtype=rotations.dtype, device=rotations.device)
        if angles.shape != (rotations.shape[0],):
            raise ValueError('angles must have shape [B]')

    axis_tags = sym_info[:, 1:4].to(device=rotations.device)
    continuous = axis_tags == _ANY_TAG
    for axis in range(3):
        mask = continuous[:, axis] & (continuous.sum(dim=1) == 1)
        if torch.any(mask):
            group_rotation = _axis_angle_rotation(
                axis,
                angles[mask],
                rotations.dtype,
                rotations.device,
            )
            result[mask] = torch.matmul(rotations[mask], group_rotation)
    return result


def continuous_axis_error_deg(pred_rotation, gt_rotation, sym_info):
    """Signed-axis angular error for single-axis continuous symmetries.

    Non-matching rows are returned as NaN so callers can combine this with a
    generic SO(3) metric for non-symmetric objects.
    """
    if pred_rotation.shape != gt_rotation.shape or pred_rotation.shape[-2:] != (3, 3):
        raise ValueError('rotation tensors must have matching [B, 3, 3] shapes')
    errors = torch.full(
        (pred_rotation.shape[0],),
        float('nan'),
        dtype=pred_rotation.dtype,
        device=pred_rotation.device,
    )
    axis_tags = sym_info[:, 1:4].to(device=pred_rotation.device)
    continuous = axis_tags == _ANY_TAG
    for axis in range(3):
        mask = continuous[:, axis] & (continuous.sum(dim=1) == 1)
        if torch.any(mask):
            pred_axis = pred_rotation[mask, :, axis]
            gt_axis = gt_rotation[mask, :, axis]
            cosine = torch.sum(pred_axis * gt_axis, dim=1).clamp(-1.0, 1.0)
            errors[mask] = torch.rad2deg(torch.acos(cosine))
    return errors
