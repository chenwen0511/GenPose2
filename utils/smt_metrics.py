from functools import lru_cache

import numpy as np
import torch
from scipy.spatial import cKDTree

from utils.genpose_utils import get_pose_dim
from utils.misc import get_rot_matrix
from utils.symmetry import continuous_axis_error_deg


@lru_cache(maxsize=8)
def load_obj_vertices(path):
    vertices = []
    with open(path, 'r') as stream:
        for line in stream:
            if line.startswith('v '):
                vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f'OBJ contains no vertices: {path}')
    return np.asarray(vertices, dtype=np.float64)


def _transform_points(vertices, rotation, translation):
    return vertices @ rotation.T + translation[None, :]


def adds_errors_m(pred_rotation, pred_translation, gt_rotation, gt_translation, vertices):
    """Compute standard ADD-S mean closest-point distance in metres."""
    errors = []
    for pred_r, pred_t, gt_r, gt_t in zip(
        pred_rotation, pred_translation, gt_rotation, gt_translation
    ):
        pred_points = _transform_points(vertices, pred_r, pred_t)
        gt_points = _transform_points(vertices, gt_r, gt_t)
        distances, _ = cKDTree(gt_points).query(pred_points, k=1)
        errors.append(float(np.mean(distances)))
    return np.asarray(errors, dtype=np.float64)


def continuous_symmetry_adds_errors_m(
    pred_rotation,
    pred_translation,
    gt_rotation,
    gt_translation,
    vertices,
    symmetry_axes,
):
    """Minimize ADD analytically over a continuous local-axis symmetry group.

    This avoids a discretization penalty when a physically circular object is
    represented by a sparse polygonal OBJ. It is the known-symmetry form of
    ADD-S: ``min_{S in symmetry group} ADD(T_pred S, T_gt)``.
    """
    errors = []
    for pred_r, pred_t, gt_r, gt_t, axis in zip(
        pred_rotation,
        pred_translation,
        gt_rotation,
        gt_translation,
        symmetry_axes,
    ):
        axis = int(axis)
        plane = [index for index in range(3) if index != axis]
        first, second = plane
        relative = pred_r.T @ gt_r
        angle = np.arctan2(
            relative[second, first] - relative[first, second],
            relative[first, first] + relative[second, second],
        )
        cosine, sine = np.cos(angle), np.sin(angle)
        group = np.eye(3)
        group[first, first] = cosine
        group[first, second] = -sine
        group[second, first] = sine
        group[second, second] = cosine
        aligned_pred = pred_r @ group
        pred_points = _transform_points(vertices, aligned_pred, pred_t)
        gt_points = _transform_points(vertices, gt_r, gt_t)
        errors.append(float(np.linalg.norm(pred_points - gt_points, axis=1).mean()))
    return np.asarray(errors, dtype=np.float64)


def _summary(values):
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'item': values,
    }


def get_smt_pose_metrics(
    pred_pose,
    gt_pose,
    sym_info,
    pose_mode,
    model_path,
    object_diameter=0.1778,
):
    """Return symmetry-aware SMT tray metrics.

    Normal error is the primary orientation metric for a continuously
    rotationally symmetric tray. ADD-S remains a secondary geometry metric.
    """
    rotation_dims = get_pose_dim(pose_mode) - 3
    pred_rotation_t = get_rot_matrix(pred_pose[:, :rotation_dims], pose_mode)
    gt_rotation_t = get_rot_matrix(gt_pose[:, :rotation_dims], pose_mode)
    pred_translation = pred_pose[:, rotation_dims:].detach().cpu().numpy()
    gt_translation = gt_pose[:, rotation_dims:].detach().cpu().numpy()

    normal_error = continuous_axis_error_deg(
        pred_rotation_t,
        gt_rotation_t,
        sym_info.to(pred_rotation_t.device),
    ).detach().cpu().numpy()
    if np.isnan(normal_error).any():
        raise ValueError('SMT metrics require one continuous symmetry axis per sample')

    vertices = load_obj_vertices(model_path)
    sym_info_numpy = sym_info.detach().cpu().numpy()
    symmetry_axes = []
    for row in sym_info_numpy:
        axes = np.flatnonzero(row[1:4] == 1)
        if len(axes) != 1:
            raise ValueError('SMT ADD-S requires exactly one continuous symmetry axis')
        symmetry_axes.append(int(axes[0]))
    adds = continuous_symmetry_adds_errors_m(
        pred_rotation_t.detach().cpu().numpy(),
        pred_translation,
        gt_rotation_t.detach().cpu().numpy(),
        gt_translation,
        vertices,
        symmetry_axes,
    )
    translation_m = np.linalg.norm(pred_translation - gt_translation, axis=1)
    threshold_005d = 0.05 * object_diameter
    threshold_01d = 0.10 * object_diameter

    return {
        'normal_error_deg': _summary(normal_error),
        'adds_m': _summary(adds),
        'smt_accuracy': {
            'normal_5deg': float(np.mean(normal_error < 5.0)),
            'normal_10deg': float(np.mean(normal_error < 10.0)),
            'adds_0.05d': float(np.mean(adds < threshold_005d)),
            'adds_0.10d': float(np.mean(adds < threshold_01d)),
            'adds_auc_0.10d': float(np.mean(1.0 - np.minimum(adds, threshold_01d) / threshold_01d)),
            'normal_10deg_trans_2cm': float(
                np.mean((normal_error < 10.0) & (translation_m < 0.02))
            ),
        },
    }
