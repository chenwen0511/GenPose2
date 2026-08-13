import math

import numpy as np

from utils.smt_metrics import adds_errors_m


def test_adds_is_zero_for_identity():
    vertices = np.array([
        [-0.1, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, -0.1, 0.0],
    ])
    rotation = np.eye(3)[None, ...]
    translation = np.zeros((1, 3))
    error = adds_errors_m(rotation, translation, rotation, translation, vertices)
    assert np.allclose(error, 0.0)


def test_adds_respects_discrete_symmetric_correspondence():
    vertices = np.array([
        [-0.1, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, -0.1, 0.0],
    ])
    rotation_gt = np.eye(3)[None, ...]
    angle = math.pi / 2
    rotation_pred = np.array([[
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]])
    translation = np.zeros((1, 3))
    error = adds_errors_m(
        rotation_pred, translation, rotation_gt, translation, vertices
    )
    assert np.allclose(error, 0.0, atol=1e-8)
