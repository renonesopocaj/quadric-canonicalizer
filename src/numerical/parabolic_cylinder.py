"""
Canonicalize parabolic cylinders by rotation and minimum-norm translation.

Run the focused checks with
``python -m pytest tests/test_transformer.py -q -k parabolic_cylinder``.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as la

from src.numerical.models import FloatArray
from src.numerical.numerical_helpers import clean_near_zero, relative_tolerance


ROUNDOFF_FACTOR = 100.0


def _roundoff_threshold(matrix: FloatArray) -> float:
    """Return a scale-aware threshold for floating-point cancellation artifacts."""

    return relative_tolerance(matrix, ROUNDOFF_FACTOR)


def _homogeneous_transform(linear_map: FloatArray, offset: FloatArray) -> FloatArray:
    """Return a homogeneous affine matrix after validating explicit shapes."""

    if linear_map.shape != (3, 3) or offset.shape != (3,):
        raise ValueError("expected linear_map shape (3, 3) and offset shape (3,)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear_map
    transform[:3, 3] = offset
    return transform


def parabolic_cylinder_canonize(
    A_overline: FloatArray,
    A: FloatArray,
    b: FloatArray,
    eq: str,
    A_overline_og: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Build the two canonical stages for a rank-one parabolic cylinder.

    One reported rotation combines the quadratic eigendirection rotation and
    the additional null-plane rotation from the analytic reduction. It aligns
    the null-space linear term with the second canonical axis. The returned
    coordinate translation is expressed in that rotated frame and has no
    component along the free cylinder axis.

    Args:
        A_overline: numpy.ndarray
            Working 4x4 homogeneous matrix.
        A: numpy.ndarray
            Symmetric rank-one quadratic block.
        b: numpy.ndarray
            Three linear half-coefficients.
        eq: str
            Original equation retained for compatibility; the numerical
            algorithm does not need to parse it again.
        A_overline_og: numpy.ndarray
            Original 4x4 homogeneous matrix.
        return: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Final matrix, proper basis, coordinate translation, and
            rotation-only middle matrix.
    """

    if A_overline.shape != (4, 4) or A_overline_og.shape != (4, 4):
        raise ValueError("parabolic-cylinder homogeneous matrices must have shape (4, 4)")
    if A.shape != (3, 3) or b.size != 3:
        raise ValueError("expected quadratic shape (3, 3) and three linear coefficients")
    if not eq:
        raise ValueError("the original equation must not be empty")

    eigenvalues, eigenvectors = la.eigh(A)
    nonzero_indices = np.flatnonzero(np.abs(eigenvalues) > _roundoff_threshold(A))
    if nonzero_indices.size != 1:
        raise ValueError("a parabolic cylinder must have a rank-one quadratic block")

    quadratic_index = int(nonzero_indices[0])
    quadratic_value = float(eigenvalues[quadratic_index])
    quadratic_direction = eigenvectors[:, quadratic_index]
    linear = np.asarray(b, dtype=np.float64).reshape(3)
    quadratic_linear = float(quadratic_direction @ linear)
    null_linear = linear - quadratic_linear * quadratic_direction
    null_linear_norm = float(la.norm(null_linear))
    if null_linear_norm <= _roundoff_threshold(linear):
        raise ValueError("a parabolic cylinder must have a linear term in the quadratic null space")

    parabolic_direction = null_linear / null_linear_norm
    free_direction = np.cross(quadratic_direction, parabolic_direction)
    free_direction /= la.norm(free_direction)
    basis = np.column_stack([quadratic_direction, parabolic_direction, free_direction])
    if np.linalg.det(basis) < 0:
        free_direction *= -1
        basis = np.column_stack([quadratic_direction, parabolic_direction, free_direction])

    quadratic_coordinate = -quadratic_linear / quadratic_value
    reduced_constant = float(
        A_overline_og[3, 3] - (quadratic_linear * quadratic_linear) / quadratic_value
    )
    parabolic_coordinate = -reduced_constant / (2.0 * null_linear_norm)
    coordinate_translation = np.array(
        [quadratic_coordinate, parabolic_coordinate, 0.0],
        dtype=np.float64,
    )

    rotation = _homogeneous_transform(np.asarray(basis, dtype=np.float64), np.zeros(3, dtype=np.float64))
    middle_matrix = rotation.T @ A_overline_og @ rotation
    translation = _homogeneous_transform(np.eye(3, dtype=np.float64), coordinate_translation)
    final_matrix = translation.T @ middle_matrix @ translation

    return (
        clean_near_zero(final_matrix, _roundoff_threshold(final_matrix)),
        np.asarray(basis, dtype=np.float64),
        np.asarray(coordinate_translation, dtype=np.float64),
        clean_near_zero(middle_matrix, _roundoff_threshold(middle_matrix)),
    )


__all__ = ["parabolic_cylinder_canonize"]
