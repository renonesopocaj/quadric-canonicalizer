"""
Compute canonical metric forms and typed transformation artifacts.

Run the transformation tests with ``python -m pytest tests/test_transformer.py -q``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy import linalg as la

from src.numerical.numerical_helpers import (
    clean_near_zero,
    expression_from_matrix,
    normalize_integer_coefficients,
    numerical_rank,
    relative_tolerance,
)
from src.numerical.classifier import QuadricClassifier
from src.numerical.models import CanonicalizationResult, FloatArray, QuadricType
from src.numerical.parabolic_cylinder import parabolic_cylinder_canonize
from src.numerical.parser import QuadricParser


NUMERICAL_TOLERANCE = 1e-10
ROUNDOFF_FACTOR = 100.0
COEFFICIENT_ROUNDOFF_TOLERANCE = ROUNDOFF_FACTOR * float(np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class TransformationData:
    """Store the unvalidated internal artifacts of one transformation."""

    initial_matrix: FloatArray
    middle_matrix: FloatArray
    final_matrix: FloatArray
    translation_vector: FloatArray
    rotation_matrix: FloatArray


def convert_poly_coeffs(expr: sp.Expr) -> sp.Expr:
    """
    Convert numerically integral polynomial coefficients to exact integers.

    Args:
        expr: sympy.Expr
            Polynomial expression in x, y, and z.
        return: sympy.Expr
            Equivalent expression with exact integral coefficients.
    """

    return normalize_integer_coefficients(expr, COEFFICIENT_ROUNDOFF_TOLERANCE)


def _homogeneous_transform(linear_map: FloatArray, offset: FloatArray) -> FloatArray:
    """Return a 4x4 affine matrix from an explicit linear map and offset."""

    if linear_map.shape != (3, 3) or offset.shape != (3,):
        raise ValueError("expected linear_map shape (3, 3) and offset shape (3,)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear_map
    transform[:3, 3] = offset
    return transform


def _roundoff_threshold(matrix: FloatArray) -> float:
    """Return a scale-aware threshold for floating-point cancellation artifacts."""

    return relative_tolerance(matrix, ROUNDOFF_FACTOR)


def _clean_roundoff(matrix: FloatArray) -> FloatArray:
    """Set only scale-relative floating-point cancellation artifacts to zero."""

    return clean_near_zero(matrix, _roundoff_threshold(matrix))


def _proper_symmetric_eigendecomposition(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
    """
    Diagonalize a symmetric matrix in positive-negative-null eigenvalue order.

    The input equation sign is preserved. Repeated eigenspaces and eigenvector
    signs do not define a unique basis; this function only fixes the ordering
    and positive orientation of one valid orthonormal basis.

    Args:
        matrix: numpy.ndarray
            Symmetric 3x3 quadratic coefficient matrix.
        return: tuple[numpy.ndarray, numpy.ndarray]
            Ordered diagonal eigenvalue matrix and determinant-one eigenvector
            matrix whose columns are the corresponding eigenvectors.
    """

    eigenvalues, eigenvectors = la.eigh(matrix)
    ordered_threshold = _roundoff_threshold(matrix)
    cleaned_eigenvalues = clean_near_zero(eigenvalues, ordered_threshold)
    positive_indices = np.flatnonzero(cleaned_eigenvalues > 0.0)
    negative_indices = np.flatnonzero(cleaned_eigenvalues < 0.0)
    null_indices = np.flatnonzero(cleaned_eigenvalues == 0.0)
    ordered_indices = np.concatenate((positive_indices, negative_indices, null_indices))
    ordered_eigenvalues = cleaned_eigenvalues[ordered_indices]
    ordered_eigenvectors = eigenvectors[:, ordered_indices]
    diagonal = np.diag(ordered_eigenvalues)
    if np.linalg.det(ordered_eigenvectors) < 0:
        ordered_eigenvectors[:, -1] *= -1
    return (
        np.asarray(diagonal, dtype=np.float64),
        np.asarray(ordered_eigenvectors, dtype=np.float64),
    )


def centered_quadric(
    A_overline: FloatArray,
    A: FloatArray,
    b: FloatArray,
) -> TransformationData:
    """
    Canonicalize a full-rank quadric through rotation then translation.

    Args:
        A_overline: numpy.ndarray
            Normalized initial homogeneous matrix.
        A: numpy.ndarray
            Full-rank symmetric quadratic block.
        b: numpy.ndarray
            Linear half-coefficient column.
        return: TransformationData
            Exact stage matrices and active point transformations.
    """

    initial_matrix = A_overline.copy()
    diagonal, basis = _proper_symmetric_eigendecomposition(A)
    coordinate_rotation = _homogeneous_transform(basis, np.zeros(3, dtype=np.float64))
    middle_matrix = coordinate_rotation.T @ initial_matrix @ coordinate_rotation
    diagonal_values = np.diag(diagonal)
    if np.any(np.abs(diagonal_values) <= _roundoff_threshold(diagonal)):
        raise ValueError("full-rank canonicalization requires three non-zero eigenvalues")
    transformed_linear = np.asarray(basis.T @ b.reshape(3), dtype=np.float64)
    coordinate_translation = -transformed_linear / diagonal_values
    translation_transform = _homogeneous_transform(
        np.eye(3, dtype=np.float64),
        np.asarray(coordinate_translation, dtype=np.float64),
    )
    final_matrix = translation_transform.T @ middle_matrix @ translation_transform

    return TransformationData(
        initial_matrix=initial_matrix,
        middle_matrix=_clean_roundoff(middle_matrix),
        final_matrix=_clean_roundoff(final_matrix),
        translation_vector=np.asarray(-coordinate_translation, dtype=np.float64),
        rotation_matrix=np.asarray(basis.T, dtype=np.float64),
    )


def _rank_two_coordinate_translation(
    middle_matrix: FloatArray,
    diagonal: FloatArray,
    linear: FloatArray,
) -> FloatArray:
    """
    Return the coordinate translation for a rank-two canonical quadratic block.

    Args:
        middle_matrix: numpy.ndarray
            Matrix after the rotation stage.
        diagonal: numpy.ndarray
            Diagonal rank-two quadratic block.
        linear: numpy.ndarray
            Rotated linear half-coefficients.
        return: numpy.ndarray
            Coordinate offset from the final stage to the middle stage.
    """

    diagonal_values = np.diag(diagonal)
    active = np.abs(diagonal_values) > _roundoff_threshold(diagonal)
    if int(np.count_nonzero(active)) != 2:
        raise ValueError("rank-two translation requires exactly two non-zero diagonal coefficients")
    coordinate_translation = np.zeros(3, dtype=np.float64)
    coordinate_translation[active] = -linear[active] / diagonal_values[active]
    reduced_constant = float(
        middle_matrix[3, 3] - np.sum((linear[active] ** 2) / diagonal_values[active])
    )
    null_index = int(np.flatnonzero(~active)[0])
    if abs(linear[null_index]) > _roundoff_threshold(linear):
        coordinate_translation[null_index] = -reduced_constant / (2.0 * linear[null_index])
    return coordinate_translation


def _rank_one_coordinate_translation(diagonal: FloatArray, linear: FloatArray) -> FloatArray:
    """
    Return the minimum-norm translation for a rank-one nonparabolic form.

    Args:
        diagonal: numpy.ndarray
            Diagonal rank-one quadratic block.
        linear: numpy.ndarray
            Rotated linear half-coefficients.
        return: numpy.ndarray
            Coordinate offset eliminating the active linear coefficient.
    """

    diagonal_values = np.diag(diagonal)
    active_indices = np.flatnonzero(np.abs(diagonal_values) > _roundoff_threshold(diagonal))
    if active_indices.size != 1:
        raise ValueError("rank-one translation requires exactly one non-zero diagonal coefficient")
    active_index = int(active_indices[0])
    null_linear = np.delete(linear, active_index)
    if np.any(np.abs(null_linear) > _roundoff_threshold(linear)):
        raise ValueError("rank-one non-parabolic quadric has an unexpected null-space linear term")
    coordinate_translation = np.zeros(3, dtype=np.float64)
    coordinate_translation[active_index] = -linear[active_index] / diagonal_values[active_index]
    return coordinate_translation


def acentered_quadric(
    quadric_type: QuadricType,
    A_overline: FloatArray,
    A: FloatArray,
    b: FloatArray,
    eq: str,
) -> TransformationData:
    """
    Canonicalize a rank-deficient quadric through rotation then translation.

    Args:
        quadric_type: QuadricType
            Classified non-centered quadric family.
        A_overline: numpy.ndarray
            Normalized initial homogeneous matrix.
        A: numpy.ndarray
            Rank-one or rank-two symmetric quadratic block.
        b: numpy.ndarray
            Linear half-coefficient column.
        eq: str
            Original equation retained by the parabolic-cylinder compatibility
            boundary.
        return: TransformationData
            Exact stage matrices and active point transformations.
    """

    A_overline_og = A_overline.copy()
    if quadric_type is QuadricType.PARABOLIC_CYLINDER:
        A_overline, basis, coordinate_translation, A_overline_middle = parabolic_cylinder_canonize(
            A_overline.copy(), A.copy(), b, eq, A_overline_og.copy()
        )
    else:
        diagonal, basis = _proper_symmetric_eigendecomposition(A)
        coordinate_rotation = _homogeneous_transform(basis, np.zeros(3, dtype=np.float64))
        A_overline_middle = coordinate_rotation.T @ A_overline_og @ coordinate_rotation
        transformed_linear = A_overline_middle[:3, 3].copy()
        rank = numerical_rank(diagonal, ROUNDOFF_FACTOR)
        if rank == 2:
            coordinate_translation = _rank_two_coordinate_translation(
                A_overline_middle,
                diagonal,
                transformed_linear,
            )
        elif rank == 1:
            coordinate_translation = _rank_one_coordinate_translation(diagonal, transformed_linear)
        else:
            raise ValueError(f"non-centered canonicalization requires rank one or two; received rank {rank}")
        translation_transform = _homogeneous_transform(
            np.eye(3, dtype=np.float64),
            np.asarray(coordinate_translation, dtype=np.float64).reshape(3),
        )
        A_overline = translation_transform.T @ A_overline_middle @ translation_transform
    A_overline_middle = _clean_roundoff(A_overline_middle)
    A_overline = _clean_roundoff(A_overline)
    return TransformationData(
        initial_matrix=A_overline_og,
        middle_matrix=A_overline_middle,
        final_matrix=A_overline,
        translation_vector=-np.asarray(coordinate_translation, dtype=np.float64).reshape(3),
        rotation_matrix=np.asarray(basis.T, dtype=np.float64),
    )


class QuadricCanonicalizer:
    """Orchestrate parsing, classification, and canonical transformation strategies."""

    parser: QuadricParser
    classifier: QuadricClassifier

    def __init__(self, parser: QuadricParser, classifier: QuadricClassifier) -> None:
        self.parser = parser
        self.classifier = classifier

    def canonize(self, eq: str) -> CanonicalizationResult:
        """
        Transform one equation into a validated canonicalization result.

        Args:
            eq: str
                Degree-two equation in x, y, and z.
            return: CanonicalizationResult
                Typed matrices, equations, and transformations for the quadric.
        """

        matrices = self.parser.parse_matrices(eq)
        matrix_scale = float(np.max(np.abs(matrices.homogeneous)))
        if matrix_scale == 0:
            raise ValueError("quadric matrix cannot be identically zero")
        homogeneous = matrices.homogeneous / matrix_scale
        quadratic = matrices.quadratic / matrix_scale
        linear = matrices.linear / matrix_scale
        quadric_type = self.classifier.classify(quadratic, homogeneous)
        centered = numerical_rank(quadratic, ROUNDOFF_FACTOR) == 3
        if centered:
            data = centered_quadric(
                homogeneous.copy(), quadratic.copy(), linear.copy()
            )
        else:
            data = acentered_quadric(
                quadric_type, homogeneous.copy(), quadratic.copy(), linear.copy(), eq
            )
        return _build_result(quadric_type, centered, data, matrix_scale)


def _build_result(
    quadric_type: QuadricType,
    centered: bool,
    data: TransformationData,
    matrix_scale: float,
) -> CanonicalizationResult:
    initial_matrix = np.asarray(data.initial_matrix * matrix_scale, dtype=np.float64)
    middle_matrix = np.asarray(data.middle_matrix * matrix_scale, dtype=np.float64)
    final_matrix = np.asarray(data.final_matrix * matrix_scale, dtype=np.float64)
    return CanonicalizationResult(
        quadric_type=quadric_type,
        centered=centered,
        initial_matrix=initial_matrix,
        middle_matrix=middle_matrix,
        final_matrix=final_matrix,
        translation_vector=data.translation_vector,
        rotation_matrix=data.rotation_matrix,
        initial_equation=convert_poly_coeffs(expression_from_matrix(initial_matrix)),
        middle_equation=convert_poly_coeffs(expression_from_matrix(middle_matrix)),
        final_equation=convert_poly_coeffs(expression_from_matrix(final_matrix)),
    )


def canonize_quadric(eq: str) -> CanonicalizationResult:
    """
    Parse, classify, and transform one quadric into canonical metric form.

    The result reports ordered active point transformations using
    ``next = linear_map @ current + offset``. Numerical values remain full
    precision; presentation code is responsible for rounding.

    Args:
        eq: str
            Degree-two equation in x, y, and z containing one equals sign.
        return: CanonicalizationResult
            Immutable matrices, equations, classification, and transformation
            steps.
    """

    canonicalizer = QuadricCanonicalizer(parser=QuadricParser(),
                                         classifier=QuadricClassifier(tolerance=NUMERICAL_TOLERANCE))
    return canonicalizer.canonize(eq)


__all__ = ["QuadricCanonicalizer", "canonize_quadric"]
