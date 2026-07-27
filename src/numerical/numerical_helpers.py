"""
Provide focused matrix and symbolic helpers for canonicalization.

Run the helper tests with ``python -m pytest tests/test_algebra.py -q``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sympy as sp

from src.numerical.models import FloatArray
from src.numerical.symbols import x, y, z


def relative_tolerance(values: npt.ArrayLike, factor: float) -> float:
    """
    Return a machine-precision tolerance relative to explicit numerical values.

    Args:
        values: numpy.typing.ArrayLike
            Values whose largest magnitude defines the scale.
        factor: float
            Positive safety multiplier applied to machine epsilon.
        return: float
            Positive scale-aware tolerance.
    """

    if factor <= 0:
        raise ValueError("relative tolerance factor must be positive")
    array = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(array)))
    minimum_scale = float(np.finfo(np.float64).tiny)
    effective_scale = scale if scale > minimum_scale else minimum_scale
    return factor * float(np.finfo(np.float64).eps) * effective_scale


def numerical_rank(matrix: FloatArray, factor: float) -> int:
    """
    Return matrix rank under an explicit scale-relative tolerance.

    Args:
        matrix: numpy.ndarray
            Two-dimensional numerical matrix.
        factor: float
            Positive machine-epsilon safety multiplier.
        return: int
            Numerical matrix rank.
    """

    if matrix.ndim != 2:
        raise ValueError("numerical rank requires a two-dimensional matrix")
    return int(np.linalg.matrix_rank(matrix, tol=relative_tolerance(matrix, factor)))



def clean_near_zero(matrix: FloatArray, threshold: float) -> FloatArray:
    """
    Return a copy with numerical values below an absolute threshold set to zero.

    Args:
        matrix: numpy.ndarray
            Numerical array to clean.
        threshold: float
            Strict absolute cutoff for values replaced by zero.
    return: numpy.ndarray
        Cleaned array with the same shape.
    """

    cleaned = np.asarray(matrix).copy()
    cleaned[np.abs(cleaned) < threshold] = 0
    return cleaned


def assign_quadratic_block(homogeneous: FloatArray, quadratic: FloatArray) -> FloatArray:
    """
    Return a homogeneous matrix with its 3x3 quadratic block replaced.

    Args:
        homogeneous: numpy.ndarray
            4x4 destination matrix.
        quadratic: numpy.ndarray
            3x3 replacement block.
    return: numpy.ndarray
        Updated copy of the homogeneous matrix.
    """

    if homogeneous.shape != (4, 4) or quadratic.shape != (3, 3):
        raise ValueError("expected homogeneous shape (4, 4) and quadratic shape (3, 3)")
    result = homogeneous.copy()
    result[:3, :3] = quadratic
    return result


def assign_linear_block(homogeneous: FloatArray, linear: npt.ArrayLike) -> FloatArray:
    """
    Return a homogeneous matrix with its symmetric linear block replaced.

    Args:
        homogeneous: numpy.ndarray
            4x4 destination matrix.
        linear: numpy.typing.ArrayLike
            Three linear half-coefficients.
    return: numpy.ndarray
        Updated copy of the homogeneous matrix.
    """

    if homogeneous.shape != (4, 4):
        raise ValueError("homogeneous matrix must have shape (4, 4)")
    vector = np.asarray(linear, dtype=np.float64).reshape(3)
    result = homogeneous.copy()
    result[:3, 3] = vector
    result[3, :3] = vector
    return result


def round_for_display(matrix: npt.ArrayLike, decimals: int) -> FloatArray:
    """
    Convert an array-like value to float and round it for graphics output.

    Args:
        matrix: numpy.typing.ArrayLike
            Numerical values to round.
        decimals: int
            Number of decimal places retained.
        return: numpy.ndarray
            Rounded float array.
    """

    return np.round(np.asarray(matrix, dtype=np.float64), decimals=decimals).astype(np.float64)


def expression_from_matrix(matrix: npt.ArrayLike) -> sp.Expr:
    """
    Reconstruct and expand a quadric expression from a homogeneous matrix.

    Args:
        matrix: numpy.typing.ArrayLike
            Symmetric 4x4 homogeneous quadric matrix.
        return: sympy.Expr
            Expanded expression in x, y, and z.
    """

    array = np.asarray(matrix)
    if array.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4)")
    coordinates = sp.Matrix([[x, y, z, 1]])
    return sp.expand((coordinates * sp.Matrix(array) * coordinates.T)[0, 0])


def normalize_integer_coefficients(expression: sp.Expr, tolerance: float) -> sp.Expr:
    """
    Replace float coefficients near integers with exact SymPy integers.

    Args:
        expression: sympy.Expr
            Polynomial expression to normalize.
        tolerance: float
            Maximum distance from an integer accepted for conversion.
    return: sympy.Expr
        Expanded expression with normalized coefficients.
    """

    normalized: sp.Expr = sp.Integer(0)
    for term, coefficient in sp.expand(expression).as_coefficients_dict().items():
        new_coefficient: sp.Expr | int = coefficient
        if isinstance(coefficient, sp.Float) and abs(float(coefficient) - round(float(coefficient))) < tolerance:
            new_coefficient = int(round(float(coefficient)))
        normalized += new_coefficient * term
    return normalized
