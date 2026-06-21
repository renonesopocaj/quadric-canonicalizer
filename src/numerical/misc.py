"""
Preserve legacy helper imports while implementations live in focused modules.

New code should import from :mod:`src.numerical.algebra` or
:mod:`src.numerical.parser`. Run checks with ``python -m pytest tests -q``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sympy as sp

from src.numerical.algebra import assign_linear_block, assign_quadratic_block, clean_near_zero
from src.numerical.models import FloatArray
from src.numerical.parser import QuadricParser


def assign_A_to_overA(A_overline: FloatArray, A: FloatArray) -> FloatArray:
    """
    Replace the quadratic block of a homogeneous matrix.

    Args:
        A_overline: numpy.ndarray
            Homogeneous 4x4 matrix.
        A: numpy.ndarray
            Quadratic 3x3 replacement block.
        return: numpy.ndarray
            Updated matrix copy.
    """

    return assign_quadratic_block(A_overline, A)


def assign_b_to_overA(A_overline: FloatArray, b: npt.ArrayLike) -> FloatArray:
    """
    Replace both symmetric linear blocks of a homogeneous matrix.

    Args:
        A_overline: numpy.ndarray
            Homogeneous 4x4 matrix.
        b: numpy.typing.ArrayLike
            Three linear half-coefficients.
        return: numpy.ndarray
            Updated matrix copy.
    """

    return assign_linear_block(A_overline, b)


def substitute_col(matrix: FloatArray, vector: npt.ArrayLike, col_i: int) -> FloatArray:
    """
    Return a matrix copy with one column replaced.

    Args:
        matrix: numpy.ndarray
            Two-dimensional destination matrix.
        vector: numpy.typing.ArrayLike
            Replacement values matching the matrix row count.
        col_i: int
            Zero-based destination column index.
        return: numpy.ndarray
            Matrix copy containing the replacement column.
    """

    if matrix.ndim != 2 or not 0 <= col_i < matrix.shape[1]:
        raise ValueError(f"column {col_i} is invalid for matrix shape {matrix.shape}")
    replacement = np.asarray(vector, dtype=matrix.dtype).reshape(-1)
    if replacement.size != matrix.shape[0]:
        raise ValueError(f"vector length {replacement.size} does not match matrix rows {matrix.shape[0]}")
    result = matrix.copy()
    result[:, col_i] = replacement
    return result


def string2sympy(equation_str: str) -> sp.Poly:
    """
    Parse a degree-two equation into a SymPy polynomial.

    Args:
        equation_str: str
            Equation containing one equals sign and variables x, y, and z.
        return: sympy.Poly
            Expanded degree-two polynomial.
    """

    return QuadricParser().parse(equation_str)


def get_matrices(expr: sp.Poly) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Convert a polynomial to the legacy matrix tuple.

    Args:
        expr: sympy.Poly
            Degree-two polynomial in x, y, and z.
        return: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Homogeneous matrix, quadratic block, and linear column vector.
    """

    matrices = QuadricParser().matrices_from_polynomial(expr)
    return matrices.homogeneous, matrices.quadratic, matrices.linear


def expr2matrices(eq: str) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Parse an equation and return the legacy matrix tuple.

    Args:
        eq: str
            Degree-two equation in x, y, and z.
        return: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Homogeneous matrix, quadratic block, and linear column vector.
    """

    matrices = QuadricParser().parse_matrices(eq)
    return matrices.homogeneous, matrices.quadratic, matrices.linear


__all__ = [
    "assign_A_to_overA",
    "assign_b_to_overA",
    "clean_near_zero",
    "expr2matrices",
    "get_matrices",
    "string2sympy",
    "substitute_col",
]
