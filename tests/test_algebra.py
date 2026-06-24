"""Verify algebra helpers with ``python -m pytest tests/test_algebra.py -q``."""

import numpy as np
import sympy as sp

from src.numerical.numerical_helpers import assign_linear_block, assign_quadratic_block, clean_near_zero, expression_from_matrix
from src.numerical.symbols import x, y, z


def test_matrix_block_updates_are_symmetric_and_non_mutating() -> None:
    original = np.zeros((4, 4))
    quadratic = np.diag([1.0, 2.0, 3.0])

    with_quadratic = assign_quadratic_block(original, quadratic)
    result = assign_linear_block(with_quadratic, np.array([4.0, 5.0, 6.0]))

    np.testing.assert_array_equal(original, np.zeros((4, 4)))
    np.testing.assert_array_equal(result[:3, 3], result[3, :3])
    expected = x**2 + 2 * y**2 + 3 * z**2 + 8 * x + 10 * y + 12 * z
    assert sp.simplify(expression_from_matrix(result) - expected) == 0


def test_clean_near_zero_uses_explicit_threshold() -> None:
    result = clean_near_zero(np.array([1e-11, 1e-8]), 1e-10)
    np.testing.assert_array_equal(result, np.array([0.0, 1e-8]))
