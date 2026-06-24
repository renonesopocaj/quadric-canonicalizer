"""Verify parsing contracts with ``python -m pytest tests/test_parser.py -q``."""

import numpy as np
import pytest

from src.numerical.parser import QuadricParser


def test_parser_builds_symmetric_homogeneous_matrix() -> None:
    matrices = QuadricParser().parse_matrices("2x**2 + 4xy + 6z + 7 = 0")

    np.testing.assert_allclose(matrices.quadratic, np.array([[2, 2, 0], [2, 0, 0], [0, 0, 0]]))
    np.testing.assert_allclose(matrices.linear, np.array([[0], [0], [3]]))
    np.testing.assert_allclose(matrices.homogeneous, matrices.homogeneous.T)
    assert matrices.homogeneous[3, 3] == 7


@pytest.mark.parametrize("equation", ["x**2 + y**2", "x + y = 0", "x**3 = 1", "x**2 + w = 0"])
def test_parser_rejects_non_quadric_equations(equation: str) -> None:
    with pytest.raises(ValueError):
        QuadricParser().parse(equation)
