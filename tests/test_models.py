"""Verify data contracts with ``python -m pytest tests/test_models.py -q``."""

import numpy as np
import pytest
import sympy as sp
from pydantic import ValidationError

from src.numerical.models import CanonicalizationResult, QuadricMatrices, QuadricType


def test_quadric_matrices_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="homogeneous"):
        QuadricMatrices(homogeneous=np.zeros((3, 3)), quadratic=np.zeros((3, 3)), linear=np.zeros((3, 1)))


def test_result_validates_shapes_and_preserves_legacy_access() -> None:
    result = CanonicalizationResult(
        quadric_type=QuadricType.REAL_ELLIPSOID,
        centered=True,
        initial_matrix=np.eye(4),
        middle_matrix=np.eye(4),
        final_matrix=np.eye(4),
        translation_vector=np.zeros(3),
        rotation_matrix=np.eye(3),
        initial_equation=sp.sympify("x**2"),
        middle_equation=sp.sympify("x**2"),
        final_equation=sp.sympify("x**2"),
    )

    assert result["quadric type"] is QuadricType.REAL_ELLIPSOID
    assert result.translation_vector.shape == (3,)
    assert result.initial_matrix.flags.writeable is False


def test_result_rejects_invalid_rotation_shape() -> None:
    with pytest.raises(ValidationError, match="rotation_matrix"):
        CanonicalizationResult(
            quadric_type=QuadricType.REAL_ELLIPSOID,
            centered=True,
            initial_matrix=np.eye(4),
            middle_matrix=np.eye(4),
            final_matrix=np.eye(4),
            translation_vector=np.zeros(3),
            rotation_matrix=np.eye(4),
            initial_equation=sp.sympify("x**2"),
            middle_equation=sp.sympify("x**2"),
            final_equation=sp.sympify("x**2"),
        )
