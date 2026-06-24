"""Verify canonicalization with ``python -m pytest tests/test_transformer.py -q``."""

import numpy as np
import pytest

from src.numerical.models import CanonicalizationResult, QuadricType
from src.numerical.canonicalize import canonize_quadric


@pytest.mark.parametrize(
    ("equation", "expected_type", "centered"),
    [
        ("2*(x-1)**2 + 3*(y+2)**2 + 4*(z-3)**2 = 1", QuadricType.REAL_ELLIPSOID, True),
        ("(x-1)**2 + (y+2)**2 - z = 0", QuadricType.ELLIPTIC_PARABOLOID, False),
        ("x**2 + y**2 = -1", QuadricType.COMPLEX_ELLIPTIC_CYLINDER, False),
        ("x**2 = 1", QuadricType.REAL_PARALLEL_PLANES, False),
        ("x**2 - y = 0", QuadricType.PARABOLIC_CYLINDER, False),
    ],
)
def test_canonicalizer_returns_typed_canonical_form(equation: str, expected_type: QuadricType, centered: bool) -> None:
    result = canonize_quadric(equation)

    assert isinstance(result, CanonicalizationResult)
    assert result.quadric_type is expected_type
    assert result.centered is centered
    np.testing.assert_allclose(result.rotation_matrix.T @ result.rotation_matrix, np.eye(3), atol=2e-2)
    np.testing.assert_allclose(result.final_matrix, result.final_matrix.T, atol=1e-12)
    assert np.count_nonzero(np.triu(result.final_matrix[:3, :3], k=1)) == 0


def test_centered_transformation_reconstructs_final_matrix() -> None:
    result = canonize_quadric("2*(x-1)**2 + 3*(y+2)**2 + 4*(z-3)**2 = 1")
    transform = np.eye(4)
    transform[:3, :3] = result.rotation_matrix
    transform[:3, 3] = -result.translation_vector.reshape(3)

    reconstructed = transform.T @ result.initial_matrix @ transform
    np.testing.assert_allclose(reconstructed, result.final_matrix, atol=3e-2)


def test_non_centered_transformations_reconstruct_middle_and_final_matrices() -> None:
    result = canonize_quadric("(x-1)**2 + (y+2)**2 - z = 0")
    rotation = np.eye(4)
    rotation[:3, :3] = result.rotation_matrix
    translation = np.eye(4)
    translation[:3, 3] = result.translation_vector

    reconstructed_middle = rotation.T @ result.initial_matrix @ rotation
    reconstructed_final = translation.T @ reconstructed_middle @ translation
    np.testing.assert_allclose(reconstructed_middle, result.middle_matrix, atol=3e-2)
    np.testing.assert_allclose(reconstructed_final, result.final_matrix, atol=3e-2)
