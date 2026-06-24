"""Verify classifier branches with ``python -m pytest tests/test_classifier.py -q``."""

import pytest

from src.numerical.classifier import expr2classification
from src.numerical.models import QuadricType


@pytest.mark.parametrize(
    ("equation", "expected"),
    [
        ("x**2 + y**2 + z**2 = 1", QuadricType.REAL_ELLIPSOID),
        ("x**2 + y**2 + z**2 = -1", QuadricType.COMPLEX_ELLIPSOID),
        ("x**2 + y**2 - z**2 = 1", QuadricType.ONE_SHEET_HYPERBOLOID),
        ("x**2 + y**2 - z**2 = -1", QuadricType.TWO_SHEET_HYPERBOLOID),
        ("x**2 + y**2 - z**2 = 0", QuadricType.REAL_CONE),
        ("x**2 + y**2 + z**2 = 0", QuadricType.COMPLEX_CONE),
        ("x**2 + y**2 - z = 0", QuadricType.ELLIPTIC_PARABOLOID),
        ("x**2 - y**2 - z = 0", QuadricType.HYPERBOLIC_PARABOLOID),
        ("x**2 + y**2 = 1", QuadricType.REAL_ELLIPTIC_CYLINDER),
        ("x**2 + y**2 = -1", QuadricType.COMPLEX_ELLIPTIC_CYLINDER),
        ("x**2 - y**2 = 1", QuadricType.HYPERBOLIC_CYLINDER),
        ("x**2 - y**2 = 0", QuadricType.REAL_INTERSECTING_PLANES),
        ("x**2 + y**2 = 0", QuadricType.COMPLEX_INTERSECTING_PLANES),
        ("x**2 - y = 0", QuadricType.PARABOLIC_CYLINDER),
        ("x**2 = 1", QuadricType.REAL_PARALLEL_PLANES),
        ("x**2 = -1", QuadricType.COMPLEX_PARALLEL_PLANES),
        ("x**2 = 0", QuadricType.DOUBLE_PLANE),
    ],
)
def test_classifier_covers_all_quadric_types(equation: str, expected: QuadricType) -> None:
    assert expr2classification(equation) is expected
