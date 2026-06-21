"""Provide legacy name lookup. Run checks with ``python -m pytest tests/test_classifier.py -q``."""

from types import MappingProxyType

from src.numerical.models import QuadricType

ENUM_QUADRICS = MappingProxyType(
    {
        "real ellipsoid": QuadricType.REAL_ELLIPSOID,
        "complex ellipsoid": QuadricType.COMPLEX_ELLIPSOID,
        "one sheet hyperboloid": QuadricType.ONE_SHEET_HYPERBOLOID,
        "two sheets hyperboloid": QuadricType.TWO_SHEET_HYPERBOLOID,
        "real cone": QuadricType.REAL_CONE,
        "complex cone": QuadricType.COMPLEX_CONE,
        "elliptic paraboloid": QuadricType.ELLIPTIC_PARABOLOID,
        "hyperbolic paraboloid": QuadricType.HYPERBOLIC_PARABOLOID,
        "real elliptic cylinder": QuadricType.REAL_ELLIPTIC_CYLINDER,
        "complex elliptic cylinder": QuadricType.COMPLEX_ELLIPTIC_CYLINDER,
        "hyperbolic cylinder": QuadricType.HYPERBOLIC_CYLINDER,
        "real intersecting planes": QuadricType.REAL_INTERSECTING_PLANES,
        "complex intersecting planes": QuadricType.COMPLEX_INTERSECTING_PLANES,
        "parabolic cylinder": QuadricType.PARABOLIC_CYLINDER,
        "real parallel planes": QuadricType.REAL_PARALLEL_PLANES,
        "complex parallel planes": QuadricType.COMPLEX_PARALLEL_PLANES,
        "double plane": QuadricType.DOUBLE_PLANE,
    }
)

__all__ = ["ENUM_QUADRICS", "QuadricType"]
