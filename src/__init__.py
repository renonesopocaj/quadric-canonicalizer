"""Quadric canonicalization package. Run checks with ``python -m pytest -q``."""

from src.numerical import (
    AffineTransformation,
    CanonicalizationResult,
    QuadricType,
    TransformationKind,
    canonize_quadric,
)

__all__ = [
    "AffineTransformation",
    "CanonicalizationResult",
    "QuadricType",
    "TransformationKind",
    "canonize_quadric",
]
