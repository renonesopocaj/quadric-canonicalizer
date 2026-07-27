"""Public numerical API. Run its checks with ``python -m pytest tests -q``."""

from src.numerical.classifier import NotAQuadricError, QuadricClassifier
from src.numerical.models import (
    AffineTransformation,
    CanonicalizationResult,
    QuadricMatrices,
    QuadricType,
    TransformationKind,
)
from src.numerical.parser import QuadricParser
from src.numerical.canonicalize import QuadricCanonicalizer, canonize_quadric

__all__ = [
    "CanonicalizationResult",
    "AffineTransformation",
    "NotAQuadricError",
    "QuadricCanonicalizer",
    "QuadricClassifier",
    "QuadricMatrices",
    "QuadricParser",
    "QuadricType",
    "TransformationKind",
    "canonize_quadric",
]
