"""
Define data transferred between graphics builders and scenes.

These models do not import Manim, so configuration tests can run without the
optional rendering dependency. Run them with ``python -m pytest tests/test_graphics_models.py -q``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.numerical.models import FloatArray, QuadricType


@dataclass(frozen=True, slots=True)
class RenderSettings:
    """
    Store validated output settings for one Manim render.

    Args:
        quality: str
            User-facing quality selection from "1" through "4".
        output_path: pathlib.Path
            Directory that receives rendered media.
        return: RenderSettings
            Immutable rendering configuration.
    """

    quality: str
    output_path: Path

    @property
    def manim_quality(self) -> str:
        """Translate the user-facing quality code to Manim's configuration name."""

        qualities = {
            "1": "low_quality",
            "2": "medium_quality",
            "3": "high_quality",
            "4": "production_quality",
        }
        try:
            return qualities[self.quality]
        except KeyError as error:
            raise ValueError(f"quality must be one of {tuple(qualities)}") from error


@dataclass(frozen=True, slots=True)
class TextOverlayGroups:
    """Store the five Manim groups that describe transformation text states."""

    initial: Any
    first_transformation: Any
    middle: Any
    second_transformation: Any
    final: Any


@dataclass(frozen=True, slots=True)
class SurfaceBuild:
    """Store a constructed Manim surface and its required axis distance."""

    surface: Any
    axis_distance: float


@dataclass(frozen=True, slots=True)
class SurfaceParameters:
    """Extract canonical diagonal coefficients and axis scales once per surface."""

    quadric_type: QuadricType
    matrix: FloatArray
    quadratic_coefficients: tuple[float, float, float]
    constant: float
    axis_scales: tuple[float, float, float]

    @classmethod
    def from_matrix(cls, quadric_type: QuadricType, matrix: FloatArray) -> SurfaceParameters:
        """Validate a canonical matrix and derive finite axis scales."""

        if matrix.shape != (4, 4):
            raise ValueError("canonical surface matrix must have shape (4, 4)")
        diagonal = np.diag(matrix)
        coefficients = (float(diagonal[0]), float(diagonal[1]), float(diagonal[2]))
        constant = float(matrix[3, 3])
        numerator = constant if not np.isclose(constant, 0) else 1.0
        scales = (
            0.0 if np.isclose(coefficients[0], 0) else float(np.sqrt(abs(numerator / coefficients[0]))),
            0.0 if np.isclose(coefficients[1], 0) else float(np.sqrt(abs(numerator / coefficients[1]))),
            0.0 if np.isclose(coefficients[2], 0) else float(np.sqrt(abs(numerator / coefficients[2]))),
        )
        return cls(
            quadric_type=quadric_type,
            matrix=matrix,
            quadratic_coefficients=coefficients,
            constant=constant,
            axis_scales=scales,
        )
