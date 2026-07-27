"""
Build styled Manim mobjects from pure canonical surface specifications.

Rendering requires the optional graphics dependencies. Pure parameterization
checks run with ``python -m pytest tests/test_surface_spec.py -q``.
"""

from __future__ import annotations

from typing import Any

import manim as mn

from src import CanonicalizationResult, QuadricType
from src.graphics.models import SurfaceBuild, SurfaceParameters
from src.graphics.surface_spec import SurfaceSpec, SurfaceSpecFactory, UnsupportedSurfaceError
from src.numerical.models import FloatArray


SURFACE_OPACITY = 0.8
SURFACE_COLOR = mn.BLUE
SURFACE_STROKE_COLOR = mn.WHITE
SURFACE_STROKE_WIDTH = 0.5
CENTER_COLOR = mn.RED
CENTER_RADIUS_RATIO = 0.015


# Compatibility names retained for callers of the original graphics module.
NotSupportedException = UnsupportedSurfaceError


class NotAQuadricException(ValueError):
    """Indicate that a surface request does not identify a quadric."""


class QuadricSurfaceFactory:
    """Construct Manim geometry from the single public numerical result."""

    spec_factory: SurfaceSpecFactory

    def __init__(self) -> None:
        self.spec_factory = SurfaceSpecFactory()

    def create(self, result: CanonicalizationResult) -> SurfaceBuild:
        """
        Build the canonical Manim surface represented by a numerical result.

        Args:
            result: CanonicalizationResult
                Single public numerical API object.
            return: SurfaceBuild
                Styled Manim group and its finite canonical bounds.
        """

        return self._build(self.spec_factory.create(result))

    def _build(self, spec: SurfaceSpec) -> SurfaceBuild:
        """Convert one pure surface specification into a styled Manim group."""

        surfaces = mn.VGroup()
        for patch in spec.patches:
            surface = mn.Surface(
                patch.point_function,
                u_range=patch.u_range,
                v_range=patch.v_range,
                resolution=patch.resolution,
            )
            surface.set_style(
                fill_opacity=SURFACE_OPACITY,
                fill_color=SURFACE_COLOR,
                stroke_color=SURFACE_STROKE_COLOR,
                stroke_width=SURFACE_STROKE_WIDTH,
            )
            surfaces.add(surface)

        center = mn.Dot3D(
            point=mn.ORIGIN,
            radius=CENTER_RADIUS_RATIO * spec.characteristic_length,
            color=CENTER_COLOR,
        )
        surfaces.add(center)
        return SurfaceBuild(surface=surfaces, bounds=spec.bounds)


def create_surface(quadric_type: QuadricType | int, final_matrix: FloatArray) -> tuple[Any, float]:
    """
    Build a canonical surface through the legacy two-argument function API.

    New code should pass ``CanonicalizationResult`` to
    :class:`QuadricSurfaceFactory`.

    Args:
        quadric_type: QuadricType or int
            Canonical quadric classification.
        final_matrix: numpy.ndarray
            Canonical homogeneous matrix.
        return: tuple[Any, float]
            Manim surface group and finite patch characteristic length.
    """

    parameters = SurfaceParameters.from_matrix(QuadricType(quadric_type), final_matrix)
    spec = SurfaceSpecFactory().create_from_parameters(parameters)
    build = QuadricSurfaceFactory()._build(spec)
    return build.surface, spec.characteristic_length


__all__ = [
    "NotAQuadricException",
    "NotSupportedException",
    "QuadricSurfaceFactory",
    "create_surface",
]
