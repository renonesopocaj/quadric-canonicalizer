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

from src import AffineTransformation, CanonicalizationResult, QuadricType
from src.numerical.models import FloatArray


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
class Bounds3D:
    """
    Store an axis-aligned finite three-dimensional bounding box.

    Args:
        minimum: numpy.ndarray
            Minimum x, y, and z coordinates.
        maximum: numpy.ndarray
            Maximum x, y, and z coordinates.
        return: Bounds3D
            Immutable bounds used for surfaces, axes, and camera fitting.
    """

    minimum: FloatArray
    maximum: FloatArray

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=np.float64).copy()
        maximum = np.asarray(self.maximum, dtype=np.float64).copy()
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("bounds minimum and maximum must have shape (3,)")
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError("bounds must contain finite values")
        if np.any(maximum < minimum):
            raise ValueError("bounds maximum must be greater than or equal to minimum")
        minimum.setflags(write=False)
        maximum.setflags(write=False)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @classmethod
    def from_points(cls, points: FloatArray) -> Bounds3D:
        """Return tight axis-aligned bounds for an explicit ``(n, 3)`` point array."""

        point_array = np.asarray(points, dtype=np.float64)
        if point_array.ndim != 2 or point_array.shape[1] != 3 or point_array.shape[0] == 0:
            raise ValueError("points must have shape (n, 3) with at least one row")
        return cls(minimum=np.min(point_array, axis=0), maximum=np.max(point_array, axis=0))

    @property
    def center(self) -> FloatArray:
        """Return the center point of the bounds."""

        return (self.minimum + self.maximum) / 2.0

    @property
    def corners(self) -> FloatArray:
        """Return all eight corners as an ``(8, 3)`` array."""

        return np.array(
            [
                [x_value, y_value, z_value]
                for x_value in (self.minimum[0], self.maximum[0])
                for y_value in (self.minimum[1], self.maximum[1])
                for z_value in (self.minimum[2], self.maximum[2])
            ],
            dtype=np.float64,
        )

    def transformed(self, transform: AffineTransformation) -> Bounds3D:
        """Return bounds after applying one active affine point transformation."""

        transformed_points = self.corners @ transform.linear_map.T + transform.offset
        return Bounds3D.from_points(transformed_points)

    def transformed_by_homogeneous(self, transform: FloatArray) -> Bounds3D:
        """Return bounds after applying one explicit homogeneous point matrix."""

        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("homogeneous point transform must have shape (4, 4)")
        homogeneous_corners = np.column_stack([self.corners, np.ones(8, dtype=np.float64)])
        transformed = homogeneous_corners @ matrix.T
        return Bounds3D.from_points(transformed[:, :3])

    def union(self, other: Bounds3D) -> Bounds3D:
        """Return the smallest bounds containing this box and another box."""

        return Bounds3D(
            minimum=np.minimum(self.minimum, other.minimum),
            maximum=np.maximum(self.maximum, other.maximum),
        )

    def contains(self, points: FloatArray) -> bool:
        """Return whether every explicit point lies inside the bounds."""

        point_array = np.asarray(points, dtype=np.float64)
        if point_array.ndim != 2 or point_array.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        return bool(np.all(point_array >= self.minimum) and np.all(point_array <= self.maximum))


@dataclass(frozen=True, slots=True)
class CameraFraming:
    """
    Store scale-aware camera settings for one finite surface stage.

    Args:
        frame_center: numpy.ndarray
            World-space point placed at the center of the frame.
        zoom: float
            Manim camera zoom required for the requested occupancy.
        focal_distance: float
            Scale-aware perspective distance that keeps the patch in front of
            the camera.
        return: CameraFraming
            Camera settings accepted by ``ThreeDScene``.
    """

    frame_center: FloatArray
    zoom: float
    focal_distance: float

    @classmethod
    def fit(
        cls,
        bounds: Bounds3D,
        frame_width: float,
        frame_height: float,
        phi_radians: float,
        fill_ratio: float,
    ) -> CameraFraming:
        """
        Fit bounds for every azimuth at a fixed polar camera angle.

        Args:
            bounds: Bounds3D
                Finite geometry bounds for one animation stage.
            frame_width: float
                Manim frame width in scene units.
            frame_height: float
                Manim frame height in scene units.
            phi_radians: float
                Camera polar angle in radians.
            fill_ratio: float
                Fraction of each frame dimension available to geometry.
            return: CameraFraming
                Center, zoom, and focal distance for the stage.
        """

        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("camera frame dimensions must be positive")
        if not 0 < fill_ratio < 1:
            raise ValueError("fill_ratio must be strictly between zero and one")
        relative_corners = bounds.corners - bounds.center
        radial_xy = np.linalg.norm(relative_corners[:, :2], axis=1)
        horizontal_radius = float(np.max(radial_xy))
        vertical_radii = (
            abs(np.cos(phi_radians)) * radial_xy
            + abs(np.sin(phi_radians)) * np.abs(relative_corners[:, 2])
        )
        vertical_radius = float(np.max(vertical_radii))
        if horizontal_radius == 0 and vertical_radius == 0:
            raise ValueError("camera cannot frame bounds containing only one point")
        horizontal_zoom = (
            np.inf if horizontal_radius == 0 else fill_ratio * frame_width / (2.0 * horizontal_radius)
        )
        vertical_zoom = (
            np.inf if vertical_radius == 0 else fill_ratio * frame_height / (2.0 * vertical_radius)
        )
        zoom = float(min(horizontal_zoom, vertical_zoom))
        euclidean_radius = float(np.max(np.linalg.norm(relative_corners, axis=1)))
        focal_distance = max(4.0 * euclidean_radius, 1e-6)
        return cls(frame_center=bounds.center, zoom=zoom, focal_distance=focal_distance)


@dataclass(frozen=True, slots=True)
class AxisLayout:
    """
    Store world-unit axis ranges and readable tick intervals.

    Args:
        ranges: tuple[tuple[float, float, float], ...]
            Inclusive minimum, maximum, and tick step for x, y, and z.
        return: AxisLayout
            Axis configuration whose physical lengths equal numerical spans.
    """

    ranges: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    @classmethod
    def from_stage_bounds(cls, stage_bounds: tuple[Bounds3D, Bounds3D, Bounds3D]) -> AxisLayout:
        """
        Cover all animation stages and the coordinate origin.

        Args:
            stage_bounds: tuple[Bounds3D, Bounds3D, Bounds3D]
                Initial, middle, and final geometry bounds.
            return: AxisLayout
                Rounded ranges with approximately eight major intervals.
        """

        combined = stage_bounds[0].union(stage_bounds[1]).union(stage_bounds[2])
        minimum = np.minimum(combined.minimum, np.zeros(3, dtype=np.float64))
        maximum = np.maximum(combined.maximum, np.zeros(3, dtype=np.float64))
        largest_span = float(np.max(maximum - minimum))
        if largest_span == 0:
            raise ValueError("axis layout requires non-degenerate geometry bounds")
        ranges: list[tuple[float, float, float]] = []
        for axis in range(3):
            axis_minimum = float(minimum[axis])
            axis_maximum = float(maximum[axis])
            if axis_maximum == axis_minimum:
                axis_minimum -= largest_span * 0.05
                axis_maximum += largest_span * 0.05
            span = axis_maximum - axis_minimum
            padding = 0.08 * span
            padded_minimum = axis_minimum - padding
            padded_maximum = axis_maximum + padding
            tick_step = cls._nice_tick_step(padded_maximum - padded_minimum)
            rounded_minimum = float(np.floor(padded_minimum / tick_step) * tick_step)
            rounded_maximum = float(np.ceil(padded_maximum / tick_step) * tick_step)
            ranges.append((rounded_minimum, rounded_maximum, tick_step))
        return cls(ranges=(ranges[0], ranges[1], ranges[2]))

    @staticmethod
    def _nice_tick_step(span: float) -> float:
        """Return a 1/2/5-scaled tick interval for roughly eight divisions."""

        rough_step = span / 8.0
        exponent = float(np.floor(np.log10(rough_step)))
        magnitude = 10.0**exponent
        normalized = rough_step / magnitude
        if normalized <= 1.0:
            multiplier = 1.0
        elif normalized <= 2.0:
            multiplier = 2.0
        elif normalized <= 5.0:
            multiplier = 5.0
        else:
            multiplier = 10.0
        return multiplier * magnitude


@dataclass(frozen=True, slots=True)
class SurfaceBuild:
    """Store a constructed Manim surface and its finite canonical bounds."""

    surface: Any
    bounds: Bounds3D


@dataclass(frozen=True, slots=True)
class SurfaceParameters:
    """Extract canonical diagonal coefficients and axis scales once per surface."""

    quadric_type: QuadricType
    matrix: FloatArray
    normalized_matrix: FloatArray
    quadratic_coefficients: tuple[float, float, float]
    linear_coefficients: tuple[float, float, float]
    constant: float
    axis_scales: tuple[float, float, float]

    @classmethod
    def from_matrix(cls, quadric_type: QuadricType, matrix: FloatArray) -> SurfaceParameters:
        """Validate a canonical matrix and derive scale-invariant coefficients."""

        if matrix.shape != (4, 4):
            raise ValueError("canonical surface matrix must have shape (4, 4)")
        matrix_scale = float(np.max(np.abs(matrix)))
        if matrix_scale == 0:
            raise ValueError("canonical surface matrix cannot be identically zero")
        normalized_matrix = matrix / matrix_scale
        diagonal = np.diag(normalized_matrix)
        coefficients = (float(diagonal[0]), float(diagonal[1]), float(diagonal[2]))
        normalized_constant = float(diagonal[3])
        linear = tuple(float(value) for value in normalized_matrix[:3, 3])
        numerator = normalized_constant if normalized_constant != 0.0 else 1.0
        scales = (
            0.0 if coefficients[0] == 0.0 else float(np.sqrt(abs(numerator / coefficients[0]))),
            0.0 if coefficients[1] == 0.0 else float(np.sqrt(abs(numerator / coefficients[1]))),
            0.0 if coefficients[2] == 0.0 else float(np.sqrt(abs(numerator / coefficients[2]))),
        )
        return cls(
            quadric_type=quadric_type,
            matrix=matrix,
            normalized_matrix=normalized_matrix,
            quadratic_coefficients=coefficients,
            linear_coefficients=(linear[0], linear[1], linear[2]),
            constant=normalized_constant,
            axis_scales=scales,
        )

    @classmethod
    def from_result(cls, result: CanonicalizationResult) -> SurfaceParameters:
        """Build surface parameters from the public canonicalization result."""

        return cls.from_matrix(result.quadric_type, result.final_matrix)


@dataclass(frozen=True, slots=True)
class RenderPlan:
    """
    Adapt one public numerical result into the complete graphics contract.

    Args:
        result: CanonicalizationResult
            Single numerical API object used by every graphics builder.
        surface_parameters: SurfaceParameters
            Scale-invariant canonical surface coefficients.
        transformation_steps: tuple[AffineTransformation, AffineTransformation]
            Ordered active transforms from initial to final geometry.
        return: RenderPlan
            Immutable input for surface, overlay, axes, and scene builders.
    """

    result: CanonicalizationResult
    surface_parameters: SurfaceParameters
    transformation_steps: tuple[AffineTransformation, AffineTransformation]

    @classmethod
    def from_result(cls, result: CanonicalizationResult) -> RenderPlan:
        """Create the graphics adapter exclusively from the public numerical result."""

        return cls(
            result=result,
            surface_parameters=SurfaceParameters.from_result(result),
            transformation_steps=result.transformation_steps,
        )

    def stage_bounds(self, canonical_bounds: Bounds3D) -> tuple[Bounds3D, Bounds3D, Bounds3D]:
        """
        Reconstruct initial, middle, and final bounds from the canonical patch.

        Args:
            canonical_bounds: Bounds3D
                Bounds of the final canonical surface patch.
            return: tuple[Bounds3D, Bounds3D, Bounds3D]
                Bounds in initial, middle, and final animation order.
        """

        first_step, second_step = self.transformation_steps
        middle_bounds = canonical_bounds.transformed_by_homogeneous(second_step.inverse_homogeneous_matrix)
        initial_bounds = middle_bounds.transformed_by_homogeneous(first_step.inverse_homogeneous_matrix)
        return initial_bounds, middle_bounds, canonical_bounds
