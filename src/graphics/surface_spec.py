"""
Define equation-exact, scale-invariant finite patches for real quadrics.

Run the pure geometry checks with
``python -m pytest tests/test_surface_spec.py -q``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from src import CanonicalizationResult, QuadricType
from src.graphics.models import Bounds3D, SurfaceParameters
from src.numerical.models import FloatArray


ArrayLike = npt.ArrayLike
PointFunction = Callable[[ArrayLike, ArrayLike], FloatArray]
SURFACE_TOLERANCE = 10.0 * np.finfo(np.float64).eps
PRESENTATION_RADIUS = 4.0
UNBOUNDED_SCALE_FACTOR = 3.0
PATCH_RESOLUTION = (30, 30)
BOUNDS_SAMPLES_PER_AXIS = 181


class UnsupportedSurfaceError(ValueError):
    """Indicate that a complex quadric has no real surface patch to render."""


def _stack_coordinates(x_value: ArrayLike, y_value: ArrayLike, z_value: ArrayLike) -> FloatArray:
    """Broadcast and stack three coordinate arrays along their last axis."""

    broadcast = np.broadcast_arrays(x_value, y_value, z_value)
    return np.asarray(np.stack(broadcast, axis=-1), dtype=np.float64)


def _assemble_coordinates(axis_values: tuple[tuple[int, ArrayLike], ...]) -> FloatArray:
    """Place three explicitly indexed coordinate arrays in x-y-z order."""

    if len(axis_values) != 3:
        raise ValueError("exactly three axis values are required")
    ordered: list[ArrayLike] = [0.0, 0.0, 0.0]
    seen_axes: set[int] = set()
    for axis, values in axis_values:
        if axis not in (0, 1, 2) or axis in seen_axes:
            raise ValueError("axis values must identify x, y, and z exactly once")
        ordered[axis] = values
        seen_axes.add(axis)
    return _stack_coordinates(ordered[0], ordered[1], ordered[2])


def _active_indices(values: tuple[float, float, float]) -> tuple[int, ...]:
    """Return indices whose normalized coefficients are numerically non-zero."""

    return tuple(index for index, value in enumerate(values) if abs(value) > SURFACE_TOLERANCE)


def _require_count(indices: tuple[int, ...], expected: int, description: str) -> None:
    """Fail explicitly when a canonical matrix violates a strategy contract."""

    if len(indices) != expected:
        raise ValueError(f"{description} requires {expected} active axes; received {indices}")


def _expanded_bounds(points: FloatArray) -> Bounds3D:
    """Return sampled bounds expanded only by floating-point roundoff padding."""

    bounds = Bounds3D.from_points(points)
    extent = float(np.max(bounds.maximum - bounds.minimum))
    padding = float(max(extent * 1e-12, float(np.finfo(np.float64).eps)))
    return Bounds3D(minimum=bounds.minimum - padding, maximum=bounds.maximum + padding)


@dataclass(frozen=True, slots=True)
class ParametricPatch:
    """
    Store one finite two-parameter patch independent of Manim.

    Args:
        point_function: Callable
            Vectorized function returning trailing x-y-z coordinates.
        u_range: tuple[float, float]
            Inclusive first-parameter interval.
        v_range: tuple[float, float]
            Inclusive second-parameter interval.
        resolution: tuple[int, int]
            Manim mesh resolution.
        return: ParametricPatch
            Pure parametric surface patch.
    """

    point_function: PointFunction
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    resolution: tuple[int, int]

    def sample_points(self, samples_per_axis: int) -> FloatArray:
        """
        Sample the complete rectangular parameter domain.

        Args:
            samples_per_axis: int
                Number of uniformly spaced samples on each parameter axis.
            return: numpy.ndarray
                Flattened ``(samples_per_axis**2, 3)`` point array.
        """

        if samples_per_axis < 2:
            raise ValueError("samples_per_axis must be at least two")
        u_values = np.linspace(self.u_range[0], self.u_range[1], samples_per_axis)
        v_values = np.linspace(self.v_range[0], self.v_range[1], samples_per_axis)
        u_grid, v_grid = np.meshgrid(u_values, v_values, indexing="ij")
        points = np.asarray(self.point_function(u_grid, v_grid), dtype=np.float64)
        if points.shape != (samples_per_axis, samples_per_axis, 3):
            raise ValueError(
                "point_function must return shape "
                f"({samples_per_axis}, {samples_per_axis}, 3); received {points.shape}"
            )
        return points.reshape(-1, 3)


@dataclass(frozen=True, slots=True)
class SurfaceSpec:
    """
    Store all finite canonical patches and their sampled bounds.

    Args:
        patches: tuple[ParametricPatch, ...]
            One or more equation-exact finite patches.
        bounds: Bounds3D
            Bounds containing every patch.
        characteristic_length: float
            Positive geometry scale used for marker styling.
        return: SurfaceSpec
            Manim-independent canonical surface specification.
    """

    patches: tuple[ParametricPatch, ...]
    bounds: Bounds3D
    characteristic_length: float

    def sample_points(self, samples_per_axis: int) -> FloatArray:
        """Return concatenated samples from every patch."""

        if not self.patches:
            raise ValueError("a surface specification must contain at least one patch")
        samples = tuple(patch.sample_points(samples_per_axis) for patch in self.patches)
        return np.concatenate(samples, axis=0)


class SurfaceSpecFactory:
    """Select one canonical parameterization strategy by typed quadric family."""

    _builders: dict[QuadricType, Callable[[SurfaceParameters], tuple[ParametricPatch, ...]]]

    def __init__(self) -> None:
        self._builders = {
            QuadricType.REAL_ELLIPSOID: self._ellipsoid,
            QuadricType.ONE_SHEET_HYPERBOLOID: self._one_sheet_hyperboloid,
            QuadricType.TWO_SHEET_HYPERBOLOID: self._two_sheet_hyperboloid,
            QuadricType.REAL_CONE: self._cone,
            QuadricType.ELLIPTIC_PARABOLOID: self._elliptic_paraboloid,
            QuadricType.HYPERBOLIC_PARABOLOID: self._hyperbolic_paraboloid,
            QuadricType.REAL_ELLIPTIC_CYLINDER: self._elliptic_cylinder,
            QuadricType.HYPERBOLIC_CYLINDER: self._hyperbolic_cylinder,
            QuadricType.REAL_INTERSECTING_PLANES: self._intersecting_planes,
            QuadricType.PARABOLIC_CYLINDER: self._parabolic_cylinder,
            QuadricType.REAL_PARALLEL_PLANES: self._parallel_planes,
            QuadricType.DOUBLE_PLANE: self._double_plane,
        }

    def create(self, result: CanonicalizationResult) -> SurfaceSpec:
        """
        Create a finite surface specification from the public numerical API.

        Args:
            result: CanonicalizationResult
                Canonical numerical result containing type and final matrix.
            return: SurfaceSpec
                Equation-exact finite patches and their camera bounds.
        """

        return self.create_from_parameters(SurfaceParameters.from_result(result))

    def create_from_parameters(self, parameters: SurfaceParameters) -> SurfaceSpec:
        """
        Create a finite specification from already-adapted surface parameters.

        Args:
            parameters: SurfaceParameters
                Validated canonical coefficients.
            return: SurfaceSpec
                Equation-exact finite patches and their camera bounds.
        """

        try:
            builder = self._builders[parameters.quadric_type]
        except KeyError as error:
            raise UnsupportedSurfaceError(
                f"{parameters.quadric_type.name.lower()} has no real surface to render"
            ) from error
        patches = builder(parameters)
        points = np.concatenate(
            tuple(patch.sample_points(BOUNDS_SAMPLES_PER_AXIS) for patch in patches),
            axis=0,
        )
        bounds = _expanded_bounds(points)
        characteristic_length = float(np.max(np.linalg.norm(points - bounds.center, axis=1)))
        if characteristic_length <= 0 or not np.isfinite(characteristic_length):
            raise ValueError("surface strategy produced a non-positive characteristic length")
        return SurfaceSpec(
            patches=patches,
            bounds=bounds,
            characteristic_length=characteristic_length,
        )

    def _ellipsoid(self, parameters: SurfaceParameters) -> tuple[ParametricPatch, ...]:
        radii = parameters.axis_scales
        if any(radius <= 0 for radius in radii):
            raise ValueError("a real ellipsoid requires three positive semi-axis lengths")

        def point(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            return _stack_coordinates(
                radii[0] * np.cos(u_value) * np.sin(v_value),
                radii[1] * np.sin(u_value) * np.sin(v_value),
                radii[2] * np.cos(v_value),
            )

        return (ParametricPatch(point, (0.0, 2.0 * np.pi), (0.0, np.pi), PATCH_RESOLUTION),)

    def _one_sheet_hyperboloid(self, parameters: SurfaceParameters) -> tuple[ParametricPatch, ...]:
        normalized = tuple(
            coefficient / -parameters.constant for coefficient in parameters.quadratic_coefficients
        )
        positive = tuple(index for index, value in enumerate(normalized) if value > SURFACE_TOLERANCE)
        negative = tuple(index for index, value in enumerate(normalized) if value < -SURFACE_TOLERANCE)
        _require_count(positive, 2, "one-sheet hyperboloid")
        _require_count(negative, 1, "one-sheet hyperboloid")
        radii = tuple(1.0 / np.sqrt(abs(value)) for value in normalized)
        cap = UNBOUNDED_SCALE_FACTOR * max(radii)
        limit = float(np.arcsinh(cap / radii[negative[0]]))

        def point(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            return _assemble_coordinates(
                (
                    (positive[0], radii[positive[0]] * np.cosh(v_value) * np.cos(u_value)),
                    (positive[1], radii[positive[1]] * np.cosh(v_value) * np.sin(u_value)),
                    (negative[0], radii[negative[0]] * np.sinh(v_value)),
                )
            )

        return (ParametricPatch(point, (0.0, 2.0 * np.pi), (-limit, limit), PATCH_RESOLUTION),)

    def _two_sheet_hyperboloid(self, parameters: SurfaceParameters) -> tuple[ParametricPatch, ...]:
        normalized = tuple(
            coefficient / -parameters.constant for coefficient in parameters.quadratic_coefficients
        )
        positive = tuple(index for index, value in enumerate(normalized) if value > SURFACE_TOLERANCE)
        negative = tuple(index for index, value in enumerate(normalized) if value < -SURFACE_TOLERANCE)
        _require_count(positive, 1, "two-sheet hyperboloid")
        _require_count(negative, 2, "two-sheet hyperboloid")
        radii = tuple(1.0 / np.sqrt(abs(value)) for value in normalized)
        cap = UNBOUNDED_SCALE_FACTOR * max(radii)
        limit = float(np.arcsinh(cap / max(radii[negative[0]], radii[negative[1]])))

        def positive_sheet(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            return self._two_sheet_point(u_value, v_value, 1.0, positive, negative, radii)

        def negative_sheet(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            return self._two_sheet_point(u_value, v_value, -1.0, positive, negative, radii)

        return (
            ParametricPatch(positive_sheet, (0.0, 2.0 * np.pi), (0.0, limit), PATCH_RESOLUTION),
            ParametricPatch(negative_sheet, (0.0, 2.0 * np.pi), (0.0, limit), PATCH_RESOLUTION),
        )

    def _two_sheet_point(
        self,
        u_value: ArrayLike,
        v_value: ArrayLike,
        sheet_sign: float,
        positive: tuple[int, ...],
        negative: tuple[int, ...],
        radii: tuple[float, ...],
    ) -> FloatArray:
        """Evaluate one sheet of a two-sheet hyperboloid."""

        return _assemble_coordinates(
            (
                (positive[0], sheet_sign * radii[positive[0]] * np.cosh(v_value)),
                (negative[0], radii[negative[0]] * np.sinh(v_value) * np.cos(u_value)),
                (negative[1], radii[negative[1]] * np.sinh(v_value) * np.sin(u_value)),
            )
        )

    def _cone(self, parameters: SurfaceParameters) -> tuple[ParametricPatch, ...]:
        coefficients = parameters.quadratic_coefficients
        positive = tuple(index for index, value in enumerate(coefficients) if value > SURFACE_TOLERANCE)
        negative = tuple(index for index, value in enumerate(coefficients) if value < -SURFACE_TOLERANCE)
        if len(positive) == 1 and len(negative) == 2:
            singleton = positive[0]
            paired = negative
        elif len(negative) == 1 and len(positive) == 2:
            singleton = negative[0]
            paired = positive
        else:
            raise ValueError("a real cone requires one quadratic sign opposite to the other two")

        def point(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            directions = _assemble_coordinates(
                (
                    (singleton, np.ones_like(np.asarray(u_value)) / np.sqrt(abs(coefficients[singleton]))),
                    (paired[0], np.cos(u_value) / np.sqrt(abs(coefficients[paired[0]]))),
                    (paired[1], np.sin(u_value) / np.sqrt(abs(coefficients[paired[1]]))),
                )
            )
            directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
            return np.asarray(directions * np.expand_dims(v_value, axis=-1), dtype=np.float64)

        return (
            ParametricPatch(
                point,
                (0.0, 2.0 * np.pi),
                (-PRESENTATION_RADIUS, PRESENTATION_RADIUS),
                PATCH_RESOLUTION,
            ),
        )

    def _elliptic_paraboloid(self, parameters: SurfaceParameters) -> tuple[ParametricPatch, ...]:
        quadratic_axes = _active_indices(parameters.quadratic_coefficients)
        linear_axes = _active_indices(parameters.linear_coefficients)
        _require_count(quadratic_axes, 2, "elliptic paraboloid")
        _require_count(linear_axes, 1, "elliptic paraboloid")
        axial = linear_axes[0]
        first, second = quadratic_axes
        linear = parameters.linear_coefficients[axial]
        focal_scales = (
            abs(linear / parameters.quadratic_coefficients[first]),
            abs(linear / parameters.quadratic_coefficients[second]),
        )
        height = UNBOUNDED_SCALE_FACTOR * max(focal_scales)
        first_extent = np.sqrt(2.0 * focal_scales[0] * height)
        second_extent = np.sqrt(2.0 * focal_scales[1] * height)

        def point(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            first_value = first_extent * v_value * np.cos(u_value)
            second_value = second_extent * v_value * np.sin(u_value)
            axial_value = -(
                parameters.quadratic_coefficients[first] * first_value**2
                + parameters.quadratic_coefficients[second] * second_value**2
            ) / (2.0 * linear)
            return _assemble_coordinates(
                ((first, first_value), (second, second_value), (axial, axial_value))
            )

        return (ParametricPatch(point, (0.0, 2.0 * np.pi), (0.0, 1.0), PATCH_RESOLUTION),)

    def _hyperbolic_paraboloid(self, parameters: SurfaceParameters) -> tuple[ParametricPatch, ...]:
        quadratic_axes = _active_indices(parameters.quadratic_coefficients)
        linear_axes = _active_indices(parameters.linear_coefficients)
        _require_count(quadratic_axes, 2, "hyperbolic paraboloid")
        _require_count(linear_axes, 1, "hyperbolic paraboloid")
        axial = linear_axes[0]
        first, second = quadratic_axes
        linear = parameters.linear_coefficients[axial]
        focal_scales = (
            abs(linear / parameters.quadratic_coefficients[first]),
            abs(linear / parameters.quadratic_coefficients[second]),
        )
        height = UNBOUNDED_SCALE_FACTOR * max(focal_scales)
        first_extent = np.sqrt(2.0 * focal_scales[0] * height)
        second_extent = np.sqrt(2.0 * focal_scales[1] * height)

        def point(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            first_value = first_extent * u_value
            second_value = second_extent * v_value
            axial_value = -(
                parameters.quadratic_coefficients[first] * first_value**2
                + parameters.quadratic_coefficients[second] * second_value**2
            ) / (2.0 * linear)
            return _assemble_coordinates(
                ((first, first_value), (second, second_value), (axial, axial_value))
            )

        return (ParametricPatch(point, (-1.0, 1.0), (-1.0, 1.0), PATCH_RESOLUTION),)

    def _elliptic_cylinder(self, parameters: SurfaceParameters) -> tuple[ParametricPatch, ...]:
        active = _active_indices(parameters.quadratic_coefficients)
        _require_count(active, 2, "elliptic cylinder")
        free = tuple(index for index in range(3) if index not in active)
        radii = parameters.axis_scales
        half_length = UNBOUNDED_SCALE_FACTOR * max(radii[active[0]], radii[active[1]])

        def point(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            return _assemble_coordinates(
                (
                    (active[0], radii[active[0]] * np.cos(u_value)),
                    (active[1], radii[active[1]] * np.sin(u_value)),
                    (free[0], v_value),
                )
            )

        return (
            ParametricPatch(
                point,
                (0.0, 2.0 * np.pi),
                (-half_length, half_length),
                PATCH_RESOLUTION,
            ),
        )

    def _hyperbolic_cylinder(self, parameters: SurfaceParameters) -> tuple[ParametricPatch, ...]:
        active = _active_indices(parameters.quadratic_coefficients)
        _require_count(active, 2, "hyperbolic cylinder")
        free = tuple(index for index in range(3) if index not in active)
        normalized = tuple(
            coefficient / -parameters.constant for coefficient in parameters.quadratic_coefficients
        )
        positive = tuple(index for index in active if normalized[index] > SURFACE_TOLERANCE)
        negative = tuple(index for index in active if normalized[index] < -SURFACE_TOLERANCE)
        _require_count(positive, 1, "hyperbolic cylinder")
        _require_count(negative, 1, "hyperbolic cylinder")
        positive_radius = 1.0 / np.sqrt(normalized[positive[0]])
        negative_radius = 1.0 / np.sqrt(abs(normalized[negative[0]]))
        half_length = UNBOUNDED_SCALE_FACTOR * max(positive_radius, negative_radius)
        limit = float(np.arcsinh(half_length / negative_radius))

        def positive_branch(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            return self._hyperbolic_cylinder_point(
                u_value,
                v_value,
                1.0,
                positive[0],
                negative[0],
                free[0],
                positive_radius,
                negative_radius,
            )

        def negative_branch(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            return self._hyperbolic_cylinder_point(
                u_value,
                v_value,
                -1.0,
                positive[0],
                negative[0],
                free[0],
                positive_radius,
                negative_radius,
            )

        return (
            ParametricPatch(positive_branch, (-limit, limit), (-half_length, half_length), PATCH_RESOLUTION),
            ParametricPatch(negative_branch, (-limit, limit), (-half_length, half_length), PATCH_RESOLUTION),
        )

    def _hyperbolic_cylinder_point(
        self,
        u_value: ArrayLike,
        v_value: ArrayLike,
        branch_sign: float,
        positive_axis: int,
        negative_axis: int,
        free_axis: int,
        positive_radius: float,
        negative_radius: float,
    ) -> FloatArray:
        """Evaluate one branch of a finite hyperbolic-cylinder patch."""

        return _assemble_coordinates(
            (
                (positive_axis, branch_sign * positive_radius * np.cosh(u_value)),
                (negative_axis, negative_radius * np.sinh(u_value)),
                (free_axis, v_value),
            )
        )

    def _intersecting_planes(self, parameters: SurfaceParameters) -> tuple[ParametricPatch, ...]:
        active = _active_indices(parameters.quadratic_coefficients)
        _require_count(active, 2, "intersecting planes")
        free = tuple(index for index in range(3) if index not in active)
        first, second = active
        slope = np.sqrt(
            abs(parameters.quadratic_coefficients[first] / parameters.quadratic_coefficients[second])
        )
        first_direction = np.zeros(3, dtype=np.float64)
        first_direction[first] = 1.0
        first_direction[second] = slope
        first_direction /= np.linalg.norm(first_direction)
        second_direction = first_direction.copy()
        second_direction[second] *= -1.0
        free_direction = np.zeros(3, dtype=np.float64)
        free_direction[free[0]] = 1.0

        def first_plane(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            return np.asarray(
                np.expand_dims(u_value, axis=-1) * first_direction
                + np.expand_dims(v_value, axis=-1) * free_direction,
                dtype=np.float64,
            )

        def second_plane(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            return np.asarray(
                np.expand_dims(u_value, axis=-1) * second_direction
                + np.expand_dims(v_value, axis=-1) * free_direction,
                dtype=np.float64,
            )

        domain = (-PRESENTATION_RADIUS, PRESENTATION_RADIUS)
        return (
            ParametricPatch(first_plane, domain, domain, PATCH_RESOLUTION),
            ParametricPatch(second_plane, domain, domain, PATCH_RESOLUTION),
        )

    def _parabolic_cylinder(self, parameters: SurfaceParameters) -> tuple[ParametricPatch, ...]:
        quadratic_axes = _active_indices(parameters.quadratic_coefficients)
        linear_axes = _active_indices(parameters.linear_coefficients)
        _require_count(quadratic_axes, 1, "parabolic cylinder")
        _require_count(linear_axes, 1, "parabolic cylinder")
        quadratic_axis = quadratic_axes[0]
        axial = linear_axes[0]
        free = tuple(index for index in range(3) if index not in (quadratic_axis, axial))
        quadratic = parameters.quadratic_coefficients[quadratic_axis]
        linear = parameters.linear_coefficients[axial]
        focal_scale = abs(linear / quadratic)
        height = UNBOUNDED_SCALE_FACTOR * focal_scale
        transverse_extent = np.sqrt(2.0 * focal_scale * height)

        def point(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            transverse_value = np.asarray(u_value, dtype=np.float64)
            axial_value = -(quadratic * transverse_value**2) / (2.0 * linear)
            return _assemble_coordinates(
                ((quadratic_axis, transverse_value), (axial, axial_value), (free[0], v_value))
            )

        return (
            ParametricPatch(
                point,
                (-transverse_extent, transverse_extent),
                (-height, height),
                PATCH_RESOLUTION,
            ),
        )

    def _parallel_planes(self, parameters: SurfaceParameters) -> tuple[ParametricPatch, ...]:
        active = _active_indices(parameters.quadratic_coefficients)
        _require_count(active, 1, "parallel planes")
        normal_axis = active[0]
        tangent = tuple(index for index in range(3) if index != normal_axis)
        separation = parameters.axis_scales[normal_axis]
        if separation <= 0:
            raise ValueError("real parallel planes require positive separation")
        half_length = UNBOUNDED_SCALE_FACTOR * separation

        def positive_plane(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            return _assemble_coordinates(
                ((normal_axis, separation * np.ones_like(u_value)), (tangent[0], u_value), (tangent[1], v_value))
            )

        def negative_plane(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            return _assemble_coordinates(
                ((normal_axis, -separation * np.ones_like(u_value)), (tangent[0], u_value), (tangent[1], v_value))
            )

        domain = (-half_length, half_length)
        return (
            ParametricPatch(positive_plane, domain, domain, PATCH_RESOLUTION),
            ParametricPatch(negative_plane, domain, domain, PATCH_RESOLUTION),
        )

    def _double_plane(self, parameters: SurfaceParameters) -> tuple[ParametricPatch, ...]:
        active = _active_indices(parameters.quadratic_coefficients)
        _require_count(active, 1, "double plane")
        normal_axis = active[0]
        tangent = tuple(index for index in range(3) if index != normal_axis)

        def point(u_value: ArrayLike, v_value: ArrayLike) -> FloatArray:
            return _assemble_coordinates(
                ((normal_axis, np.zeros_like(u_value)), (tangent[0], u_value), (tangent[1], v_value))
            )

        domain = (-PRESENTATION_RADIUS, PRESENTATION_RADIUS)
        return (ParametricPatch(point, domain, domain, PATCH_RESOLUTION),)


__all__ = [
    "ParametricPatch",
    "SurfaceSpec",
    "SurfaceSpecFactory",
    "UnsupportedSurfaceError",
]
