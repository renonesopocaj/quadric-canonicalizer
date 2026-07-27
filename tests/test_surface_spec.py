"""Verify pure surface specifications with ``python -m pytest tests/test_surface_spec.py -q``."""

import numpy as np
import pytest

from src import canonize_quadric
from src.graphics.surface_spec import SurfaceSpecFactory
from src.main import ExampleCatalog


@pytest.mark.parametrize(
    "equation",
    [
        "x**2 + y**2 + z**2 = 1",
        "x**2 + y**2 - z**2 = 1",
        "x**2 + y**2 - z**2 = -1",
        "x**2 + y**2 - z**2 = 0",
        "x**2 + y**2 - z = 0",
        "x**2 - y**2 - z = 0",
        "x**2 + y**2 = 1",
        "x**2 - y**2 = 1",
        "x**2 - y**2 = 0",
        "x**2 - y = 0",
        "x**2 = 1",
        "x**2 = 0",
    ],
)
def test_every_renderable_surface_patch_satisfies_the_canonical_equation(equation: str) -> None:
    result = canonize_quadric(equation)
    spec = SurfaceSpecFactory().create(result)
    points = spec.sample_points(samples_per_axis=11)
    homogeneous_points = np.column_stack([points, np.ones(points.shape[0])])
    normalized_matrix = result.final_matrix / np.max(np.abs(result.final_matrix))
    residuals = np.einsum("ni,ij,nj->n", homogeneous_points, normalized_matrix, homogeneous_points)

    np.testing.assert_allclose(residuals, np.zeros_like(residuals), atol=1e-8)
    assert spec.bounds.contains(points)


def test_surface_spec_is_invariant_to_equation_coefficient_scaling() -> None:
    reference = SurfaceSpecFactory().create(canonize_quadric("x**2 + y**2 + z**2 = 1"))
    scaled = SurfaceSpecFactory().create(
        canonize_quadric("1000000000000*x**2 + 1000000000000*y**2 + 1000000000000*z**2 = 1000000000000")
    )

    np.testing.assert_allclose(scaled.bounds.minimum, reference.bounds.minimum, atol=1e-12)
    np.testing.assert_allclose(scaled.bounds.maximum, reference.bounds.maximum, atol=1e-12)


def test_bundled_surface_points_follow_every_reported_graphics_stage() -> None:
    for example in ExampleCatalog.examples:
        result = canonize_quadric(example.equation)
        points = SurfaceSpecFactory().create(result).sample_points(samples_per_axis=7)
        first_step, second_step = result.transformation_steps
        final_homogeneous = np.column_stack([points, np.ones(points.shape[0])])
        middle_homogeneous = final_homogeneous @ second_step.inverse_homogeneous_matrix.T
        initial_homogeneous = middle_homogeneous @ first_step.inverse_homogeneous_matrix.T

        for stage_points, stage_matrix in (
            (initial_homogeneous, result.initial_matrix),
            (middle_homogeneous, result.middle_matrix),
            (final_homogeneous, result.final_matrix),
        ):
            normalized_matrix = stage_matrix / np.max(np.abs(stage_matrix))
            residuals = np.einsum("ni,ij,nj->n", stage_points, normalized_matrix, stage_points)
            np.testing.assert_allclose(residuals, np.zeros_like(residuals), atol=1e-8)
