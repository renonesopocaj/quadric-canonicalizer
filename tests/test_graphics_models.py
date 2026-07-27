"""Verify graphics contracts with ``python -m pytest tests/test_graphics_models.py -q``."""

from pathlib import Path

import numpy as np
import pytest

from src import canonize_quadric
from src.graphics.models import Bounds3D, CameraFraming, RenderPlan, RenderSettings, SurfaceParameters
from src.graphics.surface_spec import SurfaceSpecFactory
from src.numerical.models import QuadricType


@pytest.mark.parametrize(
    ("quality", "expected"),
    [
        ("1", "low_quality"),
        ("2", "medium_quality"),
        ("3", "high_quality"),
        ("4", "production_quality"),
    ],
)
def test_render_quality_mapping(quality: str, expected: str) -> None:
    assert RenderSettings(quality=quality, output_path=Path("media")).manim_quality == expected


def test_render_quality_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="quality"):
        _ = RenderSettings(quality="5", output_path=Path("media")).manim_quality


def test_surface_parameters_extract_axis_scales() -> None:
    matrix = np.diag([4.0, 1.0, 0.25, -1.0])
    parameters = SurfaceParameters.from_matrix(QuadricType.REAL_ELLIPSOID, matrix)

    assert parameters.axis_scales == (0.5, 1.0, 2.0)


def test_render_plan_consumes_the_public_numerical_result() -> None:
    result = canonize_quadric("(x-1)**2 + (y+2)**2 - z = 0")
    plan = RenderPlan.from_result(result)

    assert plan.result is result
    assert plan.surface_parameters.matrix is result.final_matrix
    for planned_step, numerical_step in zip(plan.transformation_steps, result.transformation_steps):
        assert planned_step.kind is numerical_step.kind
        np.testing.assert_allclose(planned_step.homogeneous_matrix, numerical_step.homogeneous_matrix)


@pytest.mark.parametrize("radius", [0.05, 1.0, 100.0])
def test_camera_zoom_is_scale_covariant(radius: float) -> None:
    bounds = Bounds3D(
        minimum=np.array([-radius, -radius, -radius]),
        maximum=np.array([radius, radius, radius]),
    )
    framing = CameraFraming.fit(
        bounds=bounds,
        frame_width=14.0,
        frame_height=8.0,
        phi_radians=np.deg2rad(65.0),
        fill_ratio=0.7,
    )

    vertical_radius = np.cos(np.deg2rad(65.0)) * np.sqrt(2.0) * radius + np.sin(
        np.deg2rad(65.0)
    ) * radius
    assert 2.0 * vertical_radius * framing.zoom / 8.0 == pytest.approx(0.7)
    np.testing.assert_allclose(framing.frame_center, np.zeros(3))


def test_render_plan_frames_every_animation_stage_independently() -> None:
    result = canonize_quadric("(x-100)**2 + (y+50)**2 + (z-25)**2 = 0.01")
    plan = RenderPlan.from_result(result)
    canonical_bounds = Bounds3D(
        minimum=np.array([-0.1, -0.1, -0.1]),
        maximum=np.array([0.1, 0.1, 0.1]),
    )

    stage_bounds = plan.stage_bounds(canonical_bounds)
    framings = tuple(
        CameraFraming.fit(
            bounds=bounds,
            frame_width=14.0,
            frame_height=8.0,
            phi_radians=np.deg2rad(65.0),
            fill_ratio=0.7,
        )
        for bounds in stage_bounds
    )

    assert len(stage_bounds) == 3
    for bounds, framing in zip(stage_bounds, framings):
        np.testing.assert_allclose(framing.frame_center, bounds.center)
    assert framings[0].zoom == pytest.approx(framings[-1].zoom)


@pytest.mark.parametrize("radius", [1e-6, 1e6])
def test_numerical_surface_and_camera_pipeline_supports_extreme_quadric_sizes(radius: float) -> None:
    result = canonize_quadric(f"x**2 + y**2 + z**2 = {radius**2}")
    bounds = SurfaceSpecFactory().create(result).bounds
    framing = CameraFraming.fit(
        bounds=bounds,
        frame_width=14.0,
        frame_height=8.0,
        phi_radians=np.deg2rad(65.0),
        fill_ratio=0.7,
    )

    np.testing.assert_allclose(bounds.minimum, np.full(3, -radius), rtol=1e-9, atol=radius * 1e-9)
    np.testing.assert_allclose(bounds.maximum, np.full(3, radius), rtol=1e-9, atol=radius * 1e-9)
    expected_scaled_zoom = 0.7 * 8.0 / (
        2.0 * (np.cos(np.deg2rad(65.0)) * np.sqrt(2.0) + np.sin(np.deg2rad(65.0)))
    )
    assert framing.zoom * radius == pytest.approx(expected_scaled_zoom, rel=1e-5)
