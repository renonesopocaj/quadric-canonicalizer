"""Run real-Manim smoke checks with ``python -m pytest tests/test_manim_smoke.py -q``."""

import numpy as np
import pytest

mn = pytest.importorskip("manim")

from src import canonize_quadric
from src.graphics.create_quadric_surface import QuadricSurfaceFactory
from src.graphics.models import CameraFraming, RenderPlan
from src.graphics.scene_render import CAMERA_PHI, CAMERA_THETA, SceneRender


@pytest.mark.parametrize("radius", [1e-6, 1e6])
def test_manim_surface_factory_builds_finite_extreme_scale_geometry(radius: float) -> None:
    result = canonize_quadric(f"x**2 + y**2 + z**2 = {radius**2}")
    build = QuadricSurfaceFactory().create(result)
    points = build.surface.get_all_points()

    assert points.ndim == 2
    assert points.shape[0] > 0
    assert points.shape[1] == 3
    assert np.all(np.isfinite(points))


def test_manim_camera_accepts_the_computed_stage_framing() -> None:
    result = canonize_quadric("(x-100)**2 + (y+50)**2 + (z-25)**2 = 0.01")
    build = QuadricSurfaceFactory().create(result)
    initial_bounds = RenderPlan.from_result(result).stage_bounds(build.bounds)[0]
    framing = CameraFraming.fit(
        initial_bounds,
        float(mn.config.frame_width),
        float(mn.config.frame_height),
        float(CAMERA_PHI),
        0.7,
    )
    scene = SceneRender(result)
    scene.set_camera_orientation(
        phi=CAMERA_PHI,
        theta=CAMERA_THETA,
        zoom=framing.zoom,
        focal_distance=framing.focal_distance,
        frame_center=framing.frame_center.tolist(),
    )

    assert scene.camera.get_zoom() == pytest.approx(framing.zoom)
    np.testing.assert_allclose(scene.camera.frame_center, framing.frame_center)
