"""Verify graphics contracts with ``python -m pytest tests/test_graphics_models.py -q``."""

from pathlib import Path

import numpy as np
import pytest

from src.graphics.models import RenderSettings, SurfaceParameters
from src.numerical.models import QuadricType


@pytest.mark.parametrize(("quality", "expected"), [("1", "low_quality"), ("2", "medium_quality"), ("3", "high_quality"), ("4", "production_quality")])
def test_render_quality_mapping(quality: str, expected: str) -> None:
    assert RenderSettings(quality=quality, output_path=Path("media")).manim_quality == expected


def test_render_quality_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="quality"):
        _ = RenderSettings(quality="5", output_path=Path("media")).manim_quality


def test_surface_parameters_extract_axis_scales() -> None:
    matrix = np.diag([4.0, 1.0, 0.25, -1.0])
    parameters = SurfaceParameters.from_matrix(QuadricType.REAL_ELLIPSOID, matrix)

    assert parameters.axis_scales == (0.5, 1.0, 2.0)
