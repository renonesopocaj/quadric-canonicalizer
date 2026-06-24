"""Verify application orchestration with ``python -m pytest tests/test_main.py -q``."""

from pathlib import Path
import sys

import pytest

from src.graphics.models import RenderSettings
from src.main import ExampleCatalog, VideoRenderer
from src.numerical.canonicalize import canonize_quadric


def test_every_bundled_example_matches_its_declared_type() -> None:
    for example in ExampleCatalog.examples:
        assert canonize_quadric(example.equation).quadric_type is example.quadric_type


def test_renderer_surfaces_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "manim", None)
    result = canonize_quadric("x**2 + y**2 + z**2 = 1")
    renderer = VideoRenderer(result=result, settings=RenderSettings(quality="1", output_path=tmp_path))

    with pytest.raises(RuntimeError, match="graphics"):
        renderer.render()
