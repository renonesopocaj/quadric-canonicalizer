"""
Provide the interactive command-line entry point and Manim render orchestration.

Run the application with ``python -m src`` after installing the graphics extra.
Run non-rendering tests with ``python -m pytest tests -q``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.graphics.models import RenderSettings
from src.numerical.models import CanonicalizationResult, QuadricType
from src.numerical.transformer import canonize_quadric


@dataclass(frozen=True, slots=True)
class QuadricExample:
    """Store one selectable CLI example without an untyped dictionary."""

    quadric_type: QuadricType
    equation: str


class ExampleCatalog:
    """Own and query the deterministic examples exposed by the CLI."""

    examples: tuple[QuadricExample, ...] = (
        QuadricExample(QuadricType.REAL_ELLIPSOID, "2x**2 + 2*y**2 + 4z**2 - 2xy + 2x = 0"),
        QuadricExample(QuadricType.ONE_SHEET_HYPERBOLOID, "x**2 + y**2 + z**2 +2xy - 2xz + 2yz + 4x + 4y - 4z + 2 = 0"),
        QuadricExample(QuadricType.TWO_SHEET_HYPERBOLOID, "x**2 + y**2 - 3z**2 - 2xy - 6xz - 6yz + 2x + 2y + 4z = 0"),
        QuadricExample(QuadricType.REAL_CONE, "y**2 - 6xz - 6x + 2y - 6z - 5 = 0"),
        QuadricExample(QuadricType.ELLIPTIC_PARABOLOID, "x**2 + y**2 +2z**2 + 2xy - 4x = 0"),
        QuadricExample(QuadricType.HYPERBOLIC_PARABOLOID, "12xz + 16yz - 10x = 0"),
        QuadricExample(QuadricType.REAL_ELLIPTIC_CYLINDER, "3x**2 + 2y**2 + 4z**2 - 4xy + 4xz + 6x + 12z + 3 = 0"),
        QuadricExample(QuadricType.HYPERBOLIC_CYLINDER, "x**2-4x-y**2+6y-4=0"),
        QuadricExample(QuadricType.REAL_INTERSECTING_PLANES, "x**2 - 2y**2 - 2z**2 - xy - xz + 5yz + 2x - y - z + 1 = 0"),
        QuadricExample(QuadricType.PARABOLIC_CYLINDER, "x**2 + y**2 - 2xy - 4x - 4y - 4z + 4 = 0"),
        QuadricExample(QuadricType.REAL_PARALLEL_PLANES, "x**2 + y**2 + z**2 - 2xy + 2xz - 2yz + 6x - 6y + 6z + 8 = 0"),
        QuadricExample(QuadricType.DOUBLE_PLANE, "x**2 + y**2 + z**2 + 2xy + 2xz + 2yz + 2x + 2y + 2z + 1 = 0"),
    )

    def find(self, selection: str) -> QuadricExample:
        """Return the example whose enum value matches a user selection."""

        try:
            selected_type = QuadricType(int(selection))
        except (TypeError, ValueError) as error:
            raise ValueError(f"unknown example selection: {selection}") from error
        for example in self.examples:
            if example.quadric_type is selected_type:
                return example
        raise ValueError(f"no renderable example is available for type {selection}")


class VideoRenderer:
    """Configure Manim and render one typed canonicalization result."""

    result: CanonicalizationResult
    settings: RenderSettings

    def __init__(self, result: CanonicalizationResult, settings: RenderSettings) -> None:
        self.result = result
        self.settings = settings

    def render(self) -> None:
        """Create the output directory, configure Manim, and render the scene."""

        try:
            import manim as mn
            from src.graphics.scene_render import SceneRender
        except ModuleNotFoundError as error:
            raise RuntimeError("rendering requires the 'graphics' dependencies; install with 'pip install .[graphics]'") from error

        self.settings.output_path.mkdir(parents=True, exist_ok=True)
        mn.config.media_dir = str(self.settings.output_path)
        mn.config.output_file = f"{self.result.quadric_type.slug}.mp4"
        mn.config.quality = self.settings.manim_quality
        SceneRender(self.result).render()


def graphic_wrapper_function(result: CanonicalizationResult, video_quality: str, output_path: str) -> None:
    """
    Render a canonicalization result through the compatibility function API.

    Args:
        result: CanonicalizationResult
            Typed output produced by ``canonize_quadric``.
        video_quality: str
            Quality code from "1" through "4".
        output_path: str
            Render directory; an empty value selects ``./media``.
        return: None
            The configured Manim scene is rendered to disk.
    """

    media_path = Path(output_path) if output_path else Path("./media")
    VideoRenderer(result=result, settings=RenderSettings(quality=video_quality or "1", output_path=media_path)).render()


def select_example() -> str:
    """Prompt until the user selects one available deterministic example."""

    prompt = "Select a quadric type by number (available: 1,3,4,5,7,8,9,11,12,14,15,17): "
    catalog = ExampleCatalog()
    while True:
        try:
            return catalog.find(input(prompt)).equation
        except ValueError as error:
            print(error)


def main() -> None:
    """Read CLI input, canonicalize the equation, and render its transformation."""

    use_example = input("Use a bundled quadric example? Type y/n: ")
    if use_example.lower() == "y":
        equation = select_example()
    else:
        equation = input("Insert a degree-two equation in x, y, and z with one '=' sign: ")
    result = canonize_quadric(equation)
    output_path = input("Insert an output directory or press ENTER for ./media: ")
    quality = input("Choose quality (1 low, 2 medium, 3 high, 4 production; ENTER selects 1): ")
    graphic_wrapper_function(result, quality or "1", output_path)


if __name__ == "__main__":
    main()
