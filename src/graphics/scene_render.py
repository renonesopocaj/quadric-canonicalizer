"""
Render the ordered numerical canonicalization contract with Manim.

The scene derives surface geometry, overlays, active transforms, axes, and
camera framing from one ``CanonicalizationResult``.
"""

from __future__ import annotations

from collections.abc import Callable

import manim as mn
import numpy as np
from manim import FadeIn, MathTex, VGroup

from src import AffineTransformation, CanonicalizationResult, TransformationKind
from src.graphics.create_quadric_surface import QuadricSurfaceFactory
from src.graphics.create_text_overlay import TextOverlayBuilder
from src.graphics.models import AxisLayout, CameraFraming, RenderPlan


WAIT_TIME = 5.0
CAMERA_PHI = 65.0 * mn.DEGREES
CAMERA_THETA = -20.0 * mn.DEGREES
CAMERA_FILL_RATIO = 0.7
AMBIENT_ROTATION_RATE = 0.1
AXIS_LABEL_SCALE = 0.7

StepAnimationFactory = Callable[[mn.Mobject, AffineTransformation], mn.Animation]


def _translation_animation(
    surface: mn.Mobject,
    step: AffineTransformation,
) -> mn.Animation:
    """Return a Manim animation applying one active translation step."""

    return mn.Transform(surface, surface.copy().shift(step.offset))


def _rotation_animation(
    surface: mn.Mobject,
    step: AffineTransformation,
) -> mn.Animation:
    """Return a Manim animation applying one active proper rotation step."""

    return mn.ApplyMatrix(step.linear_map, surface)


STEP_ANIMATION_FACTORIES: dict[TransformationKind, StepAnimationFactory] = {
    TransformationKind.TRANSLATION: _translation_animation,
    TransformationKind.ROTATION: _rotation_animation,
}


def _apply_homogeneous_transform(surface: mn.Mobject, transform: np.ndarray) -> None:
    """Apply an explicit affine point matrix to an existing Manim mobject."""

    if transform.shape != (4, 4):
        raise ValueError("surface transform must have shape (4, 4)")
    surface.apply_matrix(transform[:3, :3])
    surface.shift(transform[:3, 3])


def create_axes(layout: AxisLayout) -> tuple[mn.ThreeDAxes, VGroup]:
    """
    Create axes whose physical lengths match their numerical spans.

    Args:
        layout: AxisLayout
            Ranges covering all three animation stages.
        return: tuple[ThreeDAxes, VGroup]
            World-unit axes and fixed-orientation x-y-z labels.
    """

    x_range, y_range, z_range = layout.ranges
    axes = mn.ThreeDAxes(
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        x_length=x_range[1] - x_range[0],
        y_length=y_range[1] - y_range[0],
        z_length=z_range[1] - z_range[0],
    ).add_coordinates()
    labels = VGroup(
        MathTex("X", color=mn.ORANGE).scale(AXIS_LABEL_SCALE).move_to(axes.c2p(x_range[1], 0.0, 0.0)),
        MathTex("Y", color=mn.ORANGE).scale(AXIS_LABEL_SCALE).move_to(axes.c2p(0.0, y_range[1], 0.0)),
        MathTex("Z", color=mn.ORANGE).scale(AXIS_LABEL_SCALE).move_to(axes.c2p(0.0, 0.0, z_range[1])),
    )
    return axes, labels


class SceneRender(mn.ThreeDScene):
    """
    Animate one canonicalization using the numerical API's exact stage steps.

    Args:
        result: CanonicalizationResult
            Validated numerical artifacts consumed through ``RenderPlan``.
        return: SceneRender
            Manim scene configured for the complete transformation.
    """

    result: CanonicalizationResult
    plan: RenderPlan

    def __init__(self, result: CanonicalizationResult) -> None:
        super().__init__()
        self.result = result
        self.plan = RenderPlan.from_result(result)

    def construct(self) -> None:
        """Construct the surface, adaptive camera, axes, overlays, and two steps."""

        overlays = TextOverlayBuilder().build(self.plan)
        surface_build = QuadricSurfaceFactory().create(self.result)
        surface = surface_build.surface
        stage_bounds = self.plan.stage_bounds(surface_build.bounds)
        framings = tuple(
            CameraFraming.fit(
                bounds=bounds,
                frame_width=float(mn.config.frame_width),
                frame_height=float(mn.config.frame_height),
                phi_radians=float(CAMERA_PHI),
                fill_ratio=CAMERA_FILL_RATIO,
            )
            for bounds in stage_bounds
        )
        first_step, second_step = self.plan.transformation_steps

        _apply_homogeneous_transform(surface, second_step.inverse_homogeneous_matrix)
        _apply_homogeneous_transform(surface, first_step.inverse_homogeneous_matrix)

        axes, labels = create_axes(AxisLayout.from_stage_bounds(stage_bounds))
        maximum_focal_distance = max(framing.focal_distance for framing in framings)
        self.set_camera_orientation(
            phi=CAMERA_PHI,
            theta=CAMERA_THETA,
            zoom=framings[0].zoom,
            focal_distance=maximum_focal_distance,
            frame_center=framings[0].frame_center.tolist(),
        )
        self.add(axes)
        self.add_fixed_orientation_mobjects(labels)
        self.begin_ambient_camera_rotation(rate=AMBIENT_ROTATION_RATE)
        self.add_fixed_in_frame_mobjects(overlays.initial)
        self.add(surface)
        self.wait(WAIT_TIME)

        self._animate_step(
            surface=surface,
            step=first_step,
            framing=framings[1],
            transformation_overlay=overlays.first_transformation,
        )
        self.add_fixed_in_frame_mobjects(overlays.middle)
        self.play(FadeIn(overlays.middle))
        self.wait(2.0)

        self._animate_step(
            surface=surface,
            step=second_step,
            framing=framings[2],
            transformation_overlay=overlays.second_transformation,
        )
        self.add_fixed_in_frame_mobjects(overlays.final)
        self.play(FadeIn(overlays.final))
        self.wait(10.0)

    def _animate_step(
        self,
        surface: mn.Mobject,
        step: AffineTransformation,
        framing: CameraFraming,
        transformation_overlay: mn.Mobject,
    ) -> None:
        """Animate one API step while keeping its geometry tightly framed."""

        self.add_fixed_in_frame_mobjects(transformation_overlay)
        self.play(mn.GrowFromEdge(transformation_overlay, mn.LEFT))
        try:
            animation_factory = STEP_ANIMATION_FACTORIES[step.kind]
        except KeyError as error:
            raise ValueError(f"unsupported transformation kind: {step.kind}") from error
        self.move_camera(
            zoom=framing.zoom,
            frame_center=framing.frame_center.tolist(),
            added_anims=[animation_factory(surface, step)],
            run_time=WAIT_TIME,
        )


__all__ = ["SceneRender", "create_axes"]
