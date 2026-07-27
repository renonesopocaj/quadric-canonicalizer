"""
Build fixed-frame Manim text directly from a graphics render plan.

Numerical values stay full precision in the API and are rounded only while
constructing these presentation objects.
"""

from __future__ import annotations

import manim as mn
import numpy as np
import sympy as sp
from manim import Arrow, MathTex, TexTemplate, VGroup

from src import AffineTransformation, TransformationKind
from src.graphics.models import RenderPlan, TextOverlayGroups


TEXT_COLOR = mn.RED
TEXT_BOLDNESS = 1
ARROW_WIDTH = 1.5
ARROW_TIP_SCALE = 0.5
TEXT_SCALE = 0.4
LABEL_SCALE = 0.6
EQUATION_EDGE_BUFFER = 0.2
MATRIX_UPPER_BUFFER = 1.0
MATRIX_SIDE_BUFFER = 0.5
TRANSFORMATION_UPPER_BUFFER = 0.6
VECTOR_SIDE_BUFFER = 3.5
MATRIX_TRANSFORM_SIDE_BUFFER = 3.0
DISPLAY_DECIMALS = 2


def _tex_template() -> TexTemplate:
    """Return the shared compact mathematical text template."""

    template = TexTemplate()
    template.add_to_preamble(
        r"""
        \usepackage{amsmath}
        \usepackage{bm}
        \thickmuskip=1.5mu
        \medmuskip=1mu
        \thinmuskip=0.5mu
        """
    )
    return template


def convert_equation(equation: sp.Expr) -> MathTex:
    """
    Convert one polynomial expression into a compact Manim equation.

    Args:
        equation: sympy.Expr
            Polynomial expression whose zero level set is the quadric.
        return: MathTex
            Fixed-frame equation ending in ``=0``.
    """

    display_replacements = {
        coefficient: sp.Float(round(float(coefficient), DISPLAY_DECIMALS))
        for coefficient in equation.atoms(sp.Float)
    }
    display_equation = equation.xreplace(display_replacements)
    equation_text = str(display_equation).replace("**", "^").replace("*", "").replace(" ", "")
    return MathTex(f"{equation_text}=0", tex_template=_tex_template())


def _transformation_group(step: AffineTransformation, side: np.ndarray) -> VGroup:
    """Build one arrow and its active rotation or translation value."""

    if step.kind is TransformationKind.ROTATION:
        label = MathTex("R=").scale(LABEL_SCALE)
        values = mn.Matrix(np.round(step.linear_map, decimals=DISPLAY_DECIMALS)).scale(TEXT_SCALE)
        side_buffer = MATRIX_TRANSFORM_SIDE_BUFFER
    else:
        label = MathTex("v=").scale(LABEL_SCALE)
        values = mn.Matrix(
            np.round(step.offset, decimals=DISPLAY_DECIMALS).reshape(3, 1)
        ).scale(TEXT_SCALE)
        side_buffer = VECTOR_SIDE_BUFFER

    values.next_to(label, mn.RIGHT)
    value_group = VGroup(label, values)
    arrow = Arrow(start=mn.LEFT, end=mn.RIGHT, stroke_width=ARROW_WIDTH)
    arrow.tip.scale(ARROW_TIP_SCALE)
    arrow.set_color(TEXT_COLOR)
    value_group.next_to(arrow, mn.UP)
    group = VGroup(value_group, arrow)
    group.to_edge(mn.UP, buff=TRANSFORMATION_UPPER_BUFFER)
    group.to_edge(side, buff=side_buffer)
    return group


class TextOverlayBuilder:
    """Build every text state from one immutable render plan."""

    def build(self, plan: RenderPlan) -> TextOverlayGroups:
        """
        Create equations, matrices, and the two reported active transforms.

        Args:
            plan: RenderPlan
                Graphics adapter built from the public numerical result.
            return: TextOverlayGroups
                Five fixed-frame Manim groups in animation order.
        """

        mn.Text.set_default(weight="BOLD", color=TEXT_COLOR)
        MathTex.set_default(
            tex_template=_tex_template(),
            color=TEXT_COLOR,
            stroke_width=TEXT_BOLDNESS,
        )
        result = plan.result
        equations = (
            convert_equation(result.initial_equation).scale(TEXT_SCALE),
            convert_equation(result.middle_equation).scale(TEXT_SCALE),
            convert_equation(result.final_equation).scale(TEXT_SCALE),
        )
        matrices = (
            mn.Matrix(np.round(result.initial_matrix, decimals=DISPLAY_DECIMALS)).scale(TEXT_SCALE),
            mn.Matrix(np.round(result.middle_matrix, decimals=DISPLAY_DECIMALS)).scale(TEXT_SCALE),
            mn.Matrix(np.round(result.final_matrix, decimals=DISPLAY_DECIMALS)).scale(TEXT_SCALE),
        )

        equations[0].to_corner(mn.UL, buff=EQUATION_EDGE_BUFFER)
        matrices[0].to_edge(mn.LEFT, buff=MATRIX_SIDE_BUFFER)
        matrices[0].to_edge(mn.UP, buff=MATRIX_UPPER_BUFFER)
        initial = VGroup(equations[0], matrices[0])

        equations[1].to_edge(mn.UP, buff=EQUATION_EDGE_BUFFER)
        matrices[1].to_edge(mn.UP, buff=MATRIX_UPPER_BUFFER)
        middle = VGroup(equations[1], matrices[1])

        equations[2].to_corner(mn.UR, buff=EQUATION_EDGE_BUFFER)
        matrices[2].to_edge(mn.RIGHT, buff=MATRIX_SIDE_BUFFER)
        matrices[2].to_edge(mn.UP, buff=MATRIX_UPPER_BUFFER)
        final = VGroup(equations[2], matrices[2])

        first_step, second_step = plan.transformation_steps
        return TextOverlayGroups(
            initial=initial,
            first_transformation=_transformation_group(first_step, mn.LEFT),
            middle=middle,
            second_transformation=_transformation_group(second_step, mn.RIGHT),
            final=final,
        )


def text_overlay(plan: RenderPlan) -> TextOverlayGroups:
    """
    Build overlay groups through the compatibility function entry point.

    Args:
        plan: RenderPlan
            Complete graphics input produced from ``CanonicalizationResult``.
        return: TextOverlayGroups
            Fixed-frame transformation text groups.
    """

    return TextOverlayBuilder().build(plan)


__all__ = ["TextOverlayBuilder", "convert_equation", "text_overlay"]
