import manim as mn
import numpy as np
from manim import TexTemplate, MathTex, VGroup, Arrow

"""
This module manages the creation and positioning of mathematical text overlays for quadric 
surface transformations. It handles the visualization of equations, matrices, and transformation 
steps using Manim's text rendering capabilities.

Global Style Settings:
    Text:
        text_color (Color): Color for all text elements (default: RED)
        sca (float): Scale factor for equations and matrices (default: 0.4)
        lab_scale (float): Scale factor for labels (default: 0.6)

    Arrows:
        a_width (float): Width of transformation arrows (default: 1.5)
        a_tip_scale (float): Scale factor for arrow tips (default: 0.5)

    Layout Spacing:
        eq_edge_buff (float): Buffer for equation edge positioning (default: 0.2)
        mat_upper_buff (float): Upper buffer for matrices (default: 1)
        mat_side_buff (float): Side buffer for matrices (default: 0.5)
        trans_upper_buff (float): Upper buffer for transformation text (default: 0.6)
        vec_trans_side_buff (float): Side buffer for vector transformations (default: 3.5)
        mat_trans_side_buff (float): Side buffer for matrix transformations (default: 3)
"""

#text color
text_color = mn.RED
#text boldness
boldness = 1

#arrows width
a_width = 1.5
#arrow tip scale
a_tip_scale = 0.5
#text scale
sca = 0.4
#labels scale
lab_scale = 0.6

#edges
eq_edge_buff = 0.2
mat_upper_buff = 1
mat_side_buff = 0.5
trans_upper_buff = 0.6
vec_trans_side_buff = 3.5
mat_trans_side_buff = 3

#converting equations from string to MathTex object
def convert_equation(eq):
    """
    convert_equation(eq: str) -> MathTex:
        Converts a string equation to a formatted MathTex object with specific spacing settings.
        Automatically adds '=0' and handles common mathematical notation conversions.

    Args:
        eq (str): Input equation string (can use Python notation like '**' for powers)
    """
    template = TexTemplate()
    template.add_to_preamble(r"""
        \usepackage{amsmath}
        \thickmuskip=1.5mu
        \medmuskip=1mu
        \thinmuskip=0.5mu
    """)

    eq=str(eq)
    eq = (eq.replace("**", "^")).replace("*", "").replace(" ", "")
    eq += "=0"
    eq = MathTex(eq, tex_template=template)
    return eq


def text_overlay(is_centered, init_equation, init_matrix, mid_equation, mid_matrix, final_equation, final_matrix, translation_vector, rotation_matrix):
    """
    Creates and positions all text elements for the quadric transformation animation.

    Args:
        is_centered (bool): Whether the quadric is centered (affects transformation order)
        init_equation (str): Initial quadric equation
        init_matrix (np.ndarray): Initial quadric matrix
        mid_equation (str): Intermediate quadric equation
        mid_matrix (np.ndarray): Intermediate quadric matrix
        final_equation (str): Final quadric equation
        final_matrix (np.ndarray): Final quadric matrix
        translation_vector (np.ndarray): Translation vector for the transformation
        rotation_matrix (np.ndarray): Rotation matrix for the transformation

    Layout Structure:
        For centered quadrics:
            Initial -> Translation -> Middle -> Rotation -> Final
        For non-centered quadrics:
            Initial -> Rotation -> Middle -> Translation -> Final

    Note:
        - All matrices and equations are rounded to 2 decimal places for display
        - Text elements use bold formatting
        - LaTeX spacing is adjusted for better readability
    """
    #text template adjustments: characters spacing and boldness
    mn.Text.set_default(weight="BOLD")
    template = TexTemplate()
    template.add_to_preamble(r"""
        \usepackage{amsmath}
        \usepackage{bm}
        \thickmuskip=1.5mu
        \medmuskip=1mu
        \thinmuskip=0.5mu
        """)
    MathTex.set_default(tex_template=template)

    mn.Text.set_default(color=mn.RED)
    MathTex.set_default(color=mn.RED, stroke_width=boldness)

    #initialize all the text objects
    #invert rotation matrix
    rotation_matrix=np.transpose(rotation_matrix)

    #if it is not a centered quadric invert the vector
    if not is_centered:
        translation_vector=-translation_vector

    #initialize equations
    init_equation = convert_equation(init_equation).scale(sca)
    mid_equation = convert_equation(mid_equation).scale(sca)
    final_equation = convert_equation(final_equation).scale(sca)

    #initialize matrices
    init_matrix = np.round(init_matrix, decimals=2)
    init_matrix = mn.Matrix(init_matrix).scale(sca)
    mid_matrix = np.round(mid_matrix, decimals=2)
    mid_matrix = mn.Matrix(mid_matrix).scale(sca)
    final_matrix = np.round(final_matrix, decimals=2)
    final_matrix = mn.Matrix(final_matrix).scale(sca)

    #initialize rotation vector and translation matrix
    rotation_matrix = np.round(rotation_matrix, decimals=2)
    rotation_matrix = mn.Matrix(rotation_matrix).scale(sca)
    translation_vector = np.round(translation_vector, decimals=2).reshape(-1, 1)
    translation_vector = mn.Matrix(translation_vector).scale(sca)

    #starting state
    init_equation.to_corner(mn.UL, buff=eq_edge_buff)
    init_matrix.to_edge(mn.LEFT, buff=mat_side_buff)
    init_matrix.to_edge(mn.UP, buff=mat_upper_buff)
    group_init = VGroup(init_equation, init_matrix)

    #first transition
    if is_centered:
        v = MathTex("v=").scale(lab_scale)
        temp = VGroup(translation_vector, v)
        translation_vector.next_to(v, mn.RIGHT)
        arrow1 = Arrow(start=mn.LEFT, end=mn.RIGHT, stroke_width=a_width)
        arrow1.tip.scale(a_tip_scale)
        arrow1.set_color(mn.RED)
        group_trans1 = VGroup(temp, arrow1)
        temp.next_to(arrow1, mn.UP)
        group_trans1.to_edge(mn.UP, buff=trans_upper_buff)
        group_trans1.to_edge(mn.LEFT, buff=vec_trans_side_buff)

    else:
        s = MathTex("S=", tex_template=template).scale(lab_scale)
        temp = VGroup(rotation_matrix, s)
        rotation_matrix.next_to(s, mn.RIGHT)
        arrow1 = Arrow(start=mn.LEFT, end=mn.RIGHT, stroke_width=a_width)
        arrow1.tip.scale(a_tip_scale)
        arrow1.set_color(mn.RED)
        group_trans1 = VGroup(temp, arrow1)
        temp.next_to(arrow1, mn.UP)
        group_trans1.to_edge(mn.UP, buff=trans_upper_buff)
        group_trans1.to_edge(mn.LEFT, buff=mat_trans_side_buff-0.2)
        group_trans1.shift(np.array([0.2, 0, 0]))

    #middle state
    mid_equation.to_edge(mn.UP, buff=eq_edge_buff)
    mid_matrix.to_edge(mn.UP, buff=mat_upper_buff)
    group_mid = VGroup(mid_equation, mid_matrix)
    group_mid.shift(np.array([-0.1, 0, 0]))

    #second state
    if is_centered:
        s = MathTex("S=").scale(lab_scale)
        temp = VGroup(rotation_matrix, s)
        rotation_matrix.next_to(s, mn.RIGHT)
        arrow2 = Arrow(start=mn.LEFT, end=mn.RIGHT, stroke_width=a_width)
        arrow2.tip.scale(a_tip_scale)
        arrow2.set_color(mn.RED)
        group_trans2 = VGroup(temp, arrow2)
        temp.next_to(arrow2, mn.UP)
        group_trans2.to_edge(mn.UP, buff=trans_upper_buff)
        group_trans2.to_edge(mn.RIGHT, buff=mat_trans_side_buff)
    else:
        v = MathTex("v=").scale(lab_scale)
        temp = VGroup(translation_vector, v)
        translation_vector.next_to(v, mn.RIGHT)
        arrow2 = Arrow(start=mn.LEFT, end=mn.RIGHT, stroke_width=a_width)
        arrow2.tip.scale(a_tip_scale)
        arrow2.set_color(mn.RED)
        group_trans2 = VGroup(temp, arrow2)
        temp.next_to(arrow2, mn.UP)
        group_trans2.to_edge(mn.UP, buff=trans_upper_buff)
        group_trans2.to_edge(mn.RIGHT, buff=vec_trans_side_buff)

    #last state
    final_equation.to_corner(mn.UR, buff=eq_edge_buff)
    final_matrix.to_edge(mn.RIGHT, buff=mat_side_buff)
    final_matrix.to_edge(mn.UP, buff=mat_upper_buff)
    group_final = VGroup(final_equation, final_matrix)

    #returning all the states and transitions via dictionary
    returned_dict = {"initial group": group_init,
                     "first transformation": group_trans1,
                     "middle group": group_mid,
                     "second transformation": group_trans2,
                     "final group": group_final
    }

    return returned_dict

