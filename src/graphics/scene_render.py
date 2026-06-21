import math
import numpy as np
import manim as mn
from manim import VGroup, MathTex, FadeIn
from src.graphics.create_quadric_surface import QuadricSurfaceFactory
from src.graphics.create_text_overlay import TextOverlayBuilder
from src.numerical.models import CanonicalizationResult, QuadricType

wait_time = 5

"""
This module creates 3D animations of quadric surfaces using the Manim library.
It visualizes how quadric surfaces transform through translation and rotation operations,
showing the relationship between different forms of quadric equations.

Global Variables:
    wait_time (int): Duration in seconds for animation transitions (default: 5)
"""

#funcion to generate the axes
def get_perfect_axis(dist: float) -> tuple[VGroup, MathTex]:
    """
    Creates and configures the 3D coordinate system for the animation.

    Args:
        dist (float): Determines the range of the axes
    """
    #scales
    label_scale = 0.7
    z_scale = 0.5
    z_opacity = 0.8
    labels_buff = 0.2

    dist = math.ceil(dist)
    axes3d: mn.ThreeDAxes = mn.ThreeDAxes(x_range=([-dist, dist, 1]), y_range=([-dist, dist, 1]), z_range=([-dist/2, dist/2, 2])).add_coordinates()
    axes3d.z_axis.tip.scale(z_scale).set_opacity(z_opacity)
    axes3d.z_axis.set_opacity(z_opacity)
    axes3d.z_axis.numbers.set_opacity(0)

    x: MathTex = (
        MathTex("X", color=mn.ORANGE).scale(label_scale)
        .rotate(90 * mn.DEGREES, axis=mn.X_AXIS)
        .rotate(90 * mn.DEGREES, axis=mn.Z_AXIS)
        .move_to(axes3d.c2p(dist+labels_buff, 0, 0))
    )
    y: MathTex = (
        MathTex("Y", color=mn.ORANGE).scale(label_scale)
        .rotate(90 * mn.DEGREES, axis=mn.X_AXIS)
        .rotate(90 * mn.DEGREES, axis=mn.Z_AXIS)
        .move_to(axes3d.c2p(0, dist+labels_buff, 0))
    )
    z: MathTex = (
        MathTex("Z", color=mn.ORANGE).move_to(np.array([labels_buff, 0, -dist/2-labels_buff])).scale(label_scale)
    )

    result = VGroup(axes3d, x, y)

    return result, z




class SceneRender(mn.ThreeDScene):
    """
    Creates animations of quadric surface transformations in 3D space.

    This class handles the visualization of quadric surfaces and their transformations
    through translation and rotation operations. It can handle both centered and
    non-centered quadrics with different animation sequences for each case.

    Args:
        result: CanonicalizationResult
            Validated numerical artifacts consumed by the animation.
        return: SceneRender
            Manim scene configured from the typed result.

    Notes:
        - Requires custom modules CreateQuadric and CreateTextOverlay
        - Uses Manim's 3D scene capabilities for smooth transitions
        - Animation sequence varies based on whether quadric is centered
    """
    result: CanonicalizationResult
    quadric_type: QuadricType
    is_centered: bool
    translation_vector: np.ndarray
    rotation_matrix: np.ndarray
    A_overline_final: np.ndarray
    A_overline_mid: np.ndarray
    A_overline_initial: np.ndarray
    eq_final: object
    eq_mid: object
    eq_initial: object

    def __init__(self, result: CanonicalizationResult) -> None:
        """
        Initialize the QuadricAnimation with the given parameters.

        Special Cases:
            - Quadric type 14 (Parabolic Cylinder) uses the centered animation order.
        """

        super().__init__()
        self.result = result
        self.quadric_type = result.quadric_type
        self.is_centered = result.centered
        # Parabolic cylinders use translation before rotation in the specialized algorithm.
        if self.quadric_type is QuadricType.PARABOLIC_CYLINDER:
            self.is_centered=True
        
        self.translation_vector = result.translation_vector.squeeze()
        self.rotation_matrix = result.rotation_matrix
        
        self.A_overline_final = result.final_matrix
        self.A_overline_mid = result.middle_matrix
        self.A_overline_initial = result.initial_matrix
        
        self.eq_final = result.final_equation
        self.eq_mid = result.middle_equation
        self.eq_initial = result.initial_equation



    def construct(self) -> None:
        """
        Constructs and executes the main animation sequence.

        This method handles:
            1. Initial setup of text overlays, surface, camera, and axes
            2. Animation sequence execution based on quadric type
            3. Transformation visualizations including rotation and translation
            4. Text overlay management for equations and transformations
            5. Camera movement and timing control
        """
        #get text graphics
        groups = TextOverlayBuilder().build(self.is_centered, self.eq_initial, self.A_overline_initial, self.eq_mid,
                                            self.A_overline_mid, self.eq_final, self.A_overline_final, self.translation_vector,
                                            self.rotation_matrix)
        text_init = groups.initial
        text_trans1 = groups.first_transformation
        text_mid = groups.middle
        text_trans2 = groups.second_transformation
        text_final = groups.final

        #create quadric object
        surface_build = QuadricSurfaceFactory().create(self.quadric_type, self.A_overline_final)
        surface = surface_build.surface
        dist = surface_build.axis_distance

        #scene, camera and axes setup
        self.set_camera_orientation(phi=65 * mn.DEGREES, theta=-20 * mn.DEGREES)
        axes, z_label = get_perfect_axis(dist)
        self.add(axes)
        self.add_fixed_orientation_mobjects(z_label)
        self.begin_ambient_camera_rotation(0.1)
        self.add_fixed_in_frame_mobjects(text_init)

        if self.is_centered:
            #setup of the qadric in the non canonic form
            surface.apply_matrix(self.rotation_matrix)
            surface.shift(-self.translation_vector)

            #add the quadric to the scene
            self.add(surface)
            self.wait(wait_time)

            #translation
            self.add_fixed_in_frame_mobjects(text_trans1)
            self.play(mn.GrowFromEdge(text_trans1, mn.LEFT))
            self.play(surface.animate.shift(self.translation_vector), run_time=wait_time)

            #add middle text
            self.add_fixed_in_frame_mobjects(text_mid)
            self.play(FadeIn(text_mid))
            self.wait(2)

            #rotation
            self.add_fixed_in_frame_mobjects(text_trans2)
            self.play(mn.GrowFromEdge(text_trans2, mn.LEFT))
            self.play(mn.ApplyMatrix(np.transpose(self.rotation_matrix), surface), run_time=wait_time)
            self.wait(2)

            #add final text
            self.add_fixed_in_frame_mobjects(text_final)
            self.play(FadeIn(text_final))

        else:
            #setup of the qadric in the non canonic form
            surface.shift(self.translation_vector)
            surface.apply_matrix(self.rotation_matrix)

            #add the quadric to the scene
            self.add(surface)
            self.wait(wait_time)

            #rotation
            self.add_fixed_in_frame_mobjects(text_trans1)
            self.play(mn.GrowFromEdge(text_trans1, mn.LEFT))
            self.play(mn.ApplyMatrix(np.transpose(self.rotation_matrix), surface), run_time=wait_time)

            #add middle text
            self.add_fixed_in_frame_mobjects(text_mid)
            self.play(FadeIn(text_mid))
            self.wait(2)

            #traslation
            self.add_fixed_in_frame_mobjects(text_trans2)
            self.play(mn.GrowFromEdge(text_trans2, mn.LEFT))
            self.play(surface.animate.shift(-self.translation_vector), run_time=wait_time)
            self.wait(2)

            #add final text
            self.add_fixed_in_frame_mobjects(text_final)
            self.play(FadeIn(text_final))

        #end of scene
        self.wait(10)
