import manim as mn
import numpy as np
from manim import Surface, VGroup

"""
This module implements the visualization of quadric surfaces in their standard forms and their 
transformations through rotations and translations. It handles the real quadrics classified by 
their canonical form.
Each surface is created using appropriate parametric equations and can be styled with 
customizable colors, opacity, and mesh properties. The module uses Manim's 3D capabilities 
to render these surfaces and their transformations.

Global Style Settings:
    Quadric Surface:
        res (int): Resolution of the surface mesh (default: 30)
        q_opacity (float): Opacity of the quadric surface (default: 0.8)
        q_color: Color of the quadric surface (default: BLUE)
        q_stroke_color: Color of the surface mesh lines (default: WHITE)
        q_stroke_width (float): Width of the surface mesh lines (default: 0.5)

    Center Dot:
        dot_scale (float): Scale of the center dot (default: 0.7)
        dot_color: Color of the center dot (default: RED)

    Plane Range:
        plane_dist (int): Distance range for plane-type quadrics (default: 4)
"""

#quadric style settings
res = 30
q_opacity = 0.8
q_color = mn.BLUE
q_stroke_color = mn.WHITE
q_stroke_width = 0.5

#center dot style settings
dot_scale = 0.7
dot_color = mn.RED

#planes range
plane_dist = 4

class NotSupportedException(Exception):
    pass

class NotAQuadricException(Exception):
    pass

def create_surface(type, A_overline_final):
    """
    Creates a 3D quadric surface based on the specified type and matrix parameters.

    Args:
        -type (int): Type of quadric surface to create. Valid types are:
            1: Real Ellipsoid
            2: Imaginary Ellipsoid (not supported)
            3: Hyperbolic Hyperboloid
            4: Elliptic Hyperboloid
            5: Cone
            7: Elliptic Paraboloid
            8: Hyperbolic Paraboloid
            9: Imaginary Elliptic Cylinder (not supported)
            10: Real Elliptic Cylinder
            11: Hyperbolic Cylinder
            12: Real Intersecting Planes
            13: Imaginary Intersecting Planes (not supported)
            14: Parabolic Cylinder
            15: Imaginary Pair of Parallel Planes (not supported)
            16: Real Pair of Parallel Planes
            17: Double Plane

        -A_overline_final (numpy.ndarray): 4x4 matrix representing the quadric's parameters

    Raises:
        ValueError: If an unsupported quadric type is specified or if the type
                   is not implemented yet (types 2, 9, 13, 15)

    Notes:
        - The surface is styled according to global style settings
        - A center dot is added to mark the origin or focal point
        - Some quadrics (types 4, 11, 12) create multiple surfaces grouped together
    """
    A = A_overline_final[0, 0]
    B = A_overline_final[1, 1]
    C = A_overline_final[2, 2]
    d = A_overline_final[3, 3]
    a = 0
    b=0
    c=0

    if not np.isclose(d, 0):
        if not np.isclose(A, 0):
            a = np.sqrt(abs(d/A))
        else:
            A=0
        if not np.isclose(B, 0):
            b = np.sqrt(abs(d/B))
        else:
            B=0
        if not np.isclose(C, 0):
            c = np.sqrt(abs(d/C))
        else:
            C=0
    else:
        if not np.isclose(A, 0):
            a = np.sqrt(abs(1/A))
        else:
            A=0
        if not np.isclose(B, 0):
            b = np.sqrt(abs(1/B))
        else:
            B=0
        if not np.isclose(C, 0):
            c = np.sqrt(abs(1/C))
        else:
            C=0

    standard_dist=a+b+c

    # case 1, REAL ELLIPSOID
    if type == 1:
        surface = mn.Surface(
            lambda u, v: np.array([
                a * np.cos(u) * np.sin(v),
                b * np.sin(u) * np.sin(v),
                c * np.cos(v)
            ]),
            u_range=[0, mn.TAU],
            v_range=[0, mn.PI],
            resolution=(res, res)
        )
        dist = standard_dist

    #case 2, IMAGINARY ELLIPSOID
    elif type == 2:
        raise  NotSupportedException("Imaginary Ellipsoids are not yet supported")

    #case 3, HYPERBOLIC HYPERBOLOID
    elif type == 3:
        if A < 0:
            surface = Surface(
                lambda u, v: np.array([
                    a * np.sinh(v),
                    b * np.cosh(v) * np.sin(u),
                    c * np.cosh(v) * np.cos(u)
                ]),
                u_range=[0, mn.TAU],
                v_range=[-a, a],
                resolution=[res,res]
            )
        elif B < 0:
            surface = Surface(
                lambda u, v: np.array([
                    a * np.cosh(v) * np.cos(u),
                    b * np.sinh(v),
                    c * np.cosh(v) * np.sin(u)
                ]),
                u_range=[0, mn.TAU],
                v_range=[-b, b],
                resolution=[res,res]
            )
        else:
            surface = Surface(
                lambda u, v: np.array([
                    a * np.cosh(v) * np.cos(u),
                    b * np.cosh(v) * np.sin(u),
                    c * np.sinh(v)
                ]),
                u_range=[0, mn.TAU],
                v_range=[-c, c],
                resolution=[res,res]
            )
        dist=standard_dist

    #case 4, ELLIPTIC HYPERBOLOID
    elif type == 4:
        if A < 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    a * np.cosh(v),
                    b * np.sinh(v) * np.sin(u),
                    c * np.sinh(v) * np.cos(u)
                ]),
                u_range=[0, mn.TAU],
                v_range=[-3 * a, 3*a],
                resolution=[res, res]
            )
            dist=3*a

        elif B < 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    a * np.sinh(v) * np.sin(u),
                    b * np.cosh(v),
                    c * np.sinh(v) * np.cos(u)
                ]),
                u_range=[0, mn.TAU],
                v_range=[-3 * b, 3 * b],
                resolution=[res, res]
            )
            dist=3*b

        else:
            surf1 = Surface(
                lambda u, v: np.array([
                    a * np.sinh(v) * np.cos(u),
                    b * np.sinh(v) * np.sin(u),
                    c * np.cosh(v)
                ]),
                u_range=[0, mn.TAU],
                v_range=[-3 * c, 3 * c],
                resolution=[res, res]
            )
            dist=3*c

        surf1.set_style(
            fill_opacity=q_opacity,
            fill_color=q_color,
            stroke_color=q_stroke_color,
            stroke_width=q_stroke_width
        )
        dot = mn.Dot3D(color=dot_color)
        dot.scale(dot_scale)
        surf1 = VGroup(surf1, dot)
        surf2 = surf1.copy()
        surf2.apply_matrix(-np.identity(3, np.float64))
        surface = VGroup(surf1, surf2)
        surface.move_to(np.array([0, 0, 0]))
        return surface, dist

    #case 5, REAL CONE
    elif type == 5:
        if A < 0:
            surface = Surface(
                lambda u, v: np.array([
                    a * v,
                    b * v * np.sin(u),
                    c * v * np.cos(u)
                ]),
                u_range=[0, mn.TAU],
                v_range=[-standard_dist, standard_dist],
                resolution=[res, res]
            )

        elif B < 0:
            surface = Surface(
                lambda u, v: np.array([
                    a * v * np.cos(u),
                    b * v,
                    c * v * np.sin(u)
                ]),
                u_range=[0, mn.TAU],
                v_range=[-standard_dist, standard_dist],
                resolution=[res, res]
            )

        else:
            surface = Surface(
                lambda u, v: np.array([
                    a * v * np.cos(u),
                    b * v * np.sin(u),
                    c * v
                ]),
                u_range=[0, mn.TAU],
                v_range=[-standard_dist, standard_dist],
                resolution=[res, res]
            )
        dist = standard_dist

    #case 6, IMAGINARY CONE
    elif type == 6:
        raise  NotSupportedException("Imaginary Cones are not yet supported")

    #case 7, ELLIPTIC PARABOLOID
    elif type == 7:
        if A_overline_final[0, 3]>0 or A_overline_final[1, 3]>0 or A_overline_final[2, 3]>0:
            k=-1
        else:
            k=1

        if A == 0:
            b = np.sqrt(abs(A_overline_final[0, 3]/B))
            c = np.sqrt(abs(A_overline_final[0, 3]/C))
            surface = Surface(
                lambda u, v: np.array([
                    k*(pow(u, 2) + pow(v, 2))/2,
                    b * v,
                    c * u
                ]),
                u_range=[-2*c, 2*c],
                v_range=[-2*b, 2*b],
                resolution=[res, res]
            )

        elif B == 0:
            a = np.sqrt(abs(A_overline_final[1, 3]/A))
            c = np.sqrt(abs(A_overline_final[1, 3]/C))
            surface = Surface(
                lambda u, v: np.array([
                    a * u,
                    k*(pow(u, 2) + pow(v, 2))/2,
                    c * v
                ]),
                u_range=[-2*a, 2*a],
                v_range=[-2*c, 2*c],
                resolution=[res, res]
            )

        else:
            a = np.sqrt(abs(A_overline_final[2, 3]/A))
            b = np.sqrt(abs(A_overline_final[2, 3]/B))
            surface = Surface(
                lambda u, v: np.array([
                    a * u,
                    b * v,
                    k*(pow(u, 2) + pow(v, 2)) / 2
                ]),
                u_range=[-2 * a, 2 * a],
                v_range=[-2 * b, 2 * b],
                resolution=[res, res]
            )

        dist = standard_dist
        surface.set_style(
            fill_opacity=q_opacity,
            fill_color=q_color,
            stroke_color=q_stroke_color,
            stroke_width=q_stroke_width,
        )
        dot = mn.Dot3D(color=dot_color)
        dot.scale(dot_scale)
        surface = VGroup(surface, dot)
        obj = surface[1]
        point_position = obj.get_center()
        surface.shift(-point_position)
        return surface, dist

    #case 8, HYPERBOLIC PARABOLOID
    elif type == 8:
        if A_overline_final[0, 3]>0 or A_overline_final[1, 3]>0 or A_overline_final[2, 3]>0:
            A = -A
            B = -B
            C = -C

        if A == 0 and B > 0:
            b = np.sqrt(abs(A_overline_final[0, 3]/B))
            c = np.sqrt(abs(A_overline_final[0, 3]/C))
            surface = Surface(
                lambda u, v: np.array([
                    (pow(v, 2) - pow(u, 2))/2,
                    b * v,
                    c * u
                ]),
                u_range=[-2*c, 2*c],
                v_range=[-2*b, 2*b],
                resolution=[res, res]
            )

        elif A == 0 and C > 0:
            b = np.sqrt(abs(A_overline_final[0, 3]/B))
            c = np.sqrt(abs(A_overline_final[0, 3]/C))
            surface = Surface(
                lambda u, v: np.array([
                    (pow(u, 2) - pow(v, 2))/2,
                    b * v,
                    c * u
                ]),
                u_range=[-2*c, 2*c],
                v_range=[-2*b, 2*b],
                resolution=[res, res]
            )

        elif B == 0 and A > 0:
            a = np.sqrt(abs(A_overline_final[1, 3]/A))
            c = np.sqrt(abs(A_overline_final[1, 3]/C))
            surface = Surface(
                lambda u, v: np.array([
                    a * u,
                    (pow(u, 2) - pow(v, 2))/2,
                    c * v
                ]),
                u_range=[-2*a, 2*a],
                v_range=[-2*c, 2*c],
                resolution=[res, res]
            )

        elif B == 0 and C > 0:
            a = np.sqrt(abs(A_overline_final[1, 3]/A))
            c = np.sqrt(abs(A_overline_final[1, 3]/C))
            surface = Surface(
                lambda u, v: np.array([
                    a * u,
                    (pow(v, 2) - pow(u, 2))/2,
                    c * v
                ]),
                u_range=[-2*a, 2*a],
                v_range=[-2*c, 2*c],
                resolution=[res, res]
            )

        elif C == 0 and A > 0:
            a = np.sqrt(abs(A_overline_final[2, 3]/A))
            b = np.sqrt(abs(A_overline_final[2, 3]/B))
            surface = Surface(
                lambda u, v: np.array([
                    a * u,
                    b * v,
                    (pow(u, 2) - pow(v, 2)) / 2
                ]),
                u_range=[-2 * a, 2 * a],
                v_range=[-2 * b, 2 * b],
                resolution=[res, res]
            )

        else:
            a = np.sqrt(abs(A_overline_final[2, 3]/A))
            b = np.sqrt(abs(A_overline_final[2, 3]/B))
            surface = Surface(
                lambda u, v: np.array([
                    a * u,
                    b * v,
                    (pow(v, 2) - pow(u, 2)) / 2
                ]),
                u_range=[-2 * a, 2 * a],
                v_range=[-2 * b, 2 * b],
                resolution=[res, res]
            )

        dist = standard_dist

    #case 9, IMAGINARY ELLIPTIC CYLINDER
    elif type == 9:
        raise  NotSupportedException("Imaginary Elliptic Cylinders are not yet supported")

    #case 10, REAL ELLIPTIC CYLINDER
    elif type == 10:
        if A == 0:
            surface = Surface(
                lambda u, v: np.array([
                    v,
                    b * np.cos(u),
                    c * np.sin(u)
                ]),
                u_range=[0, mn.TAU],
                v_range=[-c-b, c+b],
                resolution=[res, res]
            )

        elif B == 0:
            surface = Surface(
                lambda u, v: np.array([
                    a * np.sin(u),
                    v,
                    b * np.cos(u)
                ]),
                u_range=[0, mn.TAU],
                v_range=[-a-c, a+c],
                resolution=[res, res]
            )

        else:
            surface = Surface(
                lambda u, v: np.array([
                    a * np.sin(u),
                    b * np.cos(u),
                    v
                ]),
                u_range=[0, mn.TAU],
                v_range=[-a - b, a + b],
                resolution=[res, res]
            )

        dist = standard_dist

    #case 11, HYPERBOLIC CYLINDER
    elif type == 11:
        if A == 0 and B > 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    v,
                    b * np.cosh(u),
                    c * np.sinh(u)
                ]),
                u_range=[-c - b, c + b],
                v_range=[-c - b, c + b],
                resolution=[res, res]
            )
        elif A == 0 and C > 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    v,
                    b * np.sinh(u),
                    c * np.cosh(u)
                ]),
                u_range=[-c - b, c + b],
                v_range=[-c - b, c + b],
                resolution=[res, res]
            )

        elif B == 0 and A > 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    a * np.cosh(u),
                    v,
                    c * np.sinh(u)
                ]),
                u_range=[-a - c, a + c],
                v_range=[-a - c, a + c],
                resolution=[res, res]
            )
        elif B == 0 and C > 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    a * np.sinh(u),
                    v,
                    c * np.cosh(u)
                ]),
                u_range=[-a - c, a + c],
                v_range=[-a - c, a + c],
                resolution=[res, res]
            )

        elif C == 0 and A > 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    a * np.cosh(u),
                    b * np.sinh(u),
                    v
                ]),
                u_range=[-a - b, a + b],
                v_range=[-a - b, a + b],
                resolution=[res, res]
            )
        else:
            surf1 = Surface(
                lambda u, v: np.array([
                    a * np.sinh(u),
                    b * np.cosh(u),
                    v
                ]),
                u_range=[-a - b, a + b],
                v_range=[-a - b, a + b],
                resolution=[res, res]
            )

        dist = 2 * standard_dist

        dot = mn.Dot3D(color=dot_color)
        dot.scale(dot_scale)
        surf1 = VGroup(surf1, dot)
        surf2 = surf1.copy()
        surf2.apply_matrix(-np.identity(3, np.float64))
        surface = VGroup(surf1, surf2)
        surface.move_to(np.array([0, 0, 0]))
        return surface, dist

    #case 12 REAL INTERSECTING PLANES
    elif type == 12:
        if A == 0 and B > 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    v,
                    (b/c)*u,
                    u
                ]),
                u_range=[-2*(b+c), 2*(b+c)],
                v_range=[-2*(b+c), 2*(b+c)],
                resolution=[res, res]
            )
            surf2 = Surface(
                lambda u, v: np.array([
                    v,
                    -(b/c)*u,
                    u
                ]),
                u_range=[-2*(b+c), 2*(b+c)],
                v_range=[-2*(b+c), 2*(b+c)],
                resolution=[res, res]
            )
        elif A == 0 and C > 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    v,
                    u,
                    (c / b)*u
                ]),
                u_range=[-2*(b+c), 2*(b+c)],
                v_range=[-2*(b+c), 2*(b+c)],
                resolution=[res, res]
            )
            surf2 = Surface(
                lambda u, v: np.array([
                    v,
                    u,
                    -(c / b)*u
                ]),
                u_range=[-2*(b+c), 2*(b+c)],
                v_range=[-2*(b+c), 2*(b+c)],
                resolution=[res, res]
            )
        elif B == 0 and A > 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    (a / c) * u,
                    v,
                    u
                ]),
                u_range=[-2 * (a + c), 2 * (a + c)],
                v_range=[-2 * (a + c), 2 * (a + c)],
                resolution=[res, res]
            )
            surf2 = Surface(
                lambda u, v: np.array([
                    -(a / c) * u,
                    v,
                    u
                ]),
                u_range=[-2 * (a + c), 2 * (a + c)],
                v_range=[-2 * (a + c), 2 * (a + c)],
                resolution=[res, res]
            )
        elif B == 0 and C > 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    u,
                    v,
                    (c / a) * u
                ]),
                u_range=[-2 * (a + c), 2 * (a + c)],
                v_range=[-2 * (a + c), 2 * (a + c)],
                resolution=[res, res]
            )
            surf2 = Surface(
                lambda u, v: np.array([
                    u,
                    v,
                    -(c / a) * u
                ]),
                u_range=[-2 * (a + c), 2 * (a + c)],
                v_range=[-2 * (a + c), 2 * (a + c)],
                resolution=[res, res]
            )
        elif C == 0 and A > 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    (a / b) * u,
                    u,
                    v
                ]),
                u_range=[-2 * (a + b), 2 * (a + b)],
                v_range=[-2 * (a + b), 2 * (a + b)],
                resolution=[res, res]
            )
            surf2 = Surface(
                lambda u, v: np.array([
                    -(a / b) * u,
                    u,
                    v
                ]),
                u_range=[-2 * (a + b), 2 * (a + b)],
                v_range=[-2 * (a + b), 2 * (a + b)],
                resolution=[res, res]
            )
        else:
            surf1 = Surface(
                lambda u, v: np.array([
                    u,
                    (b / a) * u,
                    v
                ]),
                u_range=[-2 * (a + b), 2 * (a + b)],
                v_range=[-2 * (a + b), 2 * (a + b)],
                resolution=[res, res]
            )
            surf2 = Surface(
                lambda u, v: np.array([
                    u,
                    -(b / a) * u,
                    v
                ]),
                u_range=[-2 * (a + b), 2 * (a + b)],
                v_range=[-2 * (a + b), 2 * (a + b)],
                resolution=[res, res]
            )
        dist = plane_dist+plane_dist/2
        dot = mn.Dot3D(color=dot_color)
        dot.scale(dot_scale)
        surface = VGroup(surf1, surf2, dot)
        return surface, 6

    #case 13, IMAGINARY INTERSECTING PLANES
    elif type == 13:
        raise  NotSupportedException("Imaginary Intersecting Planes are not yet supported")

    #case 14, PARABOLIC CYLINDER
    elif type == 14:
        if A_overline_final[0, 3]>0 or A_overline_final[1, 3]>0 or A_overline_final[2, 3]>0:
            k=-1
        else:
            k=1

        if A != 0 and A_overline_final[1, 3] != 0:
            a = np.sqrt(abs(A_overline_final[1, 3]/A))
            surface = Surface(
                lambda u, v: np.array([
                    u,
                    k * a * pow(u, 2) / 2,
                    v
                ]),
                u_range=[-a, a],
                v_range=[-a, a],
                resolution=[res, res]
            )
            dist=2*a
        elif A != 0 and A_overline_final[2, 3] != 0:
            a = np.sqrt(abs(A_overline_final[2, 3]/A))
            surface = Surface(
                lambda u, v: np.array([
                    u,
                    v,
                    k * a * pow(u, 2) / 2
                ]),
                u_range=[-a, a],
                v_range=[-a, a],
                resolution=[res, res]
            )
            dist=2*a
        elif B != 0 and A_overline_final[0, 3]:
            b = np.sqrt(abs(A_overline_final[0, 3]/A))
            surface = Surface(
                lambda u, v: np.array([
                    k * b * pow(u, 2) / 2,
                    u,
                    v
                ]),
                u_range=[-b, b],
                v_range=[-b, b],
                resolution=[res, res]
            )
            dist=2*b
        elif B != 0 and A_overline_final[2, 3]:
            b = np.sqrt(abs(A_overline_final[2, 3] / A))
            surface = Surface(
                lambda u, v: np.array([
                    v,
                    u,
                    k * b * pow(u, 2) / 2,

                ]),
                u_range=[-b, b],
                v_range=[-b, b],
                resolution=[res, res]
            )
            dist=2*b
        elif C != 0 and A_overline_final[0, 3]:
            c = np.sqrt(abs(A_overline_final[0, 3] / A))
            surface = Surface(
                lambda u, v: np.array([
                    k * c * pow(u, 2) / 2,
                    v,
                    u
                ]),
                u_range=[-c, c],
                v_range=[-c, c],
                resolution=[res, res]
            )
            dist=2*c
        else:
            c = np.sqrt(abs(A_overline_final[1, 3] / A))
            surface = Surface(
                lambda u, v: np.array([
                    v,
                    k * c * pow(u, 2) / 2,
                    u
                ]),
                u_range=[-c, c],
                v_range=[-c, c],
                resolution=[res, res]
            )
            dist=2*c
        surface.set_style(
            fill_opacity=q_opacity,
            fill_color=q_color,
            stroke_color=q_stroke_color,
            stroke_width=q_stroke_width
        )
        dot = mn.Dot3D(color=dot_color)
        dot.scale(dot_scale)
        surface = VGroup(surface, dot)
        obj = surface[1]
        point_position = obj.get_center()
        surface.shift(-point_position)
        return surface, dist

    #case 15, IMAGINARY PAIR OF PARALLEL PLANES
    elif type == 15:
        raise  NotSupportedException("Imaginary pair of Parallel Planes are not yet supported")

    #case 16, REAL PAIR OF PARALLEL PLANES
    elif type == 16:
        if A != 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    a,
                    u,
                    v
                ]),
            u_range = [-plane_dist, plane_dist],
            v_range = [-plane_dist, plane_dist],
            resolution = [res, res]
            )
            rot_axes = np.array([0, 1, 0])
        elif B != 0:
            surf1 = Surface(
                lambda u, v: np.array([
                    u,
                    b,
                    v
                ]),
            u_range = [-plane_dist, plane_dist],
            v_range = [-plane_dist, plane_dist],
            resolution = [res, res]
            )
            rot_axes = np.array([1, 0, 0])
        else:
            surf1 = Surface(
                lambda u, v: np.array([
                    v,
                    u,
                    c
                ]),
            u_range = [-plane_dist, plane_dist],
            v_range = [-plane_dist, plane_dist],
            resolution = [res, res]
            )
            rot_axes = np.array([1, 0, 0])

        surf2 = surf1.copy()
        surf2.rotate(mn.PI, axis=rot_axes, about_point=mn.ORIGIN)
        surface = VGroup(surf1,  surf2)
        dist = plane_dist + plane_dist/2

    #caso 17 DOUBLE PLANE
    elif type == 17:
        if A != 0:
            surface = Surface(
                lambda u, v: np.array([
                    0,
                    u,
                    v
                ]),
                u_range=[-plane_dist, plane_dist],
                v_range=[-plane_dist, plane_dist],
                resolution=[res, res]
            )
        elif B != 0:
            surface = Surface(
                lambda u, v: np.array([
                    v,
                    0,
                    u
                ]),
                u_range=[-plane_dist, plane_dist],
                v_range=[-plane_dist, plane_dist],
                resolution=[res, res]
            )
        else:
            surface = Surface(
                lambda u, v: np.array([
                    v,
                    u,
                    0
                ]),
                u_range=[-plane_dist, plane_dist],
                v_range=[-plane_dist, plane_dist],
                resolution=[res, res]
            )
        dist=plane_dist+plane_dist/2

    else:
        raise  NotAQuadricException("Not a quadric")

    #quadric style
    surface.set_style(
        fill_opacity=q_opacity,
        fill_color=q_color,
        stroke_color=q_stroke_color,
        stroke_width=q_stroke_width,
    )
    dot = mn.Dot3D(color=dot_color)
    dot.scale(dot_scale)
    surface = VGroup(surface, dot)
    surface.move_to(surface[1].get_center())

    return surface, dist