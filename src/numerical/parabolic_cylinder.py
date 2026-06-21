"""
Canonicalize parabolic cylinders through their vertex-line geometry.

Run the focused tests with ``python -m pytest tests/test_transformer.py -q``.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from src.numerical.algebra import clean_near_zero, normalize_integer_coefficients
from src.numerical.misc import string2sympy
from src.numerical.symbols import x, y, z

"""
This module computes the canonical metric form of the parabolic cylinder and its transformations.
"""

NUMERICAL_TOLERANCE = 1e-10


def non_null_eigvalue(A: sp.Matrix) -> tuple[float, np.ndarray]:
    A = np.array(A, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.where(abs(eigenvalues) > 1e-10)[0][0] # Find the index of the non-null eigenvalue
    eigenvector_found = eigenvectors[:, idx] # Extract the corresponding eigenvector
    eigenvalue_found = eigenvalues[idx] # Extract the corresponding eigenvalue
    return float(eigenvalue_found), eigenvector_found

def solve_linear_system_least_squares(A: sp.Matrix, b: sp.Matrix) -> tuple[sp.Matrix, tuple[sp.Symbol, ...]]:
    ATA = A.transpose() * A # Compute A^T * A and A^T * b
    ATb = A.transpose() * b
    null_space = ATA.nullspace() # Calculate the null space of A^T * A
    if (len(null_space) != 2):
        raise ValueError("The result must be a plane")
    free_params = sp.symbols('t0:%d' % len(null_space))
    particular = ATA.pinv() * ATb # Find a particular solution using the pseudo-inverse
    general_solution = particular # The general solution is: particular + sum(t_i * null_i)
    for t, null_vector in zip(free_params, null_space):
        general_solution = general_solution + t * null_vector
    return general_solution, free_params

def substitute_vector_in_quadric(quadric_polynomial: sp.Expr, plane_parametric: list[sp.Expr], t0: sp.Symbol, t1: sp.Symbol) -> tuple[sp.Expr, dict[sp.Symbol, sp.Expr]]:
    if not hasattr(plane_parametric, '__len__'):
        raise ValueError("Vector must be array-like")
    substitution = {
        x: plane_parametric[0],
        y: plane_parametric[1],
        z: plane_parametric[2]
    }
    result = quadric_polynomial.subs(substitution) # Perform substitution
    return result, substitution

def plane_intersection_quadric(quadric_sub_equation: sp.Expr, plane_parametric: dict[sp.Symbol, sp.Expr], t0: sp.Symbol, t1: sp.Symbol) -> tuple[dict[sp.Symbol, sp.Expr], sp.Symbol]:
    t = sp.symbols('t')
    t0_solutions = sp.solve(quadric_sub_equation, t0)
    if t0_solutions:
        t0_expr = t0_solutions[0]
        substitution_dict = {t0:t0_expr}
        final_substitution_dict = {t1: t}  # after the first substitution I will have only one parameter
        parametric_intersection = {
            x: (plane_parametric[x].subs(substitution_dict)).subs(final_substitution_dict),
            y: (plane_parametric[y].subs(substitution_dict)).subs(final_substitution_dict),
            z: (plane_parametric[z].subs(substitution_dict)).subs(final_substitution_dict),
        }
    else:
        t1_solutions = sp.solve(quadric_sub_equation, t1)
        if not t1_solutions:
            raise ValueError("quadric-plane intersection cannot be parameterized")
        t1_expr = t1_solutions[0]
        substitution_dict = {t1: t1_expr}
        final_substitution_dict = {t0: t} # after the first substitution I will have only one parameter
        parametric_intersection = {
            x: (plane_parametric[x].subs(substitution_dict)).subs(final_substitution_dict),
            y: (plane_parametric[y].subs(substitution_dict)).subs(final_substitution_dict),
            z: (plane_parametric[z].subs(substitution_dict)).subs(final_substitution_dict),
        }
    return parametric_intersection, t

def convert_poly_coeffs(expr: sp.Expr) -> sp.Expr:
    """
    Convert float coefficients ending in .00 to integers in a sympy polynomial expression.

    Parameters:
    expr: A sympy expression (polynomial in x, y, z)

    Returns:
    A sympy expression with converted coefficients
    """
    return normalize_integer_coefficients(expr, NUMERICAL_TOLERANCE)

def obtain_vertex(A_overline: sp.Matrix, A: sp.Matrix, b: sp.Matrix, eq: str) -> tuple[sp.Matrix, dict[sp.Symbol, sp.Expr], sp.Symbol]:
    quadric_expr = string2sympy(eq).as_expr()
    # the resulting plane, intersected with the quadric, shhould yield the line of the vertex
    plane_parametric, params = solve_linear_system_least_squares(A, b)
    t0 = params[0]
    t1 = params[1]
    plane_parametric = sp.Matrix([[-plane_parametric[0]], [-plane_parametric[1]], [-plane_parametric[2]]])
    substituted_quadric, plane_parametric = substitute_vector_in_quadric(quadric_expr, list(plane_parametric), t0, t1)
    substituted_quadric = substituted_quadric.simplify()
    substituted_quadric = convert_poly_coeffs(substituted_quadric)
    # at this point q with the substituted plane parameters is a linear equation in t0, t1:
    # if I isolate t0 (or t1) I then substitute an arbitrary value in the parametric equations of the plane
    # and I then can obtain the vertex
    parametric_eq_vertex_line, t = plane_intersection_quadric(substituted_quadric, plane_parametric, t0, t1)
    i = 1
    vertex = sp.Matrix([
        (parametric_eq_vertex_line[x].subs(t, i)).simplify(),
        (parametric_eq_vertex_line[y].subs(t, i)).simplify(),
        (parametric_eq_vertex_line[z].subs(t, i)).simplify()
    ])
    while (all(coord == 0 for coord in vertex)==True): # be sure that the obtained vector isn't 0,0,0
        vertex = sp.Matrix([
            parametric_eq_vertex_line[x].subs(t, i),
            parametric_eq_vertex_line[y].subs(t, i),
            parametric_eq_vertex_line[z].subs(t, i)
        ])
        i = i + 1
    return vertex, parametric_eq_vertex_line, t

def rotation_matrix_parabolic_cylinder(A_overline: sp.Matrix, A: sp.Matrix, b: sp.Matrix, vertex: sp.Matrix, parametric_eq_vertex_line: dict[sp.Symbol, sp.Expr], t: sp.Symbol) -> sp.Matrix:
    # vector 1: eigenspace corresponding to the non-zero eigenvalue
    _, v_1 = non_null_eigvalue(A)
    # vector 2: opposite of the normal vector of the tangent plane at vertex to the quadric, with equation vertex_overline^t * A_overline * x_overline
    expr_tangent_plane_in_vertex = (sp.Matrix([[vertex[0], vertex[1], vertex[2], 1]]) * A_overline * sp.Matrix([[x], [y], [z], [1]]))[0]
    v_2 = sp.Matrix([[-expr_tangent_plane_in_vertex.coeff(x)], [-expr_tangent_plane_in_vertex.coeff(y)], [-expr_tangent_plane_in_vertex.coeff(z)]])
    # vector 3: a vector that forms with v_1 and v_2 a positive basis and has the direction of the line of vertices
    line_direction = {
        symbol: parametric_eq_vertex_line[symbol] - parametric_eq_vertex_line[symbol].coeff(t, 0)
        for symbol in (x, y, z)
    }
    v_3_temp = sp.Matrix([line_direction[x], line_direction[y], line_direction[z]])
    S = sp.Matrix([[v_1[0], v_2[0], 0], [v_1[1], v_2[1], 0], [v_1[2], v_2[2], 0]])
    normalized_columns = [S.col(i) / S.col(i).norm() for i in range(2)]
    S = sp.Matrix.hstack(normalized_columns[0], normalized_columns[1], sp.zeros(3, 1))
    S[0,2] = v_3_temp[0]
    S[1,2] = v_3_temp[1]
    S[2,2] = v_3_temp[2]
    det_S = S.det()
    solutions = sp.solve(sp.Poly(det_S - 1, t), t)
    if not solutions:
        raise ValueError("cannot construct a positively oriented rotation matrix")
    t_value = solutions[0]
    substitution_dict = {t : t_value}
    v_3 = sp.Matrix([line_direction[symbol].subs(substitution_dict).simplify() for symbol in (x, y, z)])
    # create rotation matrix
    S[0,2] = v_3[0]
    S[1,2] = v_3[1]
    S[2,2] = v_3[2]
    S_norm = S
    return S_norm

def parabolic_cylinder_canonize(A_overline: np.ndarray, A: np.ndarray, b: np.ndarray, eq: str, A_overline_og: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A_overline = sp.Matrix(A_overline)
    A = sp.Matrix(A)
    b = sp.Matrix(b)
    vertex, parametric_eq_vertex_line, t = obtain_vertex(A_overline,A,b,eq) # ottiene un vertice
    S_norm = rotation_matrix_parabolic_cylinder(A_overline, A, b, vertex, parametric_eq_vertex_line, t) # ottiene una matrice di rot
    # create matrix P_overline and calculate A_overline after translation and A_overline after rotation
    transl_vector = vertex
    P_overline_trasl = sp.BlockMatrix([[sp.BlockMatrix([sp.eye(3), transl_vector]).as_explicit()], [sp.Matrix([[0, 0, 0, 1]])]]).as_explicit()
    P_overline_trasl = np.array(P_overline_trasl, dtype=np.float64)
    A_overline_trasl = (np.transpose(P_overline_trasl) @ A_overline_og) @ P_overline_trasl
    A_overline_trasl = clean_near_zero(A_overline_trasl, NUMERICAL_TOLERANCE)
    P_overline_tot = sp.BlockMatrix([[sp.BlockMatrix([S_norm, transl_vector]).as_explicit()], [sp.Matrix([[0, 0, 0, 1]])]]).as_explicit()
    P_overline_tot = np.array(P_overline_tot, dtype=np.float64)
    P_overline_tot = clean_near_zero(P_overline_tot, NUMERICAL_TOLERANCE)
    A_overline_CMF = (np.transpose(P_overline_tot) @ A_overline_og) @ P_overline_tot
    A_overline_CMF = clean_near_zero(A_overline_CMF, NUMERICAL_TOLERANCE)
    return A_overline_CMF, np.array(S_norm, dtype=np.float64), np.array(transl_vector, dtype=np.float64), A_overline_trasl
