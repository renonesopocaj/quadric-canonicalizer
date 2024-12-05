import sympy as sp
from misc import *
from checker import *

"""
This module computes the canonical metric form of the parabolic cylinder and its transformations.
"""

global x, y, z
x, y, z = sp.symbols('x y z')

def non_null_eigvalue(A):
    A = np.array(A, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.where(abs(eigenvalues) > 1e-10)[0][0] # Find the index of the non-null eigenvalue
    eigenvector_found = eigenvectors[:, idx] # Extract the corresponding eigenvector
    eigenvalue_found = eigenvalues[idx] # Extract the corresponding eigenvalue
    eigenvalue_found = sp.Matrix(eigenvalue_found)
    return eigenvalue_found, eigenvector_found

def solve_linear_system_least_squares(A, b):
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

def substitute_vector_in_quadric(quadric_polynomial, plane_parametric, t0, t1):
    if not hasattr(plane_parametric, '__len__'):
        raise ValueError("Vector must be array-like")
    plane_parametric = {
        x: plane_parametric[0],
        y: plane_parametric[1],
        z: plane_parametric[2]
    }
    result = quadric_polynomial.subs(plane_parametric) # Perform substitution
    print(f"result is {result}")
    return result, plane_parametric

def plane_intersection_quadric(quadric_sub_equation, plane_parametric, t0, t1):
    t = sp.symbols('t')
    print(f"quadric_sub_equation e' {quadric_sub_equation} e plane_parametric e' {plane_parametric}")
    try:
        t0_expr = sp.solve(quadric_sub_equation, t0)[0] # l'expr sara' in t1
        substitution_dict = {t0:t0_expr}
        final_substitution_dict = {t1: t}  # after the first substitution I will have only one parameter
        parametric_intersection = {
            x: (plane_parametric[x].subs(substitution_dict)).subs(final_substitution_dict),
            y: (plane_parametric[y].subs(substitution_dict)).subs(final_substitution_dict),
            z: (plane_parametric[z].subs(substitution_dict)).subs(final_substitution_dict),
        }
    except:
        try:
            t1_expr = sp.solve(quadric_sub_equation, t1)[0] # l'expr sara' in t0
            substitution_dict = {t1: t1_expr}
            final_substitution_dict = {t0: t} # after the first substitution I will have only one parameter
            parametric_intersection = {
                x: (plane_parametric[x].subs(substitution_dict)).subs(final_substitution_dict),
                y: (plane_parametric[y].subs(substitution_dict)).subs(final_substitution_dict),
                z: (plane_parametric[z].subs(substitution_dict)).subs(final_substitution_dict),
            }
        except:
            raise
    return parametric_intersection, t

def convert_poly_coeffs(expr):
    """
    Convert float coefficients ending in .00 to integers in a sympy polynomial expression.

    Parameters:
    expr: A sympy expression (polynomial in x, y, z)

    Returns:
    A sympy expression with converted coefficients
    """
    # Expand the expression to get standard form
    expanded = sp.expand(expr)
    # Function to convert a single coefficient
    def convert_coeff(coeff):
        if isinstance(coeff, sp.core.numbers.Float):
            # Convert to float and check if it's effectively an integer
            float_val = float(coeff)
            if abs(float_val - round(float_val)) < 1e-10:
                return int(round(float_val))
        return coeff
    # Get all terms and their coefficients
    terms = expanded.as_coefficients_dict()
    # Create new expression with converted coefficients
    new_expr = 0
    for term, coeff in terms.items():
        new_coeff = convert_coeff(coeff)
        new_expr += new_coeff * term
    return new_expr

def obtain_vertex(A_overline, A,b,eq):
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

def rotation_matrix_parabolic_cylinder(A_overline, A, b, vertex, parametric_eq_vertex_line, t):
    # vector 1: eigenspace corresponding to the non-zero eigenvalue
    v_1_eigenvalue, v_1 = non_null_eigvalue(A)
    # vector 2: opposite of the normal vector of the tangent plane at vertex to the quadric, with equation vertex_overline^t * A_overline * x_overline
    expr_tangent_plane_in_vertex = (sp.Matrix([[vertex[0], vertex[1], vertex[2], 1]]) * A_overline * sp.Matrix([[x], [y], [z], [1]]))[0]
    v_2 = sp.Matrix([[-expr_tangent_plane_in_vertex.coeff(x)], [-expr_tangent_plane_in_vertex.coeff(y)], [-expr_tangent_plane_in_vertex.coeff(z)]])
    # vector 3: a vector that forms with v_1 and v_2 a positive basis and has the direction of the line of vertices
    parametric_eq_vertex_line[x] = parametric_eq_vertex_line[x] - parametric_eq_vertex_line[x].coeff(t, 0)
    parametric_eq_vertex_line[y] = parametric_eq_vertex_line[y] - parametric_eq_vertex_line[y].coeff(t, 0)
    parametric_eq_vertex_line[z] = parametric_eq_vertex_line[z] - parametric_eq_vertex_line[z].coeff(t, 0)
    v_3_temp = sp.Matrix([[parametric_eq_vertex_line[x]], [parametric_eq_vertex_line[y]], [parametric_eq_vertex_line[z]]])
    S = sp.Matrix([[v_1[0], v_2[0], 0], [v_1[1], v_2[1], 0], [v_1[2], v_2[2], 0]])
    S = sp.Matrix([[(S.col(i) / (S.col(i).norm())) if (S.col(i).norm()).simplify() != 0 else sp.Matrix([0,0,0]) for i in range(A.cols)]])
    S[0,2] = v_3_temp[0]
    S[1,2] = v_3_temp[1]
    S[2,2] = v_3_temp[2]
    det_S = S.det()
    solutions = sp.solve(sp.Poly((det_S)-1), t)
    t_value = solutions[0]
    substitution_dict = {t : t_value}
    v_3 = sp.Matrix([[((parametric_eq_vertex_line[x]).subs(substitution_dict)).simplify()], [((parametric_eq_vertex_line[y]).subs(substitution_dict)).simplify()],
                     [((parametric_eq_vertex_line[z]).subs(substitution_dict)).simplify()]])
    # create rotation matrix
    S[0,2] = v_3[0]
    S[1,2] = v_3[1]
    S[2,2] = v_3[2]
    S_norm = S
    return S_norm

def parabolic_cylinder_canonize(A_overline, A, b, eq, A_overline_og):
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
    A_overline_trasl = clean_near_zero(A_overline_trasl)
    obtain_quadric_expr(A_overline_trasl, display_string="Dopo traslazione", display_string_two="after translation")
    P_overline_tot = sp.BlockMatrix([[sp.BlockMatrix([S_norm, transl_vector]).as_explicit()], [sp.Matrix([[0, 0, 0, 1]])]]).as_explicit()
    P_overline_tot = np.array(P_overline_tot, dtype=np.float64)
    P_overline_tot = clean_near_zero(P_overline_tot)
    A_overline_CMF = (np.transpose(P_overline_tot) @ A_overline_og) @ P_overline_tot
    A_overline_CMF = clean_near_zero(A_overline_CMF)
    obtain_quadric_expr(A_overline_CMF, display_string="Dopo rotazione", display_string_two="canonical")
    return A_overline_CMF, np.array(S_norm, dtype=np.float64), np.array(transl_vector, dtype=np.float64), A_overline_trasl

