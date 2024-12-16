import numpy as np
import sympy as sp
from scipy import linalg as la
import warnings

from src.numerical.classifier import *
from src.numerical.misc import *
from src.numerical.parabolic_cylinder import *
from src.numerical.checker import *

"""
This module computes the canonical metric form of the quadric and its transformations.
"""

global x, y, z
x, y, z = sp.symbols('x y z')

def matrix_approximate_for_graphics(matrix):
    """
    Rounds each float in the input matrix to two decimal digits, in order to avoid
    having very long floats in the graphical part.

    Parameters:
    matrix: A numpy array

    Returns:
    The rounded matrix (a numpy array)
    """
    vround = np.vectorize(lambda x: round(float(x), 2))
    return vround(matrix) # Apply to the whole matrix at once

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
            float_val = float(coeff) # Convert to float and check if it's effectively an integer
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

def substitute_col(matrix, vector, col_i):
    result = matrix.copy()
    result[:, col_i] = vector.flatten()
    return result

def orthogonalize(A, S, D):
    eigenvals, eigenvects = np.linalg.eig(A) # Get eigenvalues and eigenvectors using numpy
    eigenvals = np.round(eigenvals, decimals=10) # Round eigenvalues to handle numerical precision issues
    unique_vals, counts = np.unique(eigenvals, return_counts=True)
    for val, count in zip(unique_vals, counts):
        if count > 1:
            indices = np.where(np.abs(np.diag(D) - val) < 1e-10)[0] # Find indices of the repeated eigenvalue
            vects = [] # Get the corresponding eigenvectors
            for idx in indices:
                vects.append(S[:, idx])
            # Gram-Schmidt
            Q = np.zeros((len(vects[0]), len(vects)))
            Q[:, 0] = vects[0] / np.linalg.norm(vects[0])
            for i in range(1, len(vects)):
                v = vects[i] # Subtract projections onto previous vectors
                for j in range(i):
                    v = v - np.dot(v, Q[:, j]) * Q[:, j]
                Q[:, i] = v / np.linalg.norm(v) # Normalize
            for i, idx in enumerate(indices): # Replace the columns in S with orthonormalized vectors
                S = substitute_col(S, Q[:, i], idx)
    return S

def orthonormalize(A, S, D):
    S = orthogonalize(A=A.copy(), S=S.copy(), D=D.copy())
    norms = la.norm(S, axis=0)
    norms[norms == 0] = 1  # replace zeros with ones to avoid division by zero
    S_norm = S / norms
    return S_norm

def centered_quadric(quadric_type, A_overline, A, b):
    A_overline_og = A_overline.copy()
    warnings.filterwarnings("error")
    try:
        center_vec = la.solve(A, -b)
    except la.LinAlgWarning:
        center_vec = np.linalg.lstsq(A, b, rcond=None)[0]
    warnings.filterwarnings("ignore")
    obtain_quadric_expr(A_overline, center_vec=center_vec, display_string="Originale")
    A_overline[3, 3] = (((np.transpose(center_vec)) @ A) @ (center_vec))[0,0] + (2 * ((np.transpose(center_vec)) @ b))[0,0] +  A_overline[3, 3]
    b = np.array([[0], [0], [0]])
    A_overline = assign_b_to_overA(A_overline, b)
    obtain_quadric_expr(A_overline, display_string="Dopo la traslazione")
    eigenvals, S = la.eig(A)
    S = np.real(S)
    D = np.real(np.diag(eigenvals))
    D = clean_near_zero(D)
    S_norm = orthonormalize(A=A.copy(), S=S.copy(), D=D.copy())
    A = D
    A_overline = assign_A_to_overA(A_overline, A)
    A_overline_middle = A_overline.copy()
    obtain_quadric_expr(A_overline, display_string="Dopo la rotazione/diagonalizzazione")
    top_block = np.hstack([S_norm, center_vec.reshape(3, 1)])  # 3x4
    bottom_block = np.array([[0, 0, 0, 1]])  # 1x4
    P_overline_tot = np.vstack([top_block, bottom_block]) # 4x4
    P_overline_tot = clean_near_zero(np.array(P_overline_tot, dtype=np.float64))
    check_two_forms_centered(A_overline.copy(), A_overline_og.copy(), P_overline_tot.copy(), display_string="canonical")
    center_vec = -center_vec
    # things to return
    initial_eq = ((sp.Matrix([[x, y, z, 1]]) * sp.Matrix(matrix_approximate_for_graphics(A_overline_og)) * sp.Matrix([[x], [y], [z], [1]]))[0, 0]).expand()
    initial_eq = convert_poly_coeffs(initial_eq)
    middle_eq = ((sp.Matrix([[x, y, z, 1]]) * sp.Matrix(matrix_approximate_for_graphics(A_overline_middle)) * sp.Matrix([[x], [y], [z], [1]]))[0, 0]).expand()
    middle_eq = convert_poly_coeffs(middle_eq)
    final_eq = ((sp.Matrix([[x, y, z, 1]]) * sp.Matrix(matrix_approximate_for_graphics(A_overline)) * sp.Matrix([[x], [y], [z], [1]]))[0, 0]).expand()
    final_eq = convert_poly_coeffs(final_eq)
    returned_dict = {"quadric type": quadric_type,
                     "final quadric matrix": matrix_approximate_for_graphics(A_overline),
                     "translation vector": matrix_approximate_for_graphics(center_vec),
                     "rotation matrix": matrix_approximate_for_graphics(S_norm),
                     "initial quadric matrix": matrix_approximate_for_graphics(A_overline_og),
                     "middle quadric matrix": matrix_approximate_for_graphics(A_overline_middle),
                     "initial quadric equation": initial_eq,
                     "middle quadric equation": middle_eq,
                     "final quadric equation": final_eq}
    return returned_dict

def canonize_paraboloid(quadric_type, A_overline, A, b, v_trasl_1, null_value):
    v_trasl_1 = v_trasl_1.flatten()
    b = b.flatten()
    if (null_value == 0): # x has nulleigenvalue
        v_trasl_2 = np.array([-A_overline[3, 3] / (2 * b[0]), -v_trasl_1[1], -v_trasl_1[2]], dtype=np.float64)
        A_overline[3, 3] = 0
        obtain_quadric_expr(A_overline, display_string="After second translation")
        return A_overline, v_trasl_2
    if (null_value == 1): # y has null eigenvalue
        v_trasl_2 = np.array([-v_trasl_1[0], -A_overline[3, 3] / (2 * b[1]), -v_trasl_1[2]], dtype=np.float64)
        A_overline[3, 3] = 0
        return A_overline, v_trasl_2
    if (null_value == 2): # z has null eigenvalue
        v_trasl_2 = np.array([-v_trasl_1[0], -v_trasl_1[1], -A_overline[3, 3] / (2 * b[2])], dtype=np.float64)
        A_overline[3, 3] = 0
        obtain_quadric_expr(A_overline, display_string="After second translation")
        return A_overline, v_trasl_2

def acentered_quadric_rk2(quadric_type, A_overline, A, b):
    b = b.flatten()
    if (np.isclose(A[0, 0],0)):  # x has null eigenvalue
        null_value = 0
        v_trasl_1 = np.array([0, b[1]/A[1,1], b[2]/A[2,2]], dtype=np.float64) # (vettore di cui traslo)
        A_overline[3, 3] = A_overline[3, 3] - (((b[1]) ** 2) / A[1, 1]) - (((b[2]) ** 2) / A[2, 2])
        b = np.array([b[0], 0, 0])
        A_overline = assign_b_to_overA(A_overline, b)
        obtain_quadric_expr(A_overline, display_string="After first translation")
        if (not np.isclose(b[0], 0)):
            A_overline, v_trasl_tot = canonize_paraboloid(quadric_type, A_overline, A, b, v_trasl_1, null_value)
            return A_overline, v_trasl_tot
        else:
            v_trasl_1 = np.array([-v_trasl_1[0], -v_trasl_1[1], -v_trasl_1[2]], dtype=np.float64)
            return A_overline, v_trasl_1
    elif (np.isclose(A[1, 1],0)):  # y has null eigenvalue
        null_value = 1
        v_trasl_1 = np.array([b[0]/A[0,0], 0, b[2]/A[2,2]], dtype=np.float64) # (vettore di cui traslo)
        v_trasl_1 = v_trasl_1.flatten()
        A_overline[3, 3] = A_overline[3, 3] - ((b[0]) ** 2 / A[0, 0]) - ((b[2]) ** 2 / A[2, 2])
        b = np.array([[0], [b[1]], [0]])
        A_overline = assign_b_to_overA(A_overline, b)
        obtain_quadric_expr(A_overline, display_string="After first translation")
        if (not np.isclose(b[1], 0)):
            A_overline, v_trasl_tot = canonize_paraboloid(quadric_type, A_overline, A, b, v_trasl_1, null_value)
            return A_overline, v_trasl_tot
        else:
            v_trasl_1 = np.array([-v_trasl_1[0], -v_trasl_1[1], -v_trasl_1[2]], dtype=np.float64)
            return A_overline, v_trasl_1
    elif (np.isclose(A[2, 2],0)):  # z has null eigenvalue
        null_value = 2
        v_trasl_1 = np.array([b[0]/A[0,0], b[1]/A[1,1], 0], dtype=np.float64) # (vettore di cui traslo)
        A_overline[3, 3] = A_overline[3, 3] - ((b[0]) ** 2 / A[0, 0]) - ((b[1]) ** 2 / A[1, 1])
        b = np.array([0, 0, b[2]], dtype=np.float64)
        A_overline = assign_b_to_overA(A_overline, b)
        obtain_quadric_expr(A_overline, display_string="After first translation")
        if (not np.isclose(b[2], 0)):
            A_overline, v_trasl_tot = canonize_paraboloid(quadric_type, A_overline, A, b, v_trasl_1, null_value)
            return A_overline, v_trasl_tot
        else:
            v_trasl_1 = np.array([-v_trasl_1[0], -v_trasl_1[1], -v_trasl_1[2]], dtype=np.float64)
            return A_overline, v_trasl_1

def acentered_quadric_rk1(quadric_type, A_overline, A, b):
    b = b.flatten()
    if (np.isclose(A[0, 0],0) and np.isclose(A[1, 1],0)):  # x and y have null eigenvalue
        transl_vector1 = np.array([0, 0, -b[2]/A_overline[2,2]], dtype=np.float64)
        A_overline[3, 3] = A_overline[3, 3] - ((b[2]**2)/A_overline[2,2])
        b[2] = 0
        A_overline = assign_b_to_overA(A_overline.copy(), b)
        obtain_quadric_expr(A_overline, display_string="After first translation")
        if (np.isclose(b[0], 0) and np.isclose(b[1], 0)):
            if (np.isclose(A_overline[3, 3],0)): # A_overline rango 1, piano doppio
                return A_overline, transl_vector1
            else: # A_overline rango 2, piani paralleli
                return A_overline, transl_vector1
    elif (np.isclose(A[0, 0],0) and np.isclose(A[2, 2],0)): # x and z have null eigenvalue
        transl_vector1 = np.array([0, -b[1] / A[1, 1], 0], dtype=np.float64)
        A_overline[3, 3] = A_overline[3, 3] - ((b[1]**2) / A_overline[1, 1])
        b[1] = 0
        A_overline = assign_b_to_overA(A_overline.copy(), b)
        obtain_quadric_expr(A_overline, display_string="After first translation")
        if (np.isclose(b[0], 0) and np.isclose(b[2], 0)):
            if (np.isclose(A_overline[3, 3],0)):
                return A_overline, transl_vector1
            else:
                return A_overline, transl_vector1
    elif (np.isclose(A[1, 1],0) and np.isclose(A[2, 2],0)):  # y e z have null eigenvalue
        transl_vector1 = np.array([-b[0] / A[0, 0], 0, 0], dtype=np.float64)
        A_overline[3, 3] = A_overline[3, 3] - ((b[0]**2)/ A_overline[0, 0])
        b[0] = 0
        A_overline = assign_b_to_overA(A_overline.copy(), b)
        obtain_quadric_expr(A_overline, display_string="After first translation")
        if (np.isclose(b[1], 0) and np.isclose(b[2], 0)):
            if (np.isclose(A_overline[3, 3],0)):
                return A_overline, transl_vector1
            else:
                return A_overline, transl_vector1

def acentered_quadric(quadric_type, A_overline, A, b, eq):
    A_overline_og = A_overline.copy()
    obtain_quadric_expr(A_overline_og, display_string="OG")
    if (quadric_type == 14): # parabolic cylinder, treated separately
        A_overline, S_norm, transl_vector, A_overline_middle = parabolic_cylinder_canonize(A_overline.copy(), A.copy(), b, eq, A_overline_og.copy())
    else: # other quadrics
        eigenvals, S = la.eig(A)
        D = clean_near_zero(np.real(np.diag(eigenvals)))
        S_norm = orthonormalize(A.copy(), S.copy(), D.copy())
        A = D.copy()
        A_overline = assign_A_to_overA(A_overline, A)
        b = (np.transpose(S_norm) @ b)
        A_overline = assign_b_to_overA(A_overline.copy(), b)
        A_overline_middle = A_overline.copy()
        top_block_temp = np.hstack([S_norm, np.array([0,0,0]).reshape(3,1)])  # 3x4
        bottom_block_temp = np.array([[0, 0, 0, 1]])  # 1x4
        P_overline_tot_temp = np.vstack([top_block_temp, bottom_block_temp])  # 4x4
        check_two_forms_centered(A_overline.copy(), A_overline_og.copy(), P_overline_tot_temp.copy(), display_string="post_rotation")
        obtain_quadric_expr(A_overline, display_string="After first diagonalization/rotation")
        if (np.linalg.matrix_rank(A) == 2):
            A_overline, transl_vector = acentered_quadric_rk2(quadric_type, A_overline.copy(), A.copy(), b)
        elif (np.linalg.matrix_rank(A) == 1):
            A_overline, transl_vector = acentered_quadric_rk1(quadric_type, A_overline.copy(), A.copy(), b)
        check_two_forms_acentered(A_overline.copy(), A_overline_og.copy(), S_norm.copy(), transl_vector, display_string="canonical")
    # things to return
    A_overline_middle = clean_near_zero(A_overline_middle)
    A_overline = clean_near_zero(A_overline)
    initial_eq = ((sp.Matrix([[x, y, z, 1]]) * sp.Matrix(matrix_approximate_for_graphics(A_overline_og)) * sp.Matrix([[x], [y], [z], [1]]))[0, 0]).expand()
    initial_eq = convert_poly_coeffs(initial_eq)
    middle_eq = ((sp.Matrix([[x, y, z, 1]]) * sp.Matrix(matrix_approximate_for_graphics(A_overline_middle)) * sp.Matrix([[x], [y], [z], [1]]))[0, 0]).expand()
    middle_eq = convert_poly_coeffs(middle_eq)
    final_eq = ((sp.Matrix([[x, y, z, 1]]) * sp.Matrix(matrix_approximate_for_graphics(A_overline)) * sp.Matrix([[x], [y], [z], [1]]))[0, 0]).expand()
    final_eq = convert_poly_coeffs(final_eq)
    returned_dict = {"quadric type": quadric_type,
                     "final quadric matrix": matrix_approximate_for_graphics(A_overline),
                     "translation vector": matrix_approximate_for_graphics(transl_vector),
                     "rotation matrix": matrix_approximate_for_graphics(S_norm),
                     "initial quadric matrix": matrix_approximate_for_graphics(A_overline_og),
                     "middle quadric matrix": matrix_approximate_for_graphics(A_overline_middle),
                     "initial quadric equation": initial_eq,
                     "middle quadric equation": middle_eq,
                     "final quadric equation": final_eq}
    return returned_dict

def canonize_quadric(eq):
    """
    Transforms a quadric equation into its canonical form through translations and rotations.

    Parameters
    ----------
    eq : str
        The quadric equation to canonize, expressed as a string in terms of x, y, z variables.
        Example: "x**2 + 2*x*y + y**2 + z**2 = 0"

    Returns
    -------
    dict
        A dictionary containing:
        - quadric type (int): int representing the classification of the quadric
        - final/initial/middle quadric matrices (numpy.ndarray): matrices representing the quadric at different stages
        - translation vector (numpy.ndarray): translation applied to center the quadric
        - rotation matrix (numpy.ndarray): rotation to align with coordinate axes
        - initial/middle/final quadric equations (sp.expr): sympy expression representations at each stage
        - centered quadric (bool): whether the quadric has a center

    Notes
    -----
    The function first converts the equation to matrix form using expr2matrices().
    It then classifies the quadric type and determines if it has a center by checking if det(A) ≠ 0.
    For centered quadrics, it calls centered_quadric() to find the canonical form.
    For non-centered quadrics, it calls acentered_quadric().

    Prints intermediate steps showing the initial matrix A and equation.

    """
    A_overline, A, b = expr2matrices(eq)
    quadric_type = classify_quadric(A.copy(), A_overline.copy())
    if (not np.isclose(la.det(A), 0)): # la quadrica e' a centro
        returned_dict = centered_quadric(quadric_type, A_overline, A, b)
        returned_dict["centered quadric"] = True
        return returned_dict
    else: # la quadrica non e' a centro
        returned_dict = acentered_quadric(quadric_type, A_overline, A, b, eq)
        returned_dict["centered quadric"] = False
        return returned_dict
