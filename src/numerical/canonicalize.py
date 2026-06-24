"""
Compute canonical metric forms and typed transformation artifacts.

Run the transformation tests with ``python -m pytest tests/test_transformer.py -q``.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import numpy.typing as npt
import sympy as sp
from scipy import linalg as la

from src.numerical.numerical_helpers import (
    assign_linear_block,
    assign_quadratic_block,
    clean_near_zero,
    expression_from_matrix,
    normalize_integer_coefficients,
    round_for_display,
)
from src.numerical.classifier import QuadricClassifier
from src.numerical.models import CanonicalizationResult, FloatArray, QuadricType
from src.numerical.parabolic_cylinder import parabolic_cylinder_canonize
from src.numerical.parser import QuadricParser


NUMERICAL_TOLERANCE = 1e-10
DISPLAY_DECIMALS = 2


@dataclass(frozen=True, slots=True)
class TransformationData:
    """Store the unvalidated internal artifacts of one transformation."""

    initial_matrix: FloatArray
    middle_matrix: FloatArray
    final_matrix: FloatArray
    translation_vector: FloatArray
    rotation_matrix: FloatArray


def matrix_approximate_for_graphics(matrix: npt.ArrayLike) -> FloatArray:
    """
    Rounds each float in the input matrix to two decimal digits, in order to avoid
    having very long floats in the graphical part.

    Parameters:
    matrix: A numpy array

    Returns:
    The rounded matrix (a numpy array)
    """
    return round_for_display(matrix, DISPLAY_DECIMALS)

def convert_poly_coeffs(expr: sp.Expr) -> sp.Expr:
    """
    Convert float coefficients ending in .00 to integers in a sympy polynomial expression.

    Parameters:
    expr: A sympy expression (polynomial in x, y, z)

    Returns:
    A sympy expression with converted coefficients
    """
    return normalize_integer_coefficients(expr, NUMERICAL_TOLERANCE)

def substitute_col(matrix: FloatArray, vector: npt.ArrayLike, col_i: int) -> FloatArray:
    result = matrix.copy()
    result[:, col_i] = np.asarray(vector).flatten()
    return result

def orthogonalize(A: FloatArray, S: FloatArray, D: FloatArray) -> FloatArray:
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

def orthonormalize(A: FloatArray, S: FloatArray, D: FloatArray) -> FloatArray:
    S = orthogonalize(A=A.copy(), S=S.copy(), D=D.copy())
    norms = la.norm(S, axis=0)
    norms[norms == 0] = 1  # replace zeros with ones to avoid division by zero
    S_norm = S / norms
    return S_norm

def centered_quadric(quadric_type: QuadricType, A_overline: FloatArray, A: FloatArray, b: FloatArray) \
        -> TransformationData:
    A_overline_og = A_overline.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=la.LinAlgWarning)
        try:
            center_vec = la.solve(A, -b)
        except la.LinAlgWarning:
            center_vec = np.linalg.lstsq(A, -b, rcond=None)[0]
    A_overline[3, 3] = (((np.transpose(center_vec)) @ A) @ (center_vec))[0,0] + \
                       (2 * ((np.transpose(center_vec)) @ b))[0,0] +  A_overline[3, 3]
    b = np.array([[0], [0], [0]])
    A_overline = assign_linear_block(A_overline, b)
    eigenvals, S = la.eig(A)
    S = np.real(S)
    D = np.real(np.diag(eigenvals))
    D = clean_near_zero(D, NUMERICAL_TOLERANCE)
    S_norm = orthonormalize(A=A.copy(), S=S.copy(), D=D.copy())
    A = D
    A_overline = assign_quadratic_block(A_overline, A)
    A_overline_middle = A_overline.copy()
    top_block = np.hstack([S_norm, center_vec.reshape(3, 1)])  # 3x4
    bottom_block = np.array([[0, 0, 0, 1]])  # 1x4
    P_overline_tot = np.vstack([top_block, bottom_block]) # 4x4
    P_overline_tot = clean_near_zero(np.array(P_overline_tot, dtype=np.float64), NUMERICAL_TOLERANCE)
    center_vec = -center_vec
    return TransformationData(
        initial_matrix=A_overline_og,
        middle_matrix=A_overline_middle,
        final_matrix=A_overline,
        translation_vector=np.asarray(center_vec, dtype=np.float64),
        rotation_matrix=np.asarray(S_norm, dtype=np.float64),
    )

def canonize_paraboloid(quadric_type: QuadricType, A_overline: FloatArray, A: FloatArray, b: FloatArray,
                        v_trasl_1: FloatArray, null_value: int) -> tuple[FloatArray, FloatArray]:
    v_trasl_1 = v_trasl_1.flatten()
    b = b.flatten()
    if (null_value == 0): # x has nulleigenvalue
        v_trasl_2 = np.array([-A_overline[3, 3] / (2 * b[0]), -v_trasl_1[1], -v_trasl_1[2]], dtype=np.float64)
        A_overline[3, 3] = 0
        return A_overline, v_trasl_2
    if (null_value == 1): # y has null eigenvalue
        v_trasl_2 = np.array([-v_trasl_1[0], -A_overline[3, 3] / (2 * b[1]), -v_trasl_1[2]], dtype=np.float64)
        A_overline[3, 3] = 0
        return A_overline, v_trasl_2
    if (null_value == 2): # z has null eigenvalue
        v_trasl_2 = np.array([-v_trasl_1[0], -v_trasl_1[1], -A_overline[3, 3] / (2 * b[2])], dtype=np.float64)
        A_overline[3, 3] = 0
        return A_overline, v_trasl_2
    raise ValueError(f"null eigenvalue index must be 0, 1, or 2; received {null_value}")

def acentered_quadric_rk2(quadric_type: QuadricType, A_overline: FloatArray, A: FloatArray, b: FloatArray) \
        -> tuple[FloatArray, FloatArray]:
    b = b.flatten()
    if np.isclose(A[0, 0], 0):  # x has null eigenvalue
        null_value = 0
        v_trasl_1 = np.array([0, b[1]/A[1,1], b[2]/A[2,2]], dtype=np.float64) # (vettore di cui traslo)
        A_overline[3, 3] = A_overline[3, 3] - (((b[1]) ** 2) / A[1, 1]) - (((b[2]) ** 2) / A[2, 2])
        b = np.array([b[0], 0, 0])
        A_overline = assign_linear_block(A_overline, b)
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
        A_overline = assign_linear_block(A_overline, b)
        if not np.isclose(b[1], 0):
            A_overline, v_trasl_tot = canonize_paraboloid(quadric_type, A_overline, A, b, v_trasl_1, null_value)
            return A_overline, v_trasl_tot
        else:
            v_trasl_1 = np.array([-v_trasl_1[0], -v_trasl_1[1], -v_trasl_1[2]], dtype=np.float64)
            return A_overline, v_trasl_1
    elif np.isclose(A[2, 2], 0):  # z has null eigenvalue
        null_value = 2
        v_trasl_1 = np.array([b[0]/A[0,0], b[1]/A[1,1], 0], dtype=np.float64) # (vettore di cui traslo)
        A_overline[3, 3] = A_overline[3, 3] - ((b[0]) ** 2 / A[0, 0]) - ((b[1]) ** 2 / A[1, 1])
        b = np.array([0, 0, b[2]], dtype=np.float64)
        A_overline = assign_linear_block(A_overline, b)
        if (not np.isclose(b[2], 0)):
            A_overline, v_trasl_tot = canonize_paraboloid(quadric_type, A_overline, A, b, v_trasl_1, null_value)
            return A_overline, v_trasl_tot
        else:
            v_trasl_1 = np.array([-v_trasl_1[0], -v_trasl_1[1], -v_trasl_1[2]], dtype=np.float64)
            return A_overline, v_trasl_1
    raise ValueError("rank-two canonical matrix has no numerical zero eigenvalue")

def acentered_quadric_rk1(quadric_type: QuadricType, A_overline: FloatArray, A: FloatArray, b: FloatArray) \
        -> tuple[FloatArray, FloatArray]:
    b = b.flatten()
    if np.isclose(A[0, 0], 0) and np.isclose(A[1, 1], 0):  # x and y have null eigenvalue
        transl_vector1 = np.array([0, 0, -b[2]/A_overline[2,2]], dtype=np.float64)
        A_overline[3, 3] = A_overline[3, 3] - ((b[2]**2)/A_overline[2,2])
        b[2] = 0
        A_overline = assign_linear_block(A_overline, b)
        if (np.isclose(b[0], 0) and np.isclose(b[1], 0)):
            if (np.isclose(A_overline[3, 3],0)): # A_overline rango 1, piano doppio
                return A_overline, transl_vector1
            else: # A_overline rango 2, piani paralleli
                return A_overline, transl_vector1
    elif np.isclose(A[0, 0], 0) and np.isclose(A[2, 2], 0): # x and z have null eigenvalue
        transl_vector1 = np.array([0, -b[1] / A[1, 1], 0], dtype=np.float64)
        A_overline[3, 3] = A_overline[3, 3] - ((b[1]**2) / A_overline[1, 1])
        b[1] = 0
        A_overline = assign_linear_block(A_overline, b)
        if (np.isclose(b[0], 0) and np.isclose(b[2], 0)):
            if (np.isclose(A_overline[3, 3],0)):
                return A_overline, transl_vector1
            else:
                return A_overline, transl_vector1
    elif np.isclose(A[1, 1], 0) and np.isclose(A[2, 2], 0):  # y e z have null eigenvalue
        transl_vector1 = np.array([-b[0] / A[0, 0], 0, 0], dtype=np.float64)
        A_overline[3, 3] = A_overline[3, 3] - ((b[0]**2)/ A_overline[0, 0])
        b[0] = 0
        A_overline = assign_linear_block(A_overline, b)
        if (np.isclose(b[1], 0) and np.isclose(b[2], 0)):
            if (np.isclose(A_overline[3, 3],0)):
                return A_overline, transl_vector1
            else:
                return A_overline, transl_vector1
    raise ValueError("rank-one canonical matrix does not contain exactly two zero eigenvalues")

def acentered_quadric(quadric_type: QuadricType, A_overline: FloatArray, A: FloatArray, b: FloatArray, eq: str) \
        -> TransformationData:
    A_overline_og = A_overline.copy()
    if quadric_type is QuadricType.PARABOLIC_CYLINDER:
        A_overline, S_norm, transl_vector, A_overline_middle = parabolic_cylinder_canonize(A_overline.copy(), A.copy(),
                                                                                           b, eq, A_overline_og.copy())
    else: # other quadrics
        eigenvals, S = la.eig(A)
        D = clean_near_zero(np.real(np.diag(eigenvals)), NUMERICAL_TOLERANCE)
        S_norm = orthonormalize(A.copy(), S.copy(), D.copy())
        A = D.copy()
        A_overline = assign_quadratic_block(A_overline, A)
        b = (np.transpose(S_norm) @ b)
        A_overline = assign_linear_block(A_overline, b)
        A_overline_middle = A_overline.copy()
        top_block_temp = np.hstack([S_norm, np.array([0,0,0]).reshape(3,1)])  # 3x4
        bottom_block_temp = np.array([[0, 0, 0, 1]])  # 1x4
        P_overline_tot_temp = np.vstack([top_block_temp, bottom_block_temp])  # 4x4
        if np.linalg.matrix_rank(A) == 2:
            A_overline, transl_vector = acentered_quadric_rk2(quadric_type, A_overline.copy(), A.copy(), b)
        elif np.linalg.matrix_rank(A) == 1:
            A_overline, transl_vector = acentered_quadric_rk1(quadric_type, A_overline.copy(), A.copy(), b)
    A_overline_middle = clean_near_zero(A_overline_middle, NUMERICAL_TOLERANCE)
    A_overline = clean_near_zero(A_overline, NUMERICAL_TOLERANCE)
    return TransformationData(
        initial_matrix=A_overline_og,
        middle_matrix=A_overline_middle,
        final_matrix=A_overline,
        translation_vector=np.asarray(transl_vector, dtype=np.float64),
        rotation_matrix=np.asarray(S_norm, dtype=np.float64),
    )

class QuadricCanonicalizer:
    """Orchestrate parsing, classification, and canonical transformation strategies."""

    parser: QuadricParser
    classifier: QuadricClassifier

    def __init__(self, parser: QuadricParser, classifier: QuadricClassifier) -> None:
        self.parser = parser
        self.classifier = classifier

    def canonize(self, eq: str) -> CanonicalizationResult:
        """
        Transform one equation into a validated canonicalization result.

        Args:
            eq: str
                Degree-two equation in x, y, and z.
            return: CanonicalizationResult
                Typed matrices, equations, and transformations for the quadric.
        """

        matrices = self.parser.parse_matrices(eq)
        quadric_type = self.classifier.classify(matrices.quadratic, matrices.homogeneous)
        centered = not np.isclose(np.linalg.det(matrices.quadratic), 0, atol=NUMERICAL_TOLERANCE, rtol=0)
        if centered:
            data = centered_quadric(
                quadric_type, matrices.homogeneous.copy(), matrices.quadratic.copy(), matrices.linear.copy()
            )
        else:
            data = acentered_quadric(
                quadric_type, matrices.homogeneous.copy(), matrices.quadratic.copy(), matrices.linear.copy(), eq
            )
        return _build_result(quadric_type, centered, data)


def _build_result(quadric_type: QuadricType, centered: bool, data: TransformationData) -> CanonicalizationResult:
    initial_matrix = matrix_approximate_for_graphics(data.initial_matrix)
    middle_matrix = matrix_approximate_for_graphics(data.middle_matrix)
    final_matrix = matrix_approximate_for_graphics(data.final_matrix)
    return CanonicalizationResult(
        quadric_type=quadric_type,
        centered=centered,
        initial_matrix=initial_matrix,
        middle_matrix=middle_matrix,
        final_matrix=final_matrix,
        translation_vector=matrix_approximate_for_graphics(data.translation_vector),
        rotation_matrix=matrix_approximate_for_graphics(data.rotation_matrix),
        initial_equation=convert_poly_coeffs(expression_from_matrix(initial_matrix)),
        middle_equation=convert_poly_coeffs(expression_from_matrix(middle_matrix)),
        final_equation=convert_poly_coeffs(expression_from_matrix(final_matrix)),
    )


def canonize_quadric(eq: str) -> CanonicalizationResult:
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
    canonicalizer = QuadricCanonicalizer(parser=QuadricParser(),
                                         classifier=QuadricClassifier(tolerance=NUMERICAL_TOLERANCE))
    return canonicalizer.canonize(eq)


__all__ = ["QuadricCanonicalizer", "canonize_quadric"]
