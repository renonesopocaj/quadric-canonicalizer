import sympy as sp
from src.numerical.misc import expr2matrices
from scipy import linalg as la
import numpy as np

global x, y, z, ENUM_QUADRICS
x, y, z = sp.symbols('x y z')

"""
This module classifies the quadric surface.
We have currently not yet implemented a distinction between real elliptic cylinder (9) and complex elliptic
cylinder (10), so the function "classify" always returns 9, because real elliptic cylinder are supported
by the graphical part. The same for real parallel planes (15) and complex parallel planes (16)
"""

ENUM_QUADRICS = {"real ellipsoid": 1, "two sheets hyperboloid": 4, "complex ellipsoid": 2, "one sheet hyperboloid": 3,
                 "complex cone": 6, "real cone": 5, "elliptic paraboloid": 7, "hyperbolic paraboloid": 8,
                 "real elliptic cylinder": 9, "complex elliptic cylinder": 10, "hyperbolic cylinder": 11,
                 "real intersecting planes": 12, "complex intersecting planes": 13, "parabolic cylinder": 14,
                 "real parallel planes": 15, "complex parallel planes": 16, "double plane": 17}

class NotAQuadricException(Exception):
    pass

def get_eigenvalues_multiplicities(A, tol=1e-10):
    """
    Computes eigenvalues of a matrix and returns their multiplicities as a dictionary.
    The function handles numerical precision by setting near-zero eigenvalues to exactly zero.

    Parameters:
    -----------
    A : numpy.ndarray
        Square matrix for which to compute eigenvalues.
        Must be a 2-dimensional array of shape (n, n).
    tol : float, optional (default=1e-10)
        Tolerance threshold for considering eigenvalues as zero.
        Eigenvalues with absolute value less than tol will be set to 0.

    Returns:
    --------
    dict
        Dictionary mapping distinct eigenvalues to their algebraic multiplicities.
        Keys are the unique eigenvalues (as floats).
        Values are the number of times each eigenvalue appears.

    Notes:
    ------
    - Uses numpy.linalg.eigvals for eigenvalue computation
    - Only returns real parts of eigenvalues
    - Handles numerical precision issues by setting very small eigenvalues to zero
    - Useful for analyzing matrix properties and quadratic forms

    Examples:
    --------
    >>> import numpy as np
    >>> # Identity matrix - eigenvalue 1 with multiplicity 2
    >>> A = np.array([[1, 0],
    ...               [0, 1]])
    >>> get_eigenvalues_multiplicities(A)
    {1.0: 2}

    >>> # Matrix with zero eigenvalue
    >>> B = np.array([[1, 1],
    ...               [-1, -1]])
    >>> get_eigenvalues_multiplicities(B)
    {0.0: 1, 0.0: 1}
    """
    eigenvals = la.eigvals(A)
    eigenvals[abs(eigenvals) < tol] = 0
    eigenvals = eigenvals.real
    unique, counts = np.unique(eigenvals, return_counts=True)
    return dict(zip(unique, counts))

def classify_quadric(A, A_overline):
    """
    Analyzes and classifies a quadric surface from the matrices.

    Parameters:
    -----------
    A : numpy.ndarray
        The square matrix associated with the quadratic form associated to the quadric surface.
    A_overline : numpy.ndarray
        The square matrix associated to the quadric surface.

    Returns:
    --------
    quadric_type : int
        The function returns a type that represents the quadric surface.

    Notes:
    ------
    - Might return an exception in case the input matrix doesn't represent a quadric surface.
    """
    detAover = la.det(A_overline)
    detAover = 0 if np.isclose(detAover, 0) else detAover
    rkA = np.linalg.matrix_rank(A)
    rkAover = np.linalg.matrix_rank(A_overline)
    eigenvals_numerical = get_eigenvalues_multiplicities(A)
    p = sum(occorrenze for num, occorrenze in eigenvals_numerical.items() if num > 0)
    n = sum(occorrenze for num, occorrenze in eigenvals_numerical.items() if num < 0)
    z = 3 - p - n
    A_def_pos = (p == 3)
    A_def_neg = (n == 3)
    A_semidef_pos = (z > 0 and n == 0 and p > 0)
    A_semidef_neg = (z < 0 and p == 0 and n > 0);
    A_indef = (p > 0 and n > 0)
    A_def = (A_def_neg or A_def_pos)
    A_semidef = (A_semidef_pos or A_semidef_neg)
    if (rkA == 3):
        if (detAover < 0):
            if (A_def ==  True):
                quadric_type = 1
            elif (A_indef == True):
                quadric_type = 4
        elif (detAover > 0):
            if (A_def == True):
                quadric_type = 2
            elif (A_indef == True):
                quadric_type = 3
        else:
            if (rkAover == 3):
                if (A_def == True):
                    quadric_type = 6
                elif (A_indef == True):
                    quadric_type = 5
            else:
                raise NotAQuadricException("The input A_ovelrine matrix is not a quadric" +
                                     f"A_overline is {A_overline} p is {p}, n is {n}, z is {z}, detAover is {detAover}, rkAover is {rkAover}, rkA is {rkA}")
    elif (rkA == 2):
        if (detAover < 0):
            if (A_semidef == True):
                quadric_type = 7
        elif (detAover > 0):
            if (A_indef == True):
                quadric_type = 8
        else:
            if (rkAover == 3):
                if (A_semidef == True):
                    # TODO: currently unable to tell if the quadric is real or complex
                    quadric_type = 9
                    # quadric_type = 10 # complex
                elif (A_indef == True):
                    quadric_type = 11
            elif (rkAover == 2):
                if (A_indef == True):
                    quadric_type = 12
                elif (A_semidef == True):
                    quadric_type = 13
                else:
                    raise NotAQuadricException("The input A_ovelrine matrix is not a quadric" +
                                         f"A_overline is {A_overline} p is {p}, n is {n}, z is {z}, detAover is {detAover}, rkAover is {rkAover}, rkA is {rkA}")
    elif (rkA == 1):
        if (rkAover == 3):
            quadric_type = 14
        elif (rkAover == 2):
            quadric_type = 15
            # TODO: currently unable to tell if the quadric is real or complex
            # quadric_type = 16 # complex
        elif (rkAover == 1):
            quadric_type = 17
        else:
            raise NotAQuadricException("The input A_ovelrine matrix is not a quadric" +
                                       f"A_overline is {A_overline} p is {p}, n is {n}, z is {z}, detAover is {detAover}, rkAover is {rkAover}, rkA is {rkA}")
    else:
        raise NotAQuadricException("The input A_ovelrine matrix is not a quadric" +
                                       f"A_overline is {A_overline} p is {p}, n is {n}, z is {z}, detAover is {detAover}, rkAover is {rkAover}, rkA is {rkA}")
    try:
        return quadric_type
    except UnboundLocalError:
        print(f"A_overline is {A_overline} p is {p}, n is {n}, z is {z}, detAover is {detAover}, rkAover is {rkAover}, rkA is {rkA}")
        raise

def expr2classification(eq):
    """
    Analyzes and classifies a quadric surface from its symbolic expression.
    This function converts a symbolic equation to matrix form and performs
    classification of the quadric surface.

    Parameters:
    -----------
    eq : sympy.Expression
        Symbolic expression representing a quadric surface equation.
        Should be a polynomial equation in variables x, y, and z.
        Example: x**2 + y**2 - z**2 = 1 (hyperboloid of one sheet)

    Returns:
    --------
    quadric_type : int
        The function returns a type that represents the quadric surface.

    Notes:
    ------
    The function performs these steps:
    1. Converts the symbolic expression to matrix form using expr2matrices
    2. Performs classification using classify_quadric

    See Also:
    --------
    expr2matrices : Converts expression to matrices
    classify_quadric : Performs detailed classification
    """
    A_overline, A, b = expr2matrices(eq)
    quadric_type = classify_quadric(A, A_overline)
    return quadric_type
