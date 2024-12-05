import sympy as sp
import numpy as np

"""
This module contains various useful functions used in other modules.
"""

global x, y, z
x, y, z = sp.symbols('x y z')

def clean_near_zero(matrix, threshold=1e-10):
    """
    Cleans a numerical matrix by replacing values close to zero with exact zeros.
    This function is useful for handling floating-point precision issues in numerical computations.

    Parameters:
    -----------
    matrix : numpy.ndarray
        The input matrix to be cleaned. Can be of any shape.
    threshold : float, optional (default=1e-10)
        The absolute threshold below which values will be set to zero.
        Values v where abs(v) < threshold will be set to 0.

    Returns:
    --------
    numpy.ndarray
        A new matrix of the same shape as the input, with near-zero values replaced by zeros.
        The original matrix is not modified.

    Notes:
    ------
    - The function creates a copy of the input matrix to preserve the original data
    - Useful for post-processing numerical computations where floating-point arithmetic
      may result in very small, non-zero values that should theoretically be zero
    - Common use cases include cleaning matrices after matrix multiplication,
      eigenvalue computation, or coordinate transformations

    Examples:
    --------
    >>> import numpy as np
    >>> # Create a matrix with some near-zero values
    >>> matrix = np.array([[1.0, 1e-15, 0.1],
    ...                    [1e-11, 2.0, 1e-13]])
    >>> cleaned = clean_near_zero(matrix)
    >>> # The very small values are now exactly zero
    >>> print(cleaned)
    [[1.0, 0.0, 0.1],
     [0.0, 2.0, 0.0]]
    """
    if hasattr(matrix, 'is_Matrix'):
        cleaned = matrix.copy()
        for i in range(cleaned.rows):
            for j in range(cleaned.cols):
                if cleaned[i,j].is_number and abs(float(cleaned[i,j])) < threshold:
                    cleaned[i,j] = 0
    else:
        cleaned = matrix.copy() # Make a copy to avoid modifying the original
        cleaned[np.abs(cleaned) < threshold] = 0 # Replace small values with zero
    return cleaned

def assign_A_to_overA(A_overline, A):
    """
    This function performs an element-wise copy from input matrix A to matrix A_overline.
    It verifies that matrix A has the correct dimensions (3x3) before proceeding with
    the assignment operation.

    Parameters
    ----------
    A_overline : numpy.ndarray
        The destination matrix where values will be copied to.
        Expected to be a 4x4 matrix.

    A : numpy.ndarray
        The source matrix whose values will be copied.
        Must be a 3x3 matrix, otherwise a ValueError is raised.

    Returns
    -------
    numpy.ndarray
        The modified A_overline matrix containing the copied values from A.

    Raises
    ------
    ValueError
        If matrix A is not a 3x3 matrix.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> A_overline = np.zeros((3, 3))
    >>> A_overline = assign_A_to_overA(A_overline, A)
    """
    if (A.shape == (3, 3)):
        for i in range(0, 3):
            for j in range(0, 3):
                A_overline[i, j] = A[i, j]
    else:
        raise ValueError("A is not a 3x3 square matrix")
    if (A_overline.shape == (3, 3)):
        raise ValueError("A_overline is not a 4x4 square matrix")
    return A_overline

def substitute_col(matrix, vector, col_i):
    """
    Replaces a specified column in a 3x3 matrix with a given vector.

    This function substitutes the col_i-th column of the input matrix
    with the provided vector. The matrix must be 3x3 and the vector
    must have 3 elements.

    Parameters
    ----------
    matrix : numpy.ndarray
        The input matrix where the column will be replaced.
        Must be a 3x3 matrix.

    vector : numpy.ndarray
        The vector containing the new values for the specified column.
        Must be a 1D array with 3 elements.

    col_i : int
        The index of the column to be replaced (0, 1, or 2).

    Returns
    -------
    numpy.ndarray
        The modified matrix with the replaced column.

    Raises
    ------
    IndexError
        If the specified column does not exist in the input matrix.

    Examples
    --------
    >>> import numpy as np
    >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> new_column = np.array([10, 11, 12])
    >>> result = substitute_col(matrix, new_column, 1)
    >>> print(result)
    [[ 1 10  3]
     [ 4 11  6]
     [ 7 12  9]]

    Notes
    -----
    - The column index col_i should be in the range [0, 2]
    - The vector must have exactly 3 elements to match the matrix dimensions
    """
    try:
        for i in range(0, 3):
            matrix[i, col_i] = vector[i]
    except IndexError:
        raise ValueError(f"matrix is not a mx3 square matrix: matrix has shape {matrix.shape}, "
                         f"vector has shape {vector.shape}, col_i is {col_i}")
    return matrix

def assign_b_to_overA(A_overline, b):
    """
    Updates a homogeneous matrix A_overline by assigning the elements of vector b
    to its last row and column (excluding the corner element).
    This function is typically used in quadric surface transformations to
    incorporate linear terms into the homogeneous form.

    Parameters:
    -----------
    A_overline : numpy.ndarray
        A 4x4 homogeneous matrix to be updated.
        The function modifies this matrix in-place.
    b : numpy.ndarray
        A 3-element vector containing the linear coefficients.
        These values will be assigned to the last row and column of A_overline.

    Returns:
    --------
    numpy.ndarray
        The modified A_overline matrix, where:
        - A_overline[3,0:3] = b
        - A_overline[0:3,3] = b
        The (3,3) element remains unchanged.

    Notes:
    ------
    - The function modifies A_overline in-place but also returns it
    - Used for converting quadric surfaces from standard to homogeneous form
    - The symmetry of the matrix is preserved by copying b to both row and column

    Examples:
    --------
    >>> import numpy as np
    >>> # Create a 4x4 matrix and a vector
    >>> A = np.zeros((4, 4))
    >>> b = np.array([1, 2, 3])
    >>> result = assign_b_to_overA(A, b)
    >>> print(result)
    [[0. 0. 0. 1.]
     [0. 0. 0. 2.]
     [0. 0. 0. 3.]
     [1. 2. 3. 0.]]

    See Also:
    --------
    numpy.ndarray : For array operations and indexing
    """
    for j in range(0, 3):
        A_overline[3, j] = b[j]
        A_overline[j, 3] = b[j]
    return A_overline

def string2sympy(equation_str):
    """
    Convert a string representation of a quadric equation into a SymPy polynomial object.

    Parameters
    ----------
    equation_str : str
        String containing the quadric equation in the form "left_side = right_side".
        The equation can contain variables x, y, z and supports implicit multiplication.

    Returns
    -------
    sympy.Poly
        SymPy polynomial object representing the quadric expression in standard form
        (all terms moved to left side)

    Raises
    ------
    ValueError
        If the input string does not contain exactly one equals sign

    Examples
    --------
    >>> import sympy as sp
    >>> x, y, z = sp.symbols('x y z')
    >>> string2sympy("x^2 + y^2 = z", x, y, z)
    Poly(x**2 + y**2 - z, x, y, z)
    """
    parts = equation_str.split('=')
    if len(parts) != 2:
        raise ValueError("Input must be a valid equation with a single '=' sign")
    transformations = sp.parsing.sympy_parser.standard_transformations + (sp.parsing.sympy_parser.implicit_multiplication_application,)
    left_expr = sp.parsing.sympy_parser.parse_expr(parts[0].strip(), transformations=transformations)
    right_expr = sp.parsing.sympy_parser.parse_expr(parts[1].strip(), transformations=transformations)
    quadric_expr = left_expr - right_expr
    quadric_expr = sp.Poly(quadric_expr)
    return quadric_expr

def get_matrices(expr):
    """
    Converts a quadric polynomial into its matrix representation A_overline, A and b
    where A_overline is the 4x4 homogeneous matrix, A is the 3x3 coefficient matrix
    of quadratic terms, and b is the 3x1 vector of linear terms.

    Parameters
    ----------
    expr : sympy.Poly
        SymPy polynomial representing the quadric expression

    Returns
    -------
    A tuple containing:
    - A_overline : numpy.ndarray
        4x4 matrix representation in homogeneous coordinates
    - A : numpy.ndarray
        3x3 matrix of quadratic term coefficients
    - b : numpy.ndarray
        3x1 vector of linear term coefficients

    Notes
    -----
    The matrices are constructed as follows:
    - A_overline contains all coefficients in homogeneous form
    - A contains only the coefficients of x², y², z², xy, xz, yz terms
    - b contains the coefficients of the linear x, y, z terms
    - Mixed and linear terms (xy, xz, yz) are divided by 2 in the matrices

    Examples
    --------
    >>> import sympy as sp
    >>> x, y, z = sp.symbols('x y z')
    >>> expr = sp.Poly(x**2 + y**2 - z, x, y, z)
    >>> A_overline, A, b = get_matrices(expr, x, y, z)
    """
    try:
        a11 = float(expr.coeff_monomial(x**2))
    except ValueError:
        a11 = 0
    try:
        a22 = float(expr.coeff_monomial(y**2))
    except ValueError:
        a22 = 0
    try:
        a33 = float(expr.coeff_monomial(z**2))
    except ValueError:
        a33 = 0
    try:
        a44 = float(expr.coeff_monomial(1))
    except ValueError:
        a44 = 0
    try:
        a12 = float(expr.coeff_monomial(x*y) / 2)
    except ValueError:
        a12 = 0
    try:
        a13 = float(expr.coeff_monomial(x*z) / 2)
    except ValueError:
        a13 = 0
    try:
        a14 = float(expr.coeff_monomial(x) / 2)
    except ValueError:
        a14 = 0
    try:
        a21 = float(a12)
    except ValueError:
        a21 = 0
    try:
        a23 = float(expr.coeff_monomial(y*z) / 2)
    except ValueError:
        a23 = 0
    try:
        a24 = float(expr.coeff_monomial(y) / 2)
    except ValueError:
        a24 = 0
    a31 = float(a13)
    a32 = float(a23)
    try:
        a34 = float(expr.coeff_monomial(z) / 2)
    except ValueError:
        a34 = 0
    a41 = float(a14)
    a42 = float(a24)
    a43 = float(a34)
    A_overline = np.array([[a11, a12, a13, a14], [a21, a22, a23, a24], [a31, a32, a33, a34], [a41, a42, a43, a44]])
    A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    b = np.array([[a14], [a24], [a34]])
    return A_overline, A, b

def expr2matrices(eq):
    """
    Converts the string with the quadric surface equation into its matrix representation A_overline, A and b
    where A_overline is the 4x4 homogeneous matrix, A is the 3x3 coefficient matrix
    of quadratic terms, and b is the 3x1 vector of linear terms.

    Parameters
    ----------
    expr : str
        String representing the quadric equation

    Returns
    -------
    A tuple containing:
    - A_overline : numpy.ndarray
        4x4 matrix representation in homogeneous coordinates
    - A : numpy.ndarray
        3x3 matrix of quadratic term coefficients
    - b : numpy.ndarray
        3x1 vector of linear term coefficients

    Notes
    -----
    The matrices are constructed as follows:
    - A_overline contains all coefficients in homogeneous form
    - A contains only the coefficients of x², y², z², xy, xz, yz terms
    - b contains the coefficients of the linear x, y, z terms
    - Mixed and linear terms (xy, xz, yz) are divided by 2 in the matrices
    """
    quadric_expr = string2sympy(eq)
    if quadric_expr is None:
        raise ValueError("Equation is not valid")
    A_overline, A, b = get_matrices(quadric_expr)
    return A_overline, A, b