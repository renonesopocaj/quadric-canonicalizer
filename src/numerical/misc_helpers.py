from src.numerical.symbols import x, y, z
import numpy.typing as npt
import numpy as np
import sympy as sp

def expression_from_matrix(matrix: npt.ArrayLike) -> sp.Expr:
    """
    Reconstruct and expand a quadric expression from a 4x4 matrix.

    Args:
        matrix: numpy.typing.ArrayLike
            Homogeneous 4x4 quadric matrix.
    return: sympy.Expr
        Expanded expression in x, y, and z.
    """

    array = np.asarray(matrix)
    if array.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4)")
    coordinates = sp.Matrix([[x, y, z, 1]])
    return sp.expand((coordinates * sp.Matrix(array) * coordinates.T)[0, 0])
