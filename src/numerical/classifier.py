"""
Classify real three-dimensional quadrics from matrix invariants.

Run the classifier tests with ``python -m pytest tests/test_classifier.py -q``.
"""

from __future__ import annotations

import numpy as np

from src.numerical.models import FloatArray, MatrixInertia, QuadricType
from src.numerical.parser import QuadricParser


class NotAQuadricError(ValueError):
    """Indicate that matrix invariants do not describe a supported quadric."""


# Compatibility alias retained for existing callers.
NotAQuadricException = NotAQuadricError


class QuadricClassifier:
    """Classify a quadric through rank, determinant, and matrix inertia."""

    tolerance: float

    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance

    def inertia(self, quadratic: FloatArray) -> MatrixInertia:
        """
        Count positive, negative, and numerically zero eigenvalues.

        Args:
            quadratic: numpy.ndarray
                Symmetric 3x3 quadratic coefficient matrix.
            return: MatrixInertia
                Eigenvalue sign counts under the configured tolerance.
        """

        if quadratic.shape != (3, 3):
            raise ValueError("quadratic matrix must have shape (3, 3)")
        eigenvalues = np.linalg.eigvalsh(quadratic)
        positive = int(np.count_nonzero(eigenvalues > self.tolerance))
        negative = int(np.count_nonzero(eigenvalues < -self.tolerance))
        return MatrixInertia(positive=positive, negative=negative, zero=3 - positive - negative)

    def classify(self, quadratic: FloatArray, homogeneous: FloatArray) -> QuadricType:
        """
        Return the unique quadric type selected by the invariant decision table.

        Args:
            quadratic: numpy.ndarray
                Symmetric 3x3 quadratic block.
            homogeneous: numpy.ndarray
                Symmetric 4x4 homogeneous matrix.
            return: QuadricType
                Classified real or complex quadric family.
        """

        if quadratic.shape != (3, 3) or homogeneous.shape != (4, 4):
            raise ValueError("expected quadratic shape (3, 3) and homogeneous shape (4, 4)")

        rank_quadratic = int(np.linalg.matrix_rank(quadratic, tol=self.tolerance))
        rank_homogeneous = int(np.linalg.matrix_rank(homogeneous, tol=self.tolerance))
        determinant = float(np.linalg.det(homogeneous))
        if np.isclose(determinant, 0, atol=self.tolerance, rtol=0):
            determinant = 0.0
        inertia = self.inertia(quadratic)

        if rank_quadratic == 3:
            if determinant < 0 and inertia.is_definite:
                return QuadricType.REAL_ELLIPSOID
            if determinant < 0 and inertia.is_indefinite:
                return QuadricType.TWO_SHEET_HYPERBOLOID
            if determinant > 0 and inertia.is_definite:
                return QuadricType.COMPLEX_ELLIPSOID
            if determinant > 0 and inertia.is_indefinite:
                return QuadricType.ONE_SHEET_HYPERBOLOID
            if determinant == 0 and rank_homogeneous == 3 and inertia.is_definite:
                return QuadricType.COMPLEX_CONE
            if determinant == 0 and rank_homogeneous == 3 and inertia.is_indefinite:
                return QuadricType.REAL_CONE

        if rank_quadratic == 2:
            if determinant < 0 and inertia.is_semidefinite:
                return QuadricType.ELLIPTIC_PARABOLOID
            if determinant > 0 and inertia.is_indefinite:
                return QuadricType.HYPERBOLIC_PARABOLOID
            if determinant == 0 and rank_homogeneous == 3 and inertia.is_semidefinite:
                return self._elliptic_cylinder_type(quadratic, homogeneous, inertia)
            if determinant == 0 and rank_homogeneous == 3 and inertia.is_indefinite:
                return QuadricType.HYPERBOLIC_CYLINDER
            if determinant == 0 and rank_homogeneous == 2 and inertia.is_indefinite:
                return QuadricType.REAL_INTERSECTING_PLANES
            if determinant == 0 and rank_homogeneous == 2 and inertia.is_semidefinite:
                return QuadricType.COMPLEX_INTERSECTING_PLANES

        if rank_quadratic == 1:
            if rank_homogeneous == 3:
                return QuadricType.PARABOLIC_CYLINDER
            if rank_homogeneous == 2:
                return self._parallel_planes_type(quadratic, homogeneous, inertia)
            if rank_homogeneous == 1:
                return QuadricType.DOUBLE_PLANE

        raise NotAQuadricError(
            "unsupported quadric invariants: "
            f"rank(A)={rank_quadratic}, rank(A_overline)={rank_homogeneous}, "
            f"det(A_overline)={determinant}, inertia={inertia}"
        )

    def _reduced_constant(self, quadratic: FloatArray, homogeneous: FloatArray) -> float:
        linear = homogeneous[:3, 3]
        center = -np.linalg.pinv(quadratic, rcond=self.tolerance) @ linear
        return float(center @ quadratic @ center + 2 * linear @ center + homogeneous[3, 3])

    def _semidefinite_has_real_points(self, quadratic: FloatArray, homogeneous: FloatArray, inertia: MatrixInertia) -> bool:
        coefficient_sign = 1.0 if inertia.positive > 0 else -1.0
        return coefficient_sign * self._reduced_constant(quadratic, homogeneous) < -self.tolerance

    def _elliptic_cylinder_type(self, quadratic: FloatArray, homogeneous: FloatArray, inertia: MatrixInertia) -> QuadricType:
        if self._semidefinite_has_real_points(quadratic, homogeneous, inertia):
            return QuadricType.REAL_ELLIPTIC_CYLINDER
        return QuadricType.COMPLEX_ELLIPTIC_CYLINDER

    def _parallel_planes_type(self, quadratic: FloatArray, homogeneous: FloatArray, inertia: MatrixInertia) -> QuadricType:
        if self._semidefinite_has_real_points(quadratic, homogeneous, inertia):
            return QuadricType.REAL_PARALLEL_PLANES
        return QuadricType.COMPLEX_PARALLEL_PLANES


def get_eigenvalues_multiplicities(A: FloatArray, tol: float) -> MatrixInertia:
    """
    Return typed eigenvalue sign multiplicities for compatibility.

    Args:
        A: numpy.ndarray
            Symmetric quadratic matrix.
        tol: float
            Numerical zero tolerance.
        return: MatrixInertia
            Positive, negative, and zero eigenvalue counts.
    """

    return QuadricClassifier(tolerance=tol).inertia(A)


def classify_quadric(A: FloatArray, A_overline: FloatArray) -> QuadricType:
    """
    Classify matrices through the default numerical tolerance.

    Args:
        A: numpy.ndarray
            Symmetric 3x3 quadratic matrix.
        A_overline: numpy.ndarray
            Symmetric 4x4 homogeneous matrix.
        return: QuadricType
            Quadric classification enum, also compatible with integer comparisons.
    """

    return QuadricClassifier(tolerance=1e-10).classify(A, A_overline)


def expr2classification(eq: str) -> QuadricType:
    """
    Parse and classify a degree-two equation.

    Args:
        eq: str
            Quadric equation in x, y, and z.
        return: QuadricType
            Quadric classification enum.
    """

    matrices = QuadricParser().parse_matrices(eq)
    return classify_quadric(matrices.quadratic, matrices.homogeneous)


__all__ = [
    "NotAQuadricError",
    "NotAQuadricException",
    "QuadricClassifier",
    "classify_quadric",
    "expr2classification",
    "get_eigenvalues_multiplicities",
]
