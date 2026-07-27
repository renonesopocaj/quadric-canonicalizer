"""
Classify real three-dimensional quadrics from matrix invariants.

Run the classifier tests with ``python -m pytest tests/test_classifier.py -q``.
"""

from __future__ import annotations

import numpy as np

from src.numerical.models import FloatArray, MatrixInertia, QuadricType
from src.numerical.numerical_helpers import numerical_rank, relative_tolerance
from src.numerical.parser import QuadricParser


class NotAQuadricError(ValueError):
    """Indicate that matrix invariants do not describe a supported quadric."""


# Compatibility alias retained for existing callers.
NotAQuadricException = NotAQuadricError
ROUNDOFF_FACTOR = 100.0


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
        scale = float(np.max(np.abs(quadratic)))
        if scale == 0:
            return MatrixInertia(positive=0, negative=0, zero=3)
        eigenvalues = np.linalg.eigvalsh(quadratic / scale)
        threshold = relative_tolerance(eigenvalues, ROUNDOFF_FACTOR)
        positive = int(np.count_nonzero(eigenvalues > threshold))
        negative = int(np.count_nonzero(eigenvalues < -threshold))
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

        scale = float(np.max(np.abs(homogeneous)))
        if scale == 0:
            raise NotAQuadricError("homogeneous quadric matrix cannot be identically zero")
        normalized_homogeneous = homogeneous / scale
        normalized_quadratic = quadratic / scale
        rank_quadratic = numerical_rank(normalized_quadratic, ROUNDOFF_FACTOR)
        rank_homogeneous = numerical_rank(normalized_homogeneous, ROUNDOFF_FACTOR)
        determinant = (
            float(np.linalg.slogdet(normalized_homogeneous)[0])
            if rank_homogeneous == 4
            else 0.0
        )
        inertia = self.inertia(normalized_quadratic)

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
                return self._elliptic_cylinder_type(normalized_quadratic, normalized_homogeneous, inertia)
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
                return self._parallel_planes_type(normalized_quadratic, normalized_homogeneous, inertia)
            if rank_homogeneous == 1:
                return QuadricType.DOUBLE_PLANE

        raise NotAQuadricError(
            "unsupported quadric invariants: "
            f"rank(A)={rank_quadratic}, rank(A_overline)={rank_homogeneous}, "
            f"det(A_overline)={determinant}, inertia={inertia}"
        )

    def _reduced_constant(self, quadratic: FloatArray, homogeneous: FloatArray) -> float:
        linear = homogeneous[:3, 3]
        center = -np.linalg.pinv(
            quadratic,
            rcond=ROUNDOFF_FACTOR * float(np.finfo(np.float64).eps),
        ) @ linear
        return float(center @ quadratic @ center + 2 * linear @ center + homogeneous[3, 3])

    def _semidefinite_has_real_points(self, quadratic: FloatArray, homogeneous: FloatArray, inertia: MatrixInertia) -> bool:
        coefficient_sign = 1.0 if inertia.positive > 0 else -1.0
        return coefficient_sign * self._reduced_constant(quadratic, homogeneous) < 0.0

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
