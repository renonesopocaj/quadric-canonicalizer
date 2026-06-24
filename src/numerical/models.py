"""
Define typed contracts shared by the numerical and graphical pipelines.

Run the contract tests with ``python -m pytest tests/test_models.py -q``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import sympy as sp
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

FloatArray = npt.NDArray[np.float64]


class QuadricType(IntEnum):
    """Identify every supported mathematical quadric classification."""

    REAL_ELLIPSOID = 1
    COMPLEX_ELLIPSOID = 2
    ONE_SHEET_HYPERBOLOID = 3
    TWO_SHEET_HYPERBOLOID = 4
    REAL_CONE = 5
    COMPLEX_CONE = 6
    ELLIPTIC_PARABOLOID = 7
    HYPERBOLIC_PARABOLOID = 8
    REAL_ELLIPTIC_CYLINDER = 9
    COMPLEX_ELLIPTIC_CYLINDER = 10
    HYPERBOLIC_CYLINDER = 11
    REAL_INTERSECTING_PLANES = 12
    COMPLEX_INTERSECTING_PLANES = 13
    PARABOLIC_CYLINDER = 14
    REAL_PARALLEL_PLANES = 15
    COMPLEX_PARALLEL_PLANES = 16
    DOUBLE_PLANE = 17

    @property
    def slug(self) -> str:
        """Return the stable filename-safe name used by rendered videos."""

        return self.name.lower()


@dataclass(frozen=True, slots=True)
class QuadricMatrices:
    """
    Store one quadric in homogeneous, quadratic, and linear matrix form.

    Args:
        homogeneous: numpy.ndarray
            Symmetric 4x4 homogeneous matrix.
        quadratic: numpy.ndarray
            Symmetric 3x3 matrix of quadratic coefficients.
        linear: numpy.ndarray
            Three-row column vector containing half the linear coefficients.
    return: QuadricMatrices
        Validated matrix bundle used by classification and canonicalization.
    """

    homogeneous: FloatArray
    quadratic: FloatArray
    linear: FloatArray

    def __post_init__(self) -> None:
        if self.homogeneous.shape != (4, 4):
            raise ValueError("homogeneous matrix must have shape (4, 4)")
        if self.quadratic.shape != (3, 3):
            raise ValueError("quadratic matrix must have shape (3, 3)")
        if self.linear.shape != (3, 1):
            raise ValueError("linear vector must have shape (3, 1)")


@dataclass(frozen=True, slots=True)
class MatrixInertia:
    """
    Store counts of positive, negative, and zero eigenvalues.

    Args:
        positive: int
            Number of positive eigenvalues.
        negative: int
            Number of negative eigenvalues.
        zero: int
            Number of eigenvalues within the numerical zero tolerance.
    return: MatrixInertia
        Inertia used by the classifier decision table.
    """

    positive: int
    negative: int
    zero: int

    @property
    def is_definite(self) -> bool:
        """Return whether all eigenvalues have the same non-zero sign."""

        return self.positive == 3 or self.negative == 3

    @property
    def is_semidefinite(self) -> bool:
        """Return whether non-zero eigenvalues have one sign and at least one zero."""

        return self.zero > 0 and ((self.positive > 0 and self.negative == 0) or (self.negative > 0 and self.positive == 0))

    @property
    def is_indefinite(self) -> bool:
        """Return whether positive and negative eigenvalues are both present."""

        return self.positive > 0 and self.negative > 0


class CanonicalizationResult(BaseModel):
    """
    Validate and expose every artifact produced by canonicalization.

    Args:
        quadric_type: QuadricType
            Mathematical classification of the input equation.
        centered: bool
            Whether the quadratic block has full rank and therefore a center.
        initial_matrix: numpy.ndarray
            Original 4x4 homogeneous matrix.
        middle_matrix: numpy.ndarray
            Matrix after the first transformation.
        final_matrix: numpy.ndarray
            Matrix in canonical metric form.
        translation_vector: numpy.ndarray
            Translation represented as a three-element vector.
        rotation_matrix: numpy.ndarray
            Orthogonal 3x3 rotation matrix.
        initial_equation: sympy.Expr
            Expression reconstructed from the initial matrix.
        middle_equation: sympy.Expr
            Expression reconstructed from the intermediate matrix.
        final_equation: sympy.Expr
            Expression reconstructed from the canonical matrix.
    return: CanonicalizationResult
        Immutable validated public result model.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    quadric_type: QuadricType
    centered: bool
    initial_matrix: FloatArray
    middle_matrix: FloatArray
    final_matrix: FloatArray
    translation_vector: FloatArray
    rotation_matrix: FloatArray
    initial_equation: sp.Expr
    middle_equation: sp.Expr
    final_equation: sp.Expr

    @field_validator("initial_matrix", "middle_matrix", "final_matrix", "translation_vector", "rotation_matrix",
                     mode="before")
    @classmethod
    def coerce_array(cls, value: object) -> FloatArray:
        """Convert array-like model input to an owned, read-only float array."""

        array = np.asarray(value, dtype=np.float64).copy()
        array.setflags(write=False)
        return array

    @field_validator("translation_vector")
    @classmethod
    def flatten_translation_vector(cls, value: FloatArray) -> FloatArray:
        """Normalize every valid translation representation to shape ``(3,)``."""

        if value.size != 3:
            raise ValueError("translation_vector must contain exactly three values")
        return value.reshape(3)

    @model_validator(mode="after")
    def validate_shapes(self) -> CanonicalizationResult:
        """Reject incomplete transformation results at the public boundary."""

        for field_name in ("initial_matrix", "middle_matrix", "final_matrix"):
            if getattr(self, field_name).shape != (4, 4):
                raise ValueError(f"{field_name} must have shape (4, 4)")
        if self.rotation_matrix.shape != (3, 3):
            raise ValueError("rotation_matrix must have shape (3, 3)")
        return self

    def __getitem__(self, key: str) -> object:
        """Support legacy string keys while callers migrate to typed attributes."""

        legacy_fields = {
            "quadric type": self.quadric_type,
            "centered quadric": self.centered,
            "initial quadric matrix": self.initial_matrix,
            "middle quadric matrix": self.middle_matrix,
            "final quadric matrix": self.final_matrix,
            "translation vector": self.translation_vector,
            "rotation matrix": self.rotation_matrix,
            "initial quadric equation": self.initial_equation,
            "middle quadric equation": self.middle_equation,
            "final quadric equation": self.final_equation,
        }
        try:
            return legacy_fields[key]
        except KeyError as error:
            raise KeyError(key) from error
