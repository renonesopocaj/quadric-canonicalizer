"""
Define typed contracts shared by the numerical and graphical pipelines.

Run the contract tests with ``python -m pytest tests/test_models.py -q``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import sympy as sp
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

FloatArray = npt.NDArray[np.float64]


class TransformationKind(StrEnum):
    """Identify the active point transformation performed between two stages."""

    TRANSLATION = "translation"
    ROTATION = "rotation"


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

        has_single_sign = (
            (self.positive > 0 and self.negative == 0)
            or (self.negative > 0 and self.positive == 0)
        )
        return self.zero > 0 and has_single_sign

    @property
    def is_indefinite(self) -> bool:
        """Return whether positive and negative eigenvalues are both present."""

        return self.positive > 0 and self.negative > 0


class AffineTransformation(BaseModel):
    """
    Represent one active affine transformation between equation stages.

    Points use the convention ``next = linear_map @ current + offset``. The
    inverse homogeneous matrix is therefore the coordinate substitution used
    to transform the corresponding homogeneous quadric matrices.

    Args:
        kind: TransformationKind
            Whether the step is reported as a translation or a rotation.
        linear_map: numpy.ndarray
            Three-dimensional active linear map.
        offset: numpy.ndarray
            Three-dimensional active translation.
        return: AffineTransformation
            Immutable, validated active point transformation.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    kind: TransformationKind
    linear_map: FloatArray
    offset: FloatArray

    @field_validator("linear_map", "offset", mode="before")
    @classmethod
    def coerce_array(cls, value: object) -> FloatArray:
        """Convert array-like input to an owned, read-only float array."""

        array = np.asarray(value, dtype=np.float64).copy()
        array.setflags(write=False)
        return array

    @model_validator(mode="after")
    def validate_transform(self) -> AffineTransformation:
        """Reject malformed or non-rigid transformation steps."""

        if self.linear_map.shape != (3, 3):
            raise ValueError("linear_map must have shape (3, 3)")
        if self.offset.shape != (3,):
            raise ValueError("offset must have shape (3,)")
        if not np.all(np.isfinite(self.linear_map)) or not np.all(np.isfinite(self.offset)):
            raise ValueError("affine transformation values must be finite")
        if self.kind is TransformationKind.TRANSLATION:
            if not np.allclose(self.linear_map, np.eye(3), atol=1e-12, rtol=0):
                raise ValueError("a translation step must use the identity linear map")
        else:
            if not np.allclose(self.offset, np.zeros(3), atol=1e-12, rtol=0):
                raise ValueError("a rotation step must have zero offset")
            if not np.allclose(self.linear_map.T @ self.linear_map, np.eye(3), atol=1e-10, rtol=0):
                raise ValueError("a rotation step must be orthogonal")
            if not np.isclose(np.linalg.det(self.linear_map), 1.0, atol=1e-10, rtol=0):
                raise ValueError("a rotation step must have determinant one")
        return self

    @property
    def homogeneous_matrix(self) -> FloatArray:
        """Return the 4x4 active point transformation matrix."""

        homogeneous = np.eye(4, dtype=np.float64)
        homogeneous[:3, :3] = self.linear_map
        homogeneous[:3, 3] = self.offset
        homogeneous.setflags(write=False)
        return homogeneous

    @property
    def inverse_homogeneous_matrix(self) -> FloatArray:
        """Return the coordinate substitution from the next stage to the current stage."""

        inverse = np.asarray(np.linalg.inv(self.homogeneous_matrix), dtype=np.float64)
        inverse.setflags(write=False)
        return inverse


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
            Active point translation performed in the rotated coordinate frame.
        rotation_matrix: numpy.ndarray
            Active proper 3x3 rotation performed during canonicalization.
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
        """Reject incomplete or internally inconsistent results at the public boundary."""

        for field_name in ("initial_matrix", "middle_matrix", "final_matrix"):
            if getattr(self, field_name).shape != (4, 4):
                raise ValueError(f"{field_name} must have shape (4, 4)")
        if self.rotation_matrix.shape != (3, 3):
            raise ValueError("rotation_matrix must have shape (3, 3)")
        if not np.allclose(self.rotation_matrix.T @ self.rotation_matrix, np.eye(3), atol=1e-10, rtol=0):
            raise ValueError("rotation_matrix must be orthogonal")
        if not np.isclose(np.linalg.det(self.rotation_matrix), 1.0, atol=1e-10, rtol=0):
            raise ValueError("rotation_matrix must have determinant one")
        for field_name in (
            "initial_matrix",
            "middle_matrix",
            "final_matrix",
            "translation_vector",
            "rotation_matrix",
        ):
            if not np.all(np.isfinite(getattr(self, field_name))):
                raise ValueError(f"{field_name} must contain finite values")
        stages = (self.initial_matrix, self.middle_matrix, self.final_matrix)
        for current, expected, step in zip(stages, stages[1:], self.transformation_steps):
            coordinate_change = step.inverse_homogeneous_matrix
            reconstructed = coordinate_change.T @ current @ coordinate_change
            comparison_scale = max(float(np.max(np.abs(expected))), 1.0)
            if not np.allclose(
                reconstructed,
                expected,
                atol=1e-9 * comparison_scale,
                rtol=1e-9,
            ):
                raise ValueError(f"{step.kind.value} step does not reconstruct its reported matrix stage")
        return self

    @property
    def transformation_steps(self) -> tuple[AffineTransformation, AffineTransformation]:
        """
        Return the two ordered active transformations from initial to canonical form.

        Args:
            return: tuple[AffineTransformation, AffineTransformation]
                Initial-to-middle and middle-to-final point transformations.
        """

        translation = AffineTransformation(
            kind=TransformationKind.TRANSLATION,
            linear_map=np.eye(3, dtype=np.float64),
            offset=self.translation_vector,
        )
        rotation = AffineTransformation(
            kind=TransformationKind.ROTATION,
            linear_map=self.rotation_matrix,
            offset=np.zeros(3, dtype=np.float64),
        )
        return rotation, translation

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
