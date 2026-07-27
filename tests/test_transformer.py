"""Verify canonicalization with ``python -m pytest tests/test_transformer.py -q``."""

import numpy as np
import pytest
import sympy as sp
from scipy.spatial.transform import Rotation

from src.numerical.canonicalize import canonize_quadric
from src.numerical.models import CanonicalizationResult, FloatArray, QuadricType, TransformationKind
from src.numerical.numerical_helpers import expression_from_matrix
from src.numerical.symbols import x, y


_METRIC_CASE_COUNT = 5
_AXIS_LENGTHS = np.array(
    [
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
        [1.25, 2.25, 3.25],
        [0.8, 1.8, 2.8],
        [1.1, 1.7, 2.3],
    ],
    dtype=np.float64,
)
_QUADRATIC_MAGNITUDES = np.reciprocal(np.square(_AXIS_LENGTHS))
_LINEAR_MAGNITUDES = np.array([0.5, 1.25, 0.75, 2.0, 1.5], dtype=np.float64)
_CONSTANT_MAGNITUDES = np.array([1.0, 2.25, 0.64, 3.24, 1.44], dtype=np.float64)
_ROTATION_ANGLES_DEGREES = np.array(
    [
        [18.0, -27.0, 39.0],
        [-33.0, 14.0, 71.0],
        [47.0, 31.0, -22.0],
        [63.0, -19.0, -54.0],
        [-41.0, 58.0, 26.0],
    ],
    dtype=np.float64,
)
_TRANSLATIONS = np.array(
    [
        [1.5, -2.0, 0.75],
        [-3.0, 0.5, 2.25],
        [0.25, 3.5, -1.75],
        [2.75, -1.25, -3.5],
        [-2.5, -3.25, 1.0],
    ],
    dtype=np.float64,
)
_ROTATIONS = np.asarray(
    Rotation.from_euler("xyz", _ROTATION_ANGLES_DEGREES, degrees=True).as_matrix(),
    dtype=np.float64,
)
_INPUT_TO_CANONICAL_TRANSFORMS = np.broadcast_to(
    np.eye(4, dtype=np.float64),
    (_METRIC_CASE_COUNT, 4, 4),
).copy()
_INPUT_TO_CANONICAL_TRANSFORMS[:, :3, :3] = _ROTATIONS
_INPUT_TO_CANONICAL_TRANSFORMS[:, :3, 3] = _TRANSLATIONS


def _canonical_matrix_cases(
    quadratic_signs: tuple[float, float, float],
    linear_direction: tuple[float, float, float],
    constant_sign: float,
) -> FloatArray:
    """
    Construct five coefficient-distinct canonical matrices for one quadric type.

    The arguments define a single canonical zero pattern. The shared positive
    magnitudes vary intrinsic metric coefficients between the five cases.

    Args:
        quadratic_signs: tuple[float, float, float]
            Signs or zeros applied to the three quadratic coefficients.
        linear_direction: tuple[float, float, float]
            Canonical direction and sign of the linear half-coefficient.
        constant_sign: float
            Sign or zero applied to the five constant magnitudes.
        return: numpy.ndarray
            Five symmetric homogeneous matrices with shape (5, 4, 4).
    """

    quadratic = _QUADRATIC_MAGNITUDES * np.asarray(
        quadratic_signs,
        dtype=np.float64,
    )[np.newaxis, :]
    linear = _LINEAR_MAGNITUDES[:, np.newaxis] * np.asarray(
        linear_direction,
        dtype=np.float64,
    )[np.newaxis, :]
    matrices = np.zeros((_METRIC_CASE_COUNT, 4, 4), dtype=np.float64)
    matrices[:, :3, :3] = np.eye(3, dtype=np.float64)[np.newaxis, :, :] * quadratic[:, :, np.newaxis]
    matrices[:, :3, 3] = linear
    matrices[:, 3, :3] = linear
    matrices[:, 3, 3] = constant_sign * _CONSTANT_MAGNITUDES
    return matrices


_CANONICAL_METRIC_MATRICES = {
    QuadricType.REAL_ELLIPSOID: _canonical_matrix_cases((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), -1.0),
    QuadricType.COMPLEX_ELLIPSOID: _canonical_matrix_cases((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 1.0),
    QuadricType.ONE_SHEET_HYPERBOLOID: _canonical_matrix_cases((1.0, 1.0, -1.0), (0.0, 0.0, 0.0), -1.0),
    QuadricType.TWO_SHEET_HYPERBOLOID: _canonical_matrix_cases((1.0, 1.0, -1.0), (0.0, 0.0, 0.0), 1.0),
    QuadricType.REAL_CONE: _canonical_matrix_cases((1.0, 1.0, -1.0), (0.0, 0.0, 0.0), 0.0),
    QuadricType.COMPLEX_CONE: _canonical_matrix_cases((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.0),
    QuadricType.ELLIPTIC_PARABOLOID: _canonical_matrix_cases((1.0, 1.0, 0.0), (0.0, 0.0, -1.0), 0.0),
    QuadricType.HYPERBOLIC_PARABOLOID: _canonical_matrix_cases((1.0, -1.0, 0.0), (0.0, 0.0, -1.0), 0.0),
    QuadricType.REAL_ELLIPTIC_CYLINDER: _canonical_matrix_cases((1.0, 1.0, 0.0), (0.0, 0.0, 0.0), -1.0),
    QuadricType.COMPLEX_ELLIPTIC_CYLINDER: _canonical_matrix_cases((1.0, 1.0, 0.0), (0.0, 0.0, 0.0), 1.0),
    QuadricType.HYPERBOLIC_CYLINDER: _canonical_matrix_cases((1.0, -1.0, 0.0), (0.0, 0.0, 0.0), -1.0),
    QuadricType.REAL_INTERSECTING_PLANES: _canonical_matrix_cases((1.0, -1.0, 0.0), (0.0, 0.0, 0.0), 0.0),
    QuadricType.COMPLEX_INTERSECTING_PLANES: _canonical_matrix_cases((1.0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0),
    QuadricType.PARABOLIC_CYLINDER: _canonical_matrix_cases((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), 0.0),
    QuadricType.REAL_PARALLEL_PLANES: _canonical_matrix_cases((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), -1.0),
    QuadricType.COMPLEX_PARALLEL_PLANES: _canonical_matrix_cases((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0),
    QuadricType.DOUBLE_PLANE: _canonical_matrix_cases((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
}


def _quadric_type_id(quadric_type: QuadricType) -> str:
    """
    Return the stable pytest identifier for one quadric type.

    Args:
        quadric_type: QuadricType
            Quadric classification used by the parameterized test.
        return: str
            Lowercase enum slug.
    """

    return quadric_type.slug


def _ordered_canonical_quadratic_signs(canonical_matrix: FloatArray) -> FloatArray:
    """
    Return the canonical quadratic sign pattern in positive-negative-null order.

    Coefficient magnitudes do not define the canonical form under test. Only
    the number and ordering of positive, negative, and null terms matters.

    Args:
        canonical_matrix: numpy.ndarray
            Expected homogeneous canonical matrix with shape (4, 4).
        return: numpy.ndarray
            Three signs in the canonicalizer's documented eigenvalue order.
    """

    eigenvalues = np.linalg.eigvalsh(canonical_matrix[:3, :3])
    positive = eigenvalues[eigenvalues > 0.0]
    negative = eigenvalues[eigenvalues < 0.0]
    null = np.zeros(3 - positive.size - negative.size, dtype=np.float64)
    return np.concatenate(
        (
            np.ones(positive.size, dtype=np.float64),
            -np.ones(negative.size, dtype=np.float64),
            null,
        )
    )


def _coefficient_signs(coefficients: FloatArray, tolerance: float) -> FloatArray:
    """
    Categorize numerical coefficients as positive, negative, or zero.

    Args:
        coefficients: numpy.ndarray
            Floating-point coefficients with one known numerical scale.
        tolerance: float
            Absolute threshold below which a coefficient represents zero.
        return: numpy.ndarray
            Array of values from {-1.0, 0.0, 1.0} with the input shape.
    """

    return np.where(
        coefficients > tolerance,
        1.0,
        np.where(coefficients < -tolerance, -1.0, 0.0),
    )


def _assert_expected_metric_canonical_form(
    actual_matrix: FloatArray,
    canonical_matrix: FloatArray,
) -> None:
    """
    Verify the coefficient categories and zero pattern of a metric form.

    Proper rotations can reorder eigenaxes and reverse a parabolic axis. The
    comparison therefore checks only the canonical quadratic signs, locations
    of permitted terms, and constant sign. Coefficient magnitudes are
    intentionally not part of the oracle.

    Args:
        actual_matrix: numpy.ndarray
            Homogeneous matrix returned by the canonicalizer.
        canonical_matrix: numpy.ndarray
            Homogeneous matrix before the independent rigid scrambling.
        return: None
            The function raises an assertion failure for a noncanonical result.
    """

    actual_scale = max(float(np.max(np.abs(actual_matrix))), 1.0)
    canonical_scale = max(float(np.max(np.abs(canonical_matrix))), 1.0)
    tolerance = 1e-9 * actual_scale
    canonical_tolerance = 1e-12 * canonical_scale
    quadratic = actual_matrix[:3, :3]
    linear = actual_matrix[:3, 3]

    np.testing.assert_allclose(
        actual_matrix,
        actual_matrix.T,
        atol=tolerance,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        quadratic,
        np.diag(np.diag(quadratic)),
        atol=tolerance,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        _coefficient_signs(np.diag(quadratic), tolerance),
        _ordered_canonical_quadratic_signs(canonical_matrix),
    )
    np.testing.assert_allclose(
        quadratic @ linear,
        np.zeros(3, dtype=np.float64),
        atol=tolerance,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        np.abs(linear) > tolerance,
        np.abs(canonical_matrix[:3, 3]) > canonical_tolerance,
    )
    np.testing.assert_array_equal(
        _coefficient_signs(np.array([actual_matrix[3, 3]]), tolerance),
        _coefficient_signs(
            np.array([canonical_matrix[3, 3]]),
            canonical_tolerance,
        ),
    )


@pytest.mark.parametrize(
    ("equation", "expected_type", "centered"),
    [
        ("2*(x-1)**2 + 3*(y+2)**2 + 4*(z-3)**2 = 1", QuadricType.REAL_ELLIPSOID, True),
        ("(x-1)**2 + (y+2)**2 - z = 0", QuadricType.ELLIPTIC_PARABOLOID, False),
        ("x**2 + y**2 = -1", QuadricType.COMPLEX_ELLIPTIC_CYLINDER, False),
        ("x**2 = 1", QuadricType.REAL_PARALLEL_PLANES, False),
        ("x**2 - y = 0", QuadricType.PARABOLIC_CYLINDER, False),
    ],
)
def test_canonicalizer_returns_typed_canonical_form(equation: str, expected_type: QuadricType, centered: bool) -> None:
    result = canonize_quadric(equation)

    assert isinstance(result, CanonicalizationResult)
    assert result.quadric_type is expected_type
    assert result.centered is centered
    np.testing.assert_allclose(result.rotation_matrix.T @ result.rotation_matrix, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(result.rotation_matrix), 1.0, atol=1e-12)
    np.testing.assert_allclose(result.final_matrix, result.final_matrix.T, atol=1e-12)
    assert np.count_nonzero(np.triu(result.final_matrix[:3, :3], k=1)) == 0


@pytest.mark.parametrize("quadric_type", tuple(QuadricType), ids=_quadric_type_id)
@pytest.mark.parametrize(
    "case_index",
    range(_METRIC_CASE_COUNT),
    ids=("metric_case_1", "metric_case_2", "metric_case_3", "metric_case_4", "metric_case_5"),
)
def test_every_quadric_type_reaches_its_metric_canonical_form(
    quadric_type: QuadricType,
    case_index: int,
) -> None:
    canonical_matrix = _CANONICAL_METRIC_MATRICES[quadric_type][case_index]
    input_to_canonical = _INPUT_TO_CANONICAL_TRANSFORMS[case_index]
    initial_matrix = input_to_canonical.T @ canonical_matrix @ input_to_canonical
    equation = f"{sp.sstr(expression_from_matrix(initial_matrix))} = 0"

    result = canonize_quadric(equation)

    assert result.quadric_type is quadric_type
    rotation_step, translation_step = result.transformation_steps
    assert rotation_step.kind is TransformationKind.ROTATION
    assert translation_step.kind is TransformationKind.TRANSLATION
    np.testing.assert_array_equal(
        rotation_step.offset,
        np.zeros(3, dtype=np.float64),
    )
    np.testing.assert_allclose(
        rotation_step.linear_map.T @ rotation_step.linear_map,
        np.eye(3, dtype=np.float64),
        atol=1e-10,
        rtol=0.0,
    )
    assert np.linalg.det(rotation_step.linear_map) == pytest.approx(1.0, abs=1e-10, rel=0.0)
    np.testing.assert_array_equal(
        translation_step.linear_map,
        np.eye(3, dtype=np.float64),
    )
    _assert_expected_metric_canonical_form(result.final_matrix, canonical_matrix)


@pytest.mark.parametrize(
    "equation",
    [
        "2x**2 + 2*y**2 + 4z**2 - 2xy + 2x = 0",
        "(x-1)**2 + (y+2)**2 - z = 0",
        "x**2 + y**2 = 1",
        "(x-1)**2 - y = 0",
        "x**2 = 1",
    ],
)
def test_reported_active_steps_reconstruct_every_matrix_stage(equation: str) -> None:
    result = canonize_quadric(equation)
    stages = (result.initial_matrix, result.middle_matrix, result.final_matrix)

    for current, expected, step in zip(stages, stages[1:], result.transformation_steps):
        coordinate_change = step.inverse_homogeneous_matrix
        reconstructed = coordinate_change.T @ current @ coordinate_change
        np.testing.assert_allclose(reconstructed, expected, atol=1e-9)


def test_centered_reduction_rotates_before_completing_squares() -> None:
    result = canonize_quadric("2x**2 + 2*y**2 + 4z**2 - 2xy + 2x = 0")
    first_step, second_step = result.transformation_steps

    assert first_step.kind is TransformationKind.ROTATION
    assert second_step.kind is TransformationKind.TRANSLATION
    np.testing.assert_allclose(
        result.middle_matrix[:3, :3],
        np.diag(np.diag(result.middle_matrix[:3, :3])),
        atol=1e-12,
    )
    expected_translation = result.middle_matrix[:3, 3] / np.diag(result.middle_matrix[:3, :3])
    np.testing.assert_allclose(second_step.offset, expected_translation, atol=1e-12)
    np.testing.assert_allclose(result.final_matrix[:3, 3], np.zeros(3), atol=1e-12)
    assert not np.allclose(result.middle_matrix, result.final_matrix)


def test_parabolic_cylinder_reports_minimum_translation_with_the_correct_sign() -> None:
    result = canonize_quadric("(x-1)**2 - 4*(y+2) = 0")
    first_step, second_step = result.transformation_steps

    assert first_step.kind is TransformationKind.ROTATION
    assert second_step.kind is TransformationKind.TRANSLATION
    np.testing.assert_allclose(second_step.offset, np.array([-1.0, -2.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(result.final_matrix[:3, 3], np.array([0.0, 2.0, 0.0]), atol=1e-12)
    assert result.final_matrix[3, 3] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("equation", "expected_signs"),
    [
        ("x**2 + 2*y**2 + 3*z**2 = 1", np.array([1.0, 1.0, 1.0])),
        ("x**2 + 2*y**2 - 3*z**2 = 1", np.array([1.0, 1.0, -1.0])),
        ("x**2 + 2*y**2 - z = 0", np.array([1.0, 1.0, 0.0])),
        ("x**2 - y**2 - z = 0", np.array([1.0, -1.0, 0.0])),
        ("x**2 - y**2 = 1", np.array([1.0, -1.0, 0.0])),
        ("x**2 = 1", np.array([1.0, 0.0, 0.0])),
    ],
)
def test_quadratic_eigenvalues_are_ordered_positive_negative_null(
    equation: str,
    expected_signs: np.ndarray,
) -> None:
    result = canonize_quadric(equation)

    np.testing.assert_array_equal(np.sign(np.diag(result.middle_matrix[:3, :3])), expected_signs)


def test_rank_two_translation_matches_latex_completion_formula() -> None:
    result = canonize_quadric("2*x**2 + 3*y**2 + 4*x - 6*y + 8*z + 7 = 0")
    first_step, second_step = result.transformation_steps

    assert first_step.kind is TransformationKind.ROTATION
    assert second_step.kind is TransformationKind.TRANSLATION
    np.testing.assert_allclose(second_step.offset, np.array([1.0, -1.0, 0.25]), atol=1e-12)
    np.testing.assert_allclose(result.final_matrix[:3, 3], np.array([0.0, 0.0, 4.0]), atol=1e-12)
    assert result.final_matrix[3, 3] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "equation",
    [
        "2*x**2 + 3*y**2 + 4*x - 6*y + 7 = 0",
        "2*x**2 + 4*x - 6 = 0",
    ],
)
def test_degenerate_nonparabolic_translation_completes_nonzero_squares(equation: str) -> None:
    result = canonize_quadric(equation)
    first_step, second_step = result.transformation_steps
    diagonal = np.diag(result.middle_matrix[:3, :3])
    active = diagonal != 0.0
    expected_translation = np.zeros(3)
    expected_translation[active] = result.middle_matrix[:3, 3][active] / diagonal[active]

    assert first_step.kind is TransformationKind.ROTATION
    assert second_step.kind is TransformationKind.TRANSLATION
    np.testing.assert_allclose(second_step.offset, expected_translation, atol=1e-12)
    np.testing.assert_allclose(result.final_matrix[:3, 3][active], np.zeros(np.count_nonzero(active)), atol=1e-12)


def test_result_preserves_full_precision_rotation() -> None:
    result = canonize_quadric("2*x**2 + 2*x*y + 2*y**2 + z**2 = 1")

    assert np.any(np.abs(result.rotation_matrix - np.round(result.rotation_matrix, decimals=2)) > 1e-6)
    np.testing.assert_allclose(result.rotation_matrix.T @ result.rotation_matrix, np.eye(3), atol=1e-12)


def test_canonicalization_is_invariant_to_equation_scaling() -> None:
    reference = canonize_quadric("x**2 + y**2 + z**2 = 1")
    scaled = canonize_quadric("0.000001*x**2 + 0.000001*y**2 + 0.000001*z**2 = 0.000001")

    assert scaled.quadric_type is reference.quadric_type
    np.testing.assert_allclose(
        scaled.final_matrix / np.max(np.abs(scaled.final_matrix)),
        reference.final_matrix / np.max(np.abs(reference.final_matrix)),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("equation", "symbol", "expected_coefficient"),
    [
        ("x**2 + 0.0000000000001*y**2 = 1", y, 1e-13),
        ("1.00000000005*x**2 + y**2 = 1", x, 1.00000000005),
    ],
)
def test_reported_equations_preserve_nonzero_matrix_coefficients(
    equation: str,
    symbol: sp.Symbol,
    expected_coefficient: float,
) -> None:
    result = canonize_quadric(equation)

    assert float(result.initial_equation.coeff(symbol, 2)) == expected_coefficient
