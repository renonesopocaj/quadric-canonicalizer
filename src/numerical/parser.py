"""
Parse quadric equations and construct their matrix representation.

Run the parser tests with ``python -m pytest tests/test_parser.py -q``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from src.numerical.models import QuadricMatrices
from src.numerical.symbols import x, y, z


class QuadricParser:
    """Parse external equation strings into validated numerical matrix bundles."""

    _transformations: tuple[Any, ...]
    _local_symbols: dict[str, sp.Symbol]

    _transformations = standard_transformations + (implicit_multiplication_application,)
    _local_symbols = {"x": x, "y": y, "z": z}

    def parse(self, equation: str) -> sp.Poly:
        """
        Parse one equation and require a polynomial of total degree exactly two.

        Args:
            equation: str
                Equation with one equals sign and variables x, y, and z.
            return: sympy.Poly
                Expanded polynomial after moving the right side to the left.
        """

        parts = equation.split("=")
        if len(parts) != 2:
            raise ValueError("equation must contain exactly one '=' sign")
        left = parse_expr(parts[0].strip(), local_dict=self._local_symbols, transformations=self._transformations)
        right = parse_expr(parts[1].strip(), local_dict=self._local_symbols, transformations=self._transformations)
        polynomial = sp.Poly(sp.expand(left - right), x, y, z)
        if not polynomial.free_symbols.issubset({x, y, z}):
            raise ValueError("equation may only contain variables x, y, and z")
        if polynomial.total_degree() != 2:
            raise ValueError("equation must be a polynomial of total degree two")
        return polynomial

    def matrices_from_polynomial(self, polynomial: sp.Poly) -> QuadricMatrices:
        """
        Build homogeneous, quadratic, and linear matrices from a polynomial.

        Args:
            polynomial: sympy.Poly
                Degree-two polynomial in x, y, and z.
            return: QuadricMatrices
                Float matrices using half-coefficients for mixed and linear terms.
        """

        coefficient = lambda monomial: float(polynomial.coeff_monomial(monomial))
        quadratic = np.array(
            [
                [coefficient(x**2), coefficient(x * y) / 2, coefficient(x * z) / 2],
                [coefficient(x * y) / 2, coefficient(y**2), coefficient(y * z) / 2],
                [coefficient(x * z) / 2, coefficient(y * z) / 2, coefficient(z**2)],
            ],
            dtype=np.float64,
        )
        linear = np.array(
            [[coefficient(x) / 2], [coefficient(y) / 2], [coefficient(z) / 2]], dtype=np.float64
        )
        homogeneous = np.zeros((4, 4), dtype=np.float64)
        homogeneous[:3, :3] = quadratic
        homogeneous[:3, 3:] = linear
        homogeneous[3:, :3] = linear.T
        homogeneous[3, 3] = coefficient(1)
        return QuadricMatrices(homogeneous=homogeneous, quadratic=quadratic, linear=linear)

    def parse_matrices(self, equation: str) -> QuadricMatrices:
        """
        Parse an equation directly into its three matrix forms.

        Args:
            equation: str
                Degree-two equation accepted by :meth:`parse`.
            return: QuadricMatrices
                Validated matrices for the equation.
        """

        return self.matrices_from_polynomial(self.parse(equation))
