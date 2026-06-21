"""Own shared SymPy symbols. Run checks with ``python -m pytest tests/test_parser.py -q``."""

import sympy as sp

x, y, z = sp.symbols('x y z')

__all__ = ["x", "y", "z"]
