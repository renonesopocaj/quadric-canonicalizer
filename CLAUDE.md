# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quadric Canonicalizer: transforms a quadric surface equation into canonical form via translation and rotation, classifies the quadric type (17 types), and renders animated 3D visualizations using Manim.

## Commands

```bash
# Run the interactive program
python src/main.py

# Run tests (from project root)
python test/tester.py

# Render a Manim scene directly
manim -pql src/graphics/scene_render.py SceneRender
```

## Dependencies

- **Numerical**: sympy, numpy, scipy, matplotlib
- **Graphics**: manim (community edition), numpy
- **System**: Python 3.8+, FFMPEG, LaTeX

## Architecture

**Data flow**: User input → `main.py` → `transformer.canonize_quadric()` → result dict → `SceneRender` (Manim animation)

Two module groups under `src/`:

### `src/numerical/` — Math pipeline
- **`transformer.py`** — Orchestrates canonicalization: `canonize_quadric()` is the main entry point. Handles centered vs non-centered quadrics, applies translation then rotation.
- **`quadric_canonicalizer.py`** — OOP wrapper (`Quadric` class) around the transformer logic.
- **`classifier.py`** — Determines quadric type (1-17) via eigenvalue signature analysis on the 4×4 matrix `A_overline` and 3×3 matrix `A`.
- **`misc.py`** — Utilities: `expr2matrices()` parses equation strings into matrix form (`A_overline`, `A`, `b`).
- **`parabolic_cylinder.py`** — Special-case handling for type 14 (parabolic cylinders) using symbolic solving.
- **`checker.py`** — Validation functions to verify transformations are correct. Functions accept `bypass=True` to skip checks.

### `src/graphics/` — Manim visualization
- **`scene_render.py`** — `SceneRender(ThreeDScene)`: orchestrates the animation sequence.
- **`create_quadric_surface.py`** — Parametric surface generation for all 17 quadric types.
- **`create_text_overlay.py`** — Equation and matrix LaTeX rendering positioned in the scene.

### Key data structure

The result dict from `canonize_quadric()` contains: `quadric type` (int 1-17), `final/initial/middle quadric matrix` (numpy 4×4), `translation vector` (3×1), `rotation matrix` (3×3), and `initial/middle/final quadric equation` (sympy expressions).

## Conventions

- Matrix naming: `A_overline` = 4×4 homogeneous, `A` = 3×3, `b` = 3×1 linear terms
- Symbols `x, y, z` are SymPy globals declared at module level
- Near-zero threshold: `1e-10` used throughout for numerical comparisons
- Imports use `src.numerical` / `src.graphics` prefix (e.g., `from src.numerical.misc import ...`)
- Custom exceptions: `NotAQuadricException`, `NotSupportedException`, `WrongCheckException`
- Quadric types mapped via `ENUM_QUADRICS` dict (string name → int 1-17)

## Known Limitations

- Complex quadric types (2, 6, 10, 13, 16) cannot be rendered graphically
- Classifier does not distinguish real vs complex elliptic cylinders or parallel planes
- Parabolic cylinder (type 14) has numerical stability issues with certain coefficients
