# quadric-canonicalizer

A program that brings a quadric surface in metric canonical form and renders the transformation.
The mathematical foundations of the algorithm were drawn from 3 main sources, see the bibliography.
Main things it currently doesn't support:
- rendering of complex quadrics: complex ellipsoid, complex cone, complex elliptic cylinder, complex intersecting planes, complex parallel planes
- when using function `classifier`, distinguishing between complex vs real elliptic cylinder and complex vs real parallel planes is not supported yet, even though transformer still works

## Program

Let's divide the program into "main", "numerical part", and "graphic part".

### Main

The main receives as input the equation of the quadric, the filepath (path where to place video files rendered by Manim), and the desired output quality. The main first executes the numerical part (canonize_quadric in transformer.py) to bring the quadric into canonical form and obtain all the necessary information for the graphic part, and then renders the transformation (scene_render.py).

### Numeric part

#### Functioning

The numerical part is responsible for transforming the quadric into canonical form through a translation and a rotation. Currently, the two transformations is executed in two orders:
- In the case of centered quadrics and the parabolic cylinder, the translation is performed first, followed by the rotation.
- In other non-centered quadrics, the order is reversed.
The order is important because it affects the visualization of the quadric transformation, so the graphical part.
The numerical part saves, in a dictionary that is returned to the graphic part, all the necessary information for the latter: a translation vector and a rotation matrix, the matrix $\overline A$ (in three versions: original, after the first transformation, and after the second), the three equations of the quadric corresponding to the three versions of $\overline A$, and the type of quadric.
The process of bringing the quadric into canonical form is described in detail, from a mathematical point of view, in the wiki "applying transformations to the quadric"

#### Modules

The numerical part is divided into the following modules:
- **transformer.py**: contains the function `canonize_quadric` which takes as input a string representing the equation of the quadric and outputs a dictionary containing various information about the quadric and its transformations. It includes various support functions that are responsible for bringing the quadric into canonical metric form and calculating the necessary transformations to do so.
- **tester.py**: contains `test_quadrics`, a function that generates quadrics for each desired type of quadric and checks that it is classified correctly. Currently, it does not check that the expression/the matrix $\overline A$ of the quadric in canonical metric form (output by `canonize_quadrics`) actually contains only the terms expected from the classification theorem. See "testing" for details.
- **checker.py**: contains functions that serve to perform various correctness checks of the algorithm.
- **parabolic_cylinder.py**: contains the function `parabolic_cylinder_canonize` which calculates the necessary transformations to bring a parabolic cylinder into canonical metric form, along with the various support functions. It is located in a separate module because it is calculated differently.
- **misc**: various support functions for other functions and for converting strings into SymPy expressions.
- **classifier.py**: contains a function `classify_quadric` which returns an integer that classifies the quadric, based on its matrices (each integer is mapped to a type of quadric according to the dictionary `ENUM_QUADRICS`). To do this starting from the string containing the equation, use `expr2classification`.
- 
#### Testing

For details, see the module `tester.py`.

The testing of the numerical part was performed by randomly generating quadrics of known types and verifying:
- Whether the quadric is correctly classified, which can be fully automated in `tester.py`.
- Whether the canonical metric form "reflects" the expected canonical form (for example, an ellipsoid should not contain a rectangular or linear term in canonical metric form) – this aspect (let's call it "term checking") has not yet been automated in `tester.py` but has been done manually.

Testing was performed for each type of quadric, although not in the most general way, by inserting coefficients that introduce random rotations/translations starting from the implicit canonical metric forms.

No errors have been encountered so far, except in one case: the parabolic cylinder. In fact, it sometimes fails to be brought into canonical metric form correctly, probably due to repeated rounding errors caused by SymPy (which generally struggles to perform the required calculations accurately).

It could be tested completely randomly (i.e., starting from a generic second-degree polynomial with random coefficients) once the aforementioned "term checking" has been automated.

#### TODOs

Let's summarize things to improve in the code:

- Currently, there is no implementation to distinguish real elliptic cylinders from complex ones and real parallel planes from complex ones.
- The parabolic cylinder has cases that are not correctly brought into canonical form. This could be resolved in two ways:
    - Use a method that avoids symbolic resolutions.
    - Ensure better handling of rounding errors due to floating point arithmetic.
- Implement a way to check errors by examining the coefficients of the terms.
- Implement a robust method for error tolerance in floating point arithmetic.
- Ensure all matrices have $\det S=1$, i.e., permute those with $\det S=-1$ (and the corresponding rows in $D$).
- Optional, only if generalizing to hyperquadrics is needed: use permutation matrices that reduce a generic quadric to a specific permutation of indeterminates, instead of the various if-else statements for each possible permutation.

### Graphical part

The graphic part renders the transformation of the quadric from its original form to canonical form in Manim. Currently, it doesn't support "complex" quadrics: complex ellipsoid, complex cone, complex elliptic cylinder, complex intersecting planes, complex parallel planes.

#### Functioning

After obtaining everything necessary (the previously mentioned dictionary) from the numerical part, the graphic part works as follows:

1. Manim requires functions in parametric form to operate and does not accept implicit equations. Therefore, starting from the implicit canonical metric equation, we easily obtain the parametric form (see wiki "parametric form" and apply the transformations in reverse to obtain the original quadric. In fact, this allows us to obtain the parametric form of the original quadric without having to apply general parametrization methods, which would be more difficult to implement. This part is not rendered. In certain cases (e.g., a two-sheeted hyperboloid), only one sheet is parametrically generated, and the other is obtained by reflecting the first sheet with respect to the center.

2. The quadric is brought into canonical form by correctly applying the transformations, and the transformation is rendered (see the order of transformations in the numerical part).

#### Modules

- `scene_render.py`: coordinates the various transformations of the quadric, i.e., the scene phases. Sets the axes and the camera position.
- `create_text_overlay.py`: creates the text objects displayed over the quadric transformation and passes them to `scene_render.py`.
- `create_quadric_surface.py`: creates the initial quadric object in canonical form (initially not rendered, see functionality).

#### Testing

Yet to be done.

#### Cose da migliorare

- Conduct some testing on the limits of the graphic part.
- Test resolution based on coefficients for "large" quadrics.
- Test range (of points) "modular"/adaptable to adequately represent quadrics based on coefficients – see also resolution.
