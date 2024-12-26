# quadric-canonicalizer
<a id="readme-top"></a>
A program that brings a quadric surface in metric canonical form and renders the transformation. It can return/display the implicit (and, soon, also the parametric) equations of the quadric, before and after each transformation, with also the matrices/vectors that represent the transformations. 

The mathematical foundations of the algorithm were drawn from 3 main sources, see the bibliography.

Main things currently not supported:
- Rendering of complex quadrics: complex ellipsoid, complex cone, complex elliptic cylinder, complex intersecting planes, complex parallel planes.
- When using function `classifier`, distinguishing between complex vs real elliptic cylinder and complex vs real parallel planes is not supported yet, even though `transformer` still works.

## Table of contents

+  [Getting started](#getting-started)
    - [Dependencies](#dependencies)
    - [Installation Guide](#installation-guide)
      * [Step 1 Python Installation](#step-1-python-installation)
      * [Step 2 Install Required Python Packages](#step-2-install-required-python-packages)
      * [Step 3 Install FFMPEG](#step-3-install-ffmpeg)
      * [Step 4 Install LaTeX](#step-4-install-latex)
    - [Troubleshooting](#troubleshooting)
      * [Common Issues](#common-issues)
    - [Getting Help](#getting-help)
+ [Demos](#demos)
+ [Program usage](#program-usage)
     - [Main](#main)
+ [TODOs](#todos)
    - [Numerical part](#numerical-part)
    - [Graphical part](#graphical-part)
+ [Contributors contacts](#contributors-contacts)
+ [Bibliography and acknowledgments](#bibliography-and-acknowledgments)

## Getting started

### Dependencies

- **External dependencies of the numerical part**: `sympy`, `numpy`, `scipy`, `warnings`, `random`, `matplotlib`.
- **External dependencies of the graphic part**: `os`, [Manim (community version)](https://www.manim.community/), `numpy`, `math`. For Manim, as of writing, the required Python version is at least 3.8 and FFMPEG needs to be installed. Additionally, LaTeX must be installed for rendering the equations/matrices displayed.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation Guide

#### Step 1: Python Installation
1. Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. During installation, make sure to check "Add Python to PATH"
3. Verify installation by opening a terminal/command prompt and running:
   ```bash
   python --version
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Step 2: Install Required Python Packages
1. First, ensure pip (Python package installer) is up to date:
   ```bash
   python -m pip install --upgrade pip
   ```

2. Install numerical dependencies:
   ```bash
   pip install sympy numpy scipy matplotlib
   ```

3. Install Manim Community Edition:
   ```bash
   pip install manim
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Step 3: Install FFMPEG
##### Windows
1. Download FFMPEG from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract the downloaded file
3. Add the `bin` folder to your system's PATH environment variable
4. Verify installation:
   ```bash
   ffmpeg -version
   ```

##### macOS
Using Homebrew:
```bash
brew install ffmpeg
```

##### Linux Ubuntu Debian
```bash
sudo apt update
sudo apt install ffmpeg
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Step 4: Install LaTeX

##### Windows
1. Download and install MiKTeX from [miktex.org](https://miktex.org/download)
2. During first use, allow it to install required packages automatically

##### macOS
1. Download and install MacTeX from [tug.org/mactex](https://tug.org/mactex/)

##### Linux Ubuntu Debian
```bash
sudo apt update
sudo apt install texlive-full
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Troubleshooting

#### Common Issues

1. **Python Path Issues**
   - Ensure Python is added to PATH
   - Try restarting your terminal/command prompt after installation

2. **FFMPEG Not Found**
   - Verify FFMPEG is in your system PATH
   - Try reinstalling FFMPEG

3. **LaTeX Rendering Issues**
   - Ensure LaTeX is properly installed
   - Check if required LaTeX packages are installed
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Getting Help
If you encounter any issues:
1. Check the error message carefully
2. Search for the error in the project's issues on GitHub
3. Create a new issue if the problem persists

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Demos
<p float="left">
    <img src="https://github.com/user-attachments/assets/6f78afe6-b810-4d37-aa97-7c10984fbc61" width="500" />
    <img src="https://github.com/user-attachments/assets/6ab3fdec-65b1-4f20-b1cf-c50d154f8056" width="500" />
    <img src="https://github.com/user-attachments/assets/d3f9c9e8-27ba-400e-a322-7c635d5ac8fc" width="500" />
    <img src="https://github.com/user-attachments/assets/bcc8355c-4c29-4fc1-937c-c5e196bd55f5" width="500" />
    <img src="https://github.com/user-attachments/assets/0ab68437-8bd2-41b2-8c77-4fb6b1d56191" width="500" />
    <img src="https://github.com/user-attachments/assets/2299b124-8aa0-4b5d-bc71-6321200b4db2" width="500" />
    <img src="https://github.com/user-attachments/assets/f15a5a03-c3dc-46c2-a0b7-eb052cb70211" width="500" />
    <img src="https://github.com/user-attachments/assets/8c61c786-ee76-475d-ba6b-9e54ec62acd7" width="500" />
    <img src="https://github.com/user-attachments/assets/92221029-2613-402b-bfc2-b0dd1c93eed1" width="500" />
    <img src="https://github.com/user-attachments/assets/c42e9022-4d9a-499f-93e8-2900b4b05d60" width="500" />
    <img src="https://github.com/user-attachments/assets/dc8f58ec-faf6-46a5-9912-ff194a895fea" width="500" />
    <img src="https://github.com/user-attachments/assets/30091613-90ef-47ba-a981-c228026ccc76" width="500" />
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Program usage

Let's divide the program into "main", "numerical part", and "graphic part". The details about the functioning of the numerical and graphical part are in the wiki: here we only present the main and how to use it.

### Main

The main receives as input the equation of the quadric as a string, the filepath (path where to place video files rendered by Manim), and the desired output quality. It first executes the numerical part (`canonize_quadric` in `transformer.py`) to bring the quadric into canonical form and obtain all the necessary information for the graphic part, and then renders the transformation (`scene_render.py`).

The input string equation must respect the following rules:
- use x,y,z as variables;
- use rational or decimal numbers as coefficients of the monomials;
- it must be a polynomial of degree two;
- it must contain the `=` sign, but it doesn't necessarily have to be written as a sum of monomials or with the right side of the equation `=0`;
- powers are written as base `**` exponent, multiplication as factor1`*`factor2, sum, difference, fractions as usual with `+`,`-`,`/`.

For details about the modules of the numerical and graphical part, see the wiki.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## TODOs

#### Numerical part
- Currently, there is no implementation to distinguish real elliptic cylinders from complex ones and real parallel planes from complex ones.
- The parabolic cylinder has cases that are not correctly brought into canonical form. This could be resolved in two ways:
    - Use a method that avoids symbolic resolutions.
    - Ensure better handling of rounding errors due to floating point arithmetic.
- Implement a way to check errors by examining the coefficients of the terms.
- Implement a robust method for error tolerance in floating point arithmetic.
- Ensure all matrices have $\det S=1$, i.e., permute those with $\det S=-1$ (and the corresponding rows in $D$).
- Optional, only if generalizing to hyperquadrics is needed: use permutation matrices that reduce a generic quadric to a specific permutation of indeterminates, instead of the various if-else statements for each possible permutation.

#### Graphical part

- Render complex quadrics that are not currently supported
- Conduct some testing on the limits of the graphic part.
- Test resolution based on coefficients for "large" quadrics.
- Test range (of points) "modular"/adaptable to adequately represent quadrics based on coefficients – see also resolution.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributors contacts
- jacopo.senoner@mail.polimi.it
- alessandro.tinaoui@mail.polimi.it
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Bibliography and acknowledgments
- Professor Maurizio Citterio's notes, Politecnico di Milano (in particular: the method to bring all quadrics into canonical metric form, except the parabolic cylinder).
- Professor Luca Mauri's notes, Politecnico di Milano (in particular: the method to bring the parabolic cylinder into canonical metric form).
- Agustí Reventós Tarrida "Affine Maps, Euclidean Motions and Quadrics" (in particular for the classification of quadrics using orthogonal invariants).
<p align="right">(<a href="#readme-top">back to top</a>)</p>
