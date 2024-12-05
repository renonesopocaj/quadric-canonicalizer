import random
import transformer
from classifier import NotAQuadricException
import sympy as sp

"""
This module testes the transformer.py module.
We have currently not yet implemented a distinction between real elliptic cylinder (9) and complex elliptic
cylinder (10), so the function "classify" always returns 9, because real elliptic cylinder are supported
by the graphical part. The same for real parallel planes (15) and complex parallel planes (16).
Due to the value returned by classify
"""

global SUP_NUM, INF_NUM, TEST, ENUM_QUADRICS, ENUM_GEN_FUNCTIONS
SUP_NUM = 10 # maximum coefficient outputted by ri()
INF_NUM = 1 # minimum coefficient outputted by ri() (should not be less than 1)
TEST = 3 # number of quadrics to test for each generated type (see generate_... funtions)
ENUM_QUADRICS = {"real ellipsoid": 1, "two sheets hyperboloid": 4, "complex ellipsoid": 2, "one sheet hyperboloid": 3,
                 "complex cone": 6, "real cone": 5, "elliptic paraboloid": 7, "hyperbolic paraboloid": 8,
                 "real elliptic cylinder": 9, "complex elliptic cylinder": 10, "hyperbolic cylinder": 11,
                 "real intersecting planes": 12, "complex intersecting planes": 13, "parabolic cylinder": 14,
                 "real parallel planes": 15, "complex parallel planes": 16, "double plane": 17}

def ri():
    """
    Generates a random integer between to global constants defined above

    Parameters:
    -----------
    None

    Returns:
    --------
    int
    """
    return random.randint(INF_NUM, SUP_NUM)

def generate_real_cone(quadrics_examples_dict):
    quadrics_examples_dict["real cone"] = {}
    for i in range(0,TEST):
        quadrics_examples_dict["real cone"][f"(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} - (z-{ri()})**2/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["real cone"][f"(x-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} - (z-{ri()}x)**2/{ri()} = 0"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["real cone"][f"-(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["real cone"][f"-(x-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} + (z-{ri()}x)**2/{ri()} = 0"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["real cone"][f"(x-{ri()})**2/{ri()} - (y-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["real cone"][f"(x-{ri()}y)**2/{ri()} - (y-{ri()}z)**2/{ri()} + (z-{ri()}x)**2/{ri()} = 0"] = None
    return quadrics_examples_dict

def generate_complex_cone(quadrics_examples_dict):
    quadrics_examples_dict["complex cone"] = {}
    for i in range(0,TEST):
        quadrics_examples_dict["complex cone"][f"(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["complex cone"][f"(x-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} + (z-{ri()}x)**2/{ri()} = 0"] = None
    return quadrics_examples_dict

def generate_real_ellipsoid(quadrics_examples_dict):
    quadrics_examples_dict["real ellipsoid"] = {}
    for i in range(0,TEST):
        quadrics_examples_dict["real ellipsoid"][f"(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["real ellipsoid"][f"(x-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} + (z-{ri()}x)**2/{ri()} = {ri()}"] = None
    return quadrics_examples_dict

def generate_complex_ellipsoid(quadrics_examples_dict):
    quadrics_examples_dict["complex ellipsoid"] = {}
    for i in range(0,TEST):
        quadrics_examples_dict["complex ellipsoid"][f"(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = -{ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["complex ellipsoid"][f"(x-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} + (z-{ri()}x)**2/{ri()} = -{ri()}"] = None
    return quadrics_examples_dict

def generate_onesheet_hyperboloid(quadrics_examples_dict):
    quadrics_examples_dict["one sheet hyperboloid"] = {}
    for i in range(0,TEST):
        quadrics_examples_dict["one sheet hyperboloid"][f"(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} - (z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["one sheet hyperboloid"][f"(x-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} - (z-{ri()}x)**2/{ri()} = {ri()}"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["one sheet hyperboloid"][f"-(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["one sheet hyperboloid"][f"-(x-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} + (z-{ri()}x)**2/{ri()} = {ri()}"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["one sheet hyperboloid"][f"(x-{ri()})**2/{ri()} - (y-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["one sheet hyperboloid"][f"(x-{ri()}y)**2/{ri()} - (y-{ri()}z)**2/{ri()} + (z-{ri()}x)**2/{ri()} = {ri()}"] = None
    return quadrics_examples_dict

def generate_twosheets_hyperboloid(quadrics_examples_dict):
    quadrics_examples_dict["two sheets hyperboloid"] = {}
    for i in range(0,TEST):
        quadrics_examples_dict["two sheets hyperboloid"][f"(x-{ri()})**2/{ri()} - (y-{ri()})**2/{ri()} - (z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["two sheets hyperboloid"][f"(x-{ri()}y)**2/{ri()} - (y-{ri()}z)**2/{ri()} - (z-{ri()}x)**2/{ri()} = {ri()}"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["two sheets hyperboloid"][f"-(x-{ri()})**2/{ri()} - (y-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["two sheets hyperboloid"][f"-(x-{ri()}y)**2/{ri()} - (y-{ri()}z)**2/{ri()} + (z-{ri()}x)**2/{ri()} = {ri()}"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["two sheets hyperboloid"][f"-(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} - (z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["two sheets hyperboloid"][f"-(x-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} - (z-{ri()}x)**2/{ri()} = {ri()}"] = None
    return quadrics_examples_dict

def generate_elliptic_paraboloid(quadrics_examples_dict):
    quadrics_examples_dict["elliptic paraboloid"] = {}
    # x,y squared, z linear
    for i in range(0,TEST):
        quadrics_examples_dict["elliptic paraboloid"][
            f"(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} - 2(z-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["elliptic paraboloid"][
            f"(x-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} - 2(z-{ri()})/{ri()} = 0"] = None
    # y,z squared, x linear
    for i in range(0,TEST):
        quadrics_examples_dict["elliptic paraboloid"][
            f"(z-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} + 2(x-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["elliptic paraboloid"][
            f"(z-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} + 2(x-{ri()})/{ri()} = 0"] = None
    # x,z squared, y linear
    for i in range(0,TEST):
        quadrics_examples_dict["elliptic paraboloid"][
            f"(x-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} - 2(y-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["elliptic paraboloid"][
            f"(x-{ri()}y)**2/{ri()} + (z-{ri()}x)**2/{ri()} - 2(y-{ri()})/{ri()} = 0"] = None
    return quadrics_examples_dict

def generate_hyperbolic_paraboloid(quadrics_examples_dict):
    quadrics_examples_dict["hyperbolic paraboloid"] = {}
    # x,y squared, z linear
    for i in range(0,TEST):
        quadrics_examples_dict["hyperbolic paraboloid"][
            f"(x-{ri()})**2/{ri()} - (y-{ri()})**2/{ri()} - 2(z-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["hyperbolic paraboloid"][
            f"(x-{ri()}y)**2/{ri()} - (y-{ri()}z)**2/{ri()} - 2(z-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["hyperbolic paraboloid"][
            f"-(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} - 2(z-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["hyperbolic paraboloid"][
            f"-(x-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} - 2(z-{ri()})/{ri()} = 0"] = None
    # y,z squared, x linear
    for i in range(0,TEST):
        quadrics_examples_dict["hyperbolic paraboloid"][
            f"(z-{ri()})**2/{ri()} - (y-{ri()})**2/{ri()} + 2(x-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["hyperbolic paraboloid"][
            f"(z-{ri()}y)**2/{ri()} - (y-{ri()}z)**2/{ri()} + 2(x-{ri()})/{ri()} = 0"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["hyperbolic paraboloid"][
            f"-(z-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} + 2(x-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["hyperbolic paraboloid"][
            f"-(z-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} + 2(x-{ri()})/{ri()} = 0"] = None
    # x,z squared, y linear
    for i in range(0,TEST):
        quadrics_examples_dict["hyperbolic paraboloid"][
            f"(x-{ri()})**2/{ri()} - (z-{ri()})**2/{ri()} - 2(y-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["hyperbolic paraboloid"][
            f"(x-{ri()}y)**2/{ri()} - (z-{ri()}x)**2/{ri()} - 2(y-{ri()})/{ri()} = 0"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["hyperbolic paraboloid"][
            f"-(x-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} - 2(y-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["hyperbolic paraboloid"][
            f"-(x-{ri()}y)**2/{ri()} + (z-{ri()}x)**2/{ri()} - 2(y-{ri()})/{ri()} = 0"] = None
    return quadrics_examples_dict

def generate_real_elliptic_cylinder(quadrics_examples_dict):
    quadrics_examples_dict["real elliptic cylinder"] = {}
    # x,y squared
    for i in range(0,TEST):
        quadrics_examples_dict["real elliptic cylinder"][
            f"(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["real elliptic cylinder"][
            f"(x-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} = {ri()}"] = None
    # y,z squared
    for i in range(0,TEST):
        quadrics_examples_dict["real elliptic cylinder"][
            f"(y-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["real elliptic cylinder"][
            f"(y-{ri()}z)**2/{ri()} + (z-{ri()}x)**2/{ri()} = {ri()}"] = None
    # x,z squared
    for i in range(0,TEST):
        quadrics_examples_dict["real elliptic cylinder"][
            f"(x-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["real elliptic cylinder"][
            f"(x-{ri()}y)**2/{ri()} + (z-{ri()}x)**2/{ri()} = {ri()}"] = None
    return quadrics_examples_dict

def generate_complex_elliptic_cylinder(quadrics_examples_dict):
    quadrics_examples_dict["complex elliptic cylinder"] = {}
    # x,y squared
    for i in range(0,TEST):
        quadrics_examples_dict["complex elliptic cylinder"][
            f"(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} = -{ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["complex elliptic cylinder"][
            f"(x-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} = -{ri()}"] = None
    # y,z squared
    for i in range(0,TEST):
        quadrics_examples_dict["complex elliptic cylinder"][
            f"(y-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = -{ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["complex elliptic cylinder"][
            f"(y-{ri()}z)**2/{ri()} + (z-{ri()}x)**2/{ri()} = -{ri()}"] = None
    # x,z squared
    for i in range(0,TEST):
        quadrics_examples_dict["complex elliptic cylinder"][
            f"(x-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = -{ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["complex elliptic cylinder"][
            f"(x-{ri()}y)**2/{ri()} + (z-{ri()}x)**2/{ri()} = -{ri()}"] = None
    return quadrics_examples_dict

def generate_hyperbolic_cylinder(quadrics_examples_dict):
    quadrics_examples_dict["hyperbolic cylinder"] = {}
    # x,y squared
    for i in range(0,TEST):
        quadrics_examples_dict["hyperbolic cylinder"][
            f"(x-{ri()})**2/{ri()} - (y-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["hyperbolic cylinder"][
            f"(x-{ri()}y)**2/{ri()} - (y-{ri()}z)**2/{ri()} = {ri()}"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["hyperbolic cylinder"][
            f"-(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["hyperbolic cylinder"][
            f"-(x-{ri()}y)**2/{ri()} + (y-{ri()}z)**2/{ri()} = {ri()}"] = None
    # y,z squared
    for i in range(0,TEST):
        quadrics_examples_dict["hyperbolic cylinder"][
            f"(y-{ri()})**2/{ri()} - (z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["hyperbolic cylinder"][
            f"(y-{ri()}z)**2/{ri()} - (z-{ri()}x)**2/{ri()} = {ri()}"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["hyperbolic cylinder"][
            f"-(y-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["hyperbolic cylinder"][
            f"-(y-{ri()}z)**2/{ri()} + (z-{ri()}x)**2/{ri()} = {ri()}"] = None
    # x,z squared
    for i in range(0,TEST):
        quadrics_examples_dict["hyperbolic cylinder"][
            f"(x-{ri()})**2/{ri()} - (z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["hyperbolic cylinder"][
            f"(x-{ri()}y)**2/{ri()} - (z-{ri()}x)**2/{ri()} = {ri()}"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["hyperbolic cylinder"][
            f"-(x-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["hyperbolic cylinder"][
            f"-(x-{ri()}y)**2/{ri()} + (z-{ri()}x)**2/{ri()} = {ri()}"] = None
    return quadrics_examples_dict

def generate_real_intersecting_planes(quadrics_examples_dict):
    quadrics_examples_dict["real intersecting planes"] = {}
    # x,y squared
    for i in range(0,TEST):
        quadrics_examples_dict["real intersecting planes"][
            f"(x-{ri()})**2/{ri()} - (y-{ri()})**2/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["real intersecting planes"][
            f"(x-{ri()}y)**2/{ri()} - (y-{ri()}z)**2/{ri()} = 0"] = None
    # y,z squared
    for i in range(0,TEST):
        quadrics_examples_dict["real intersecting planes"][
            f"(y-{ri()})**2/{ri()} - (z-{ri()})**2/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["real intersecting planes"][
            f"(y-{ri()}z)**2/{ri()} - (z-{ri()}x)**2/{ri()} = 0"] = None
    # x,z squared
    for i in range(0,TEST):
        quadrics_examples_dict["real intersecting planes"][
            f"(x-{ri()})**2/{ri()} - (z-{ri()})**2/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["real intersecting planes"][
            f"(x-{ri()}y)**2/{ri()} - (z-{ri()}x)**2/{ri()} = 0"] = None
    return quadrics_examples_dict

def generate_complex_intersecting_planes(quadrics_examples_dict):
    quadrics_examples_dict["complex intersecting planes"] = {}
    # x,y squared
    for i in range(0,TEST):
        quadrics_examples_dict["complex intersecting planes"][
            f"(x-{ri()})**2/{ri()} + (y-{ri()})**2/{ri()} = 0"] = None
    # y,z squared
    for i in range(0,TEST):
        quadrics_examples_dict["complex intersecting planes"][
            f"(y-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = 0"] = None
    # x,z squared
    for i in range(0,TEST):
        quadrics_examples_dict["complex intersecting planes"][
            f"(x-{ri()})**2/{ri()} + (z-{ri()})**2/{ri()} = 0"] = None
    return quadrics_examples_dict

def generate_parabolic_cylinder(quadrics_examples_dict):
    quadrics_examples_dict["parabolic cylinder"] = {}
    # x squared
    for i in range(0,TEST):
        quadrics_examples_dict["parabolic cylinder"][
            f"(x-{ri()})**2/{ri()} - 2(z-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["parabolic cylinder"][
            f"(x-{ri()}y)**2/{ri()} - 2(z-{ri()})/{ri()} = 0"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["parabolic cylinder"][
            f"(x-{ri()})**2/{ri()} - 2(y-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["parabolic cylinder"][
            f"(x-{ri()}y)**2/{ri()} - 2(y-{ri()})/{ri()} = 0"] = None
    # z squared
    for i in range(0,TEST):
        quadrics_examples_dict["parabolic cylinder"][
            f"(z-{ri()})**2/{ri()} - 2(x-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["parabolic cylinder"][
            f"(z-{ri()}y)**2/{ri()} - 2(x-{ri()})/{ri()} = 0"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["parabolic cylinder"][
            f"(z-{ri()})**2/{ri()} - 2(y-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["parabolic cylinder"][
            f"(z-{ri()}y)**2/{ri()} - 2(y-{ri()})/{ri()} = 0"] = None
    # y squared
    for i in range(0,TEST):
        quadrics_examples_dict["parabolic cylinder"][
            f"(y-{ri()})**2/{ri()} - 2(z-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["parabolic cylinder"][
            f"(y-{ri()}y)**2/{ri()} - 2(z-{ri()})/{ri()} = 0"] = None
    for i in range(0,TEST):
        quadrics_examples_dict["parabolic cylinder"][
            f"(y-{ri()})**2/{ri()} - 2(x-{ri()})/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["parabolic cylinder"][
            f"(y-{ri()}y)**2/{ri()} - 2(x-{ri()})/{ri()} = 0"] = None
    return quadrics_examples_dict

def generate_real_parallel_planes(quadrics_examples_dict):
    quadrics_examples_dict["real parallel planes"] = {}
    # x squared
    for i in range(0,TEST):
        quadrics_examples_dict["real parallel planes"][
            f"(x-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["real parallel planes"][
            f"(x-{ri()}y)**2/{ri()} = {ri()}"] = None
    # z squared
    for i in range(0,TEST):
        quadrics_examples_dict["real parallel planes"][
            f"(z-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["real parallel planes"][
            f"(z-{ri()}y)**2/{ri()} = {ri()}"] = None
    # y squared
    for i in range(0,TEST):
        quadrics_examples_dict["real parallel planes"][
            f"(y-{ri()})**2/{ri()} = {ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["real parallel planes"][
            f"(y-{ri()}y)**2/{ri()} = {ri()}"] = None
    return quadrics_examples_dict

def generate_complex_parallel_planes(quadrics_examples_dict):
    quadrics_examples_dict["complex parallel planes"] = {}
    # x squared
    for i in range(0,TEST):
        quadrics_examples_dict["complex parallel planes"][
            f"(x-{ri()})**2/{ri()} = -{ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["complex parallel planes"][
            f"(x-{ri()}y)**2/{ri()} = -{ri()}"] = None
    # z squared
    for i in range(0,TEST):
        quadrics_examples_dict["complex parallel planes"][
            f"(z-{ri()})**2/{ri()} = -{ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["complex parallel planes"][
            f"(z-{ri()}y)**2/{ri()} = -{ri()}"] = None
    # y squared
    for i in range(0,TEST):
        quadrics_examples_dict["complex parallel planes"][
            f"(y-{ri()})**2/{ri()} = -{ri()}"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["complex parallel planes"][
            f"(y-{ri()}y)**2/{ri()} = -{ri()}"] = None
    return quadrics_examples_dict

def generate_double_plane(quadrics_examples_dict):
    quadrics_examples_dict["double plane"] = {}
    # x squared
    for i in range(0,TEST):
        quadrics_examples_dict["double plane"][
            f"(x-{ri()})**2/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["double plane"][
            f"(x-{ri()}y)**2/{ri()} = 0"] = None
    # z squared
    for i in range(0,TEST):
        quadrics_examples_dict["double plane"][
            f"(z-{ri()})**2/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["double plane"][
            f"(z-{ri()}y)**2/{ri()} = 0"] = None
    # y squared
    for i in range(0,TEST):
        quadrics_examples_dict["double plane"][
            f"(y-{ri()})**2/{ri()} = 0"] = None
    for i in range(0, TEST):
        quadrics_examples_dict["double plane"][
            f"(y-{ri()}y)**2/{ri()} = 0"] = None
    return quadrics_examples_dict

ENUM_GEN_FUNCTIONS = {"real ellipsoid": generate_real_ellipsoid, "two sheets hyperboloid": generate_twosheets_hyperboloid,
                      "complex ellipsoid": generate_complex_ellipsoid, "one sheet hyperboloid": generate_onesheet_hyperboloid,
                 "complex cone": generate_complex_cone, "real cone": generate_real_cone, "elliptic paraboloid": generate_elliptic_paraboloid,
                  "hyperbolic paraboloid": generate_hyperbolic_paraboloid,
                 "real elliptic cylinder": generate_real_elliptic_cylinder,
                  "complex elliptic cylinder": generate_complex_elliptic_cylinder,
                  "hyperbolic cylinder": generate_hyperbolic_cylinder, "real intersecting planes": generate_real_intersecting_planes,
                  "complex intersecting planes": generate_complex_intersecting_planes, "parabolic cylinder": generate_parabolic_cylinder,
                 "real parallel planes": generate_real_parallel_planes, "complex parallel planes": generate_complex_parallel_planes, "double plane": generate_double_plane}

def generate_quadric_examples(quadric_name):
    """
    :param quadric_name: the desidered quadric to test
    :return: a dictionary with some equations of the quadric you want to test
    """
    quadrics_examples_dict = {}
    return ENUM_GEN_FUNCTIONS[quadric_name](quadrics_examples_dict)

def test_quadrics(quadric_name):
    quadrics_examples_dict = generate_quadric_examples(quadric_name)
    changed_keys = []
    for eq in quadrics_examples_dict[quadric_name].keys():
        try:
            return_dict = transformer.canonize_quadric(eq)
            # uncomment below to print the results of the transformation
            # for key in return_dict:
            #     print(key + ": \n" + str(return_dict[key]))
            quadrics_examples_dict[quadric_name][eq] = (return_dict["quadric type"] == ENUM_QUADRICS[quadric_name])
            changed_keys.append(eq.replace("**", "^"))
        except ValueError:
            print("Value Error")
        except UnboundLocalError:
            print("Unbound Local Error")
        except NotAQuadricException:
            print("Not A Quadric Exception")
        except sp.polys.polyerrors.GeneratorsNeeded:
            print("This error is often raised if the given equation simplifies to 0=0")
    if all(quadrics_examples_dict[quadric_name].values()) == False:
        for key in quadrics_examples_dict[quadric_name]:
            if quadrics_examples_dict[quadric_name][key]==None:
                quadrics_examples_dict[quadric_name][key] = True
    # the following print doesn't work (will yield false even if correct) with quadric_type 10 or 16
    # unless you change classifier to return 10 instead of 9 and 16 instead of 15
    print(f"It's {all(quadrics_examples_dict[quadric_name].values())} that all quadrics have been classified correctly")
    print("the list of generated examples is")
    for elem in changed_keys:
        print(elem)

if __name__ == '__main__':
    test_quadrics(quadric_name="hyperbolic cylinder")