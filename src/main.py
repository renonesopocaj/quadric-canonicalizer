import os
import manim as mn
from src.graphics.scene_render import SceneRender
from src.numerical.transformer import canonize_quadric


def graphic_wrapper_function(q_dict, video_quality="1", output_path="./media"):
    """
   Creates and renders a quadric surface video with specified settings.

   Args:
       q_dict (dict): Dictionary containing quadric parameters and type
       video_quality (str): Quality setting ("1"-"4", default="1")
       output_path (str): Output directory path (default="./media")

   Quality Options:
       "1": low_quality
       "2": medium_quality
       "3": high_quality
       "4": production_quality

   The function:
       - Creates/validates output directory
       - Sets video quality
       - Maps quadric type to filename
       - Configures Manim settings
       - Renders scene
   """
    if output_path == "":
        output_path = "../media"

    #check the existence of the path
    if os.path.exists(output_path):
        mn.config.media_dir = output_path
    else:
        os.makedirs(output_path)
        mn.config.media_dir = output_path

    # set quality based on user input
    if video_quality == "1":
        render_quality = "low_quality"
    elif video_quality == "2":
        render_quality = "medium_quality"
    elif video_quality == "3":
        render_quality = "high_quality"
    elif video_quality == "4":
        render_quality = "production_quality"
    else:
        render_quality = "low_quality"
        print("Invalid quality selection. Using default (low_quality)")


    quadrics_name = {
        1: "real_ellipsoid",
        2: "complex_ellipsoid",
        3: "hyperbolic_hyperboloid",
        4: "elliptic_hyperboloid",
        5: "real_cone",
        6: "complex_cone",
        7: "elliptic_paraboloid",
        8: "hyperbolic_paraboloid",
        9: "real_elliptic_cylinder",
        10: "complex_elliptic_cylinder",
        11: "hyperbolic_cylinder",
        12: "real_intersecting_planes",
        13: "complex_intersecting_planes",
        14: "parabolic_cylinder",
        15: "real_parallel_planes",
        16: "complex_parallel_planes",
        17: "double_plane"
    }
    name=quadrics_name[q_dict["quadric type"]]

    mn.config.output_file = f"{name}.mp4"
    mn.config.quality = render_quality

    scene = SceneRender(q_dict)
    scene.render()

def select_example():
    examples_dictionary = {"1": "2x**2 + 2*y**2 + 4z**2 - 2xy + 2x = 0",
                           "3": "x**2 + y**2 + z**2 +2xy - 2xz + 2yz + 4x + 4y - 4z + 2 = 0",
                           "4": "x**2 + y**2 - 3z**2 - 2xy - 6xz - 6yz + 2x + 2y + 4z = 0",
                           "5": "y**2 - 6xz - 6x + 2y - 6z - 5 = 0",
                           "7": "x**2 + y**2 +2z**2 + 2xy - 4x = 0",
                           "8": "12xz + 16yz - 10x = 0",
                           "9": "3x**2 + 2y**2 + 4z**2 - 4xy + 4xz + 6x + 12z + 3 = 0",
                           "11": "x**2-4x-y**2+6y-4=0",
                           "12": "x**2 - 2y**2 - 2z**2 - xy - xz + 5yz + 2x - y - z + 1 = 0",
                           "14": "x**2 + y**2 - 2xy - 4x - 4y - 4z + 4 = 0",
                           "15": "x**2 + y**2 + z**2 - 2xy + 2xz - 2yz + 6x - 6y + 6z + 8 = 0",
                           "17": "x**2 + y**2 + z**2 + 2xy + 2xz + 2yz + 2x + 2y + 2z + 1 = 0",
                           }

    selected = input("type the number corresponding to your chosen quadric, according to the following list:" +
                     """
                     1: real ellipsoid,
                     2: complex ellipsoid - EXAMPLE CURRENTLY NOT AVAILABLE,
                     3: hyperbolic hyperboloid,
                     4: elliptic hyperboloid,
                     5: real cone,
                     6: complex cone - EXAMPLE CURRENTLY NOT AVAILABLE,
                     7: elliptic paraboloid,
                     8: hyperbolic paraboloid,
                     9: real elliptic cylinder,
                     10: complex elliptic cylinder - EXAMPLE CURRENTLY NOT AVAILABLE,
                     11: hyperbolic cylinder,
                     12: real intersecting planes,
                     13: complex intersecting planes - EXAMPLE CURRENTLY NOT AVAILABLE,
                     14: parabolic cylinder,
                     15: real parallel planes,
                     16: complex parallel planes - EXAMPLE CURRENTLY NOT AVAILABLE,
                     17: double plane
                     """)
    while True:
        if (selected in ["2","6","10","13","16"]):
            print("example currently not available, please select another one")
            selected = input("type the number corresponding to your chosen quadric, according to the previous list")
            continue
        else:
            quadric_equation = examples_dictionary[selected]
            break
    return quadric_equation

if __name__ == "__main__":
    flag = input("Do you want to use one of our examples of quadric surface? Type y/n:")
    if flag.lower() == "y":
        eq = select_example()
    else:
        #quadric equation input from user
        eq = input("Please insert the equation of the quadric surface. To ensure proper functioning, insert it in the form:\n"+
                   "ax**2 + by**2 + cz**2 + dxy + exz + fyz + gx + hy + iz + j = 0, with x,y,z as variables and the coefficients either as fractions or as decimal numbers.\n"+
                   "See documentation for more options.\n")
    dict = canonize_quadric(eq)

    #get file path from user
    file_path = input("\nInsert file path or press ENTER for default path:\n")

    #get video quality form user
    quality = input("Choose rendering quality (press Enter for default=low_quality):\n" +
                    "1: low_quality\n" +
                    "2: medium_quality\n" +
                    "3: high_quality\n" +
                    "4: production_quality\n")

    graphic_wrapper_function(dict, quality, file_path)