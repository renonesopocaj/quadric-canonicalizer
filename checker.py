import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from misc import clean_near_zero

global x, y, z
x, y, z = sp.symbols('x y z')

"""
This module contains various functions that are used in transformer.py and parabolic_cylinder.py
to check its correctness. obtain_quadric_expr also plots the quadric.
Bypass is set to True to avoid using them. Change it to False to use them while testing.
"""

class WrongCheckException(Exception):
    pass

def obtain_quadric_expr(A_overline, center_vec=(sp.Matrix([[0], [0], [0]])), display_string=None, display_string_two="", bypass=True):
    """
    Write the quadric surface expression/polynomial associated with the input A_overline matrix,
    and plot it.

    Parameters:
    -----------
    A_overline : numpy.ndarray
        The input matrix associated witht the quadric surface.
    center_vec : float, optional (default=null vector)
        If the quadric has a center and is not translated yet, insert the center vector to highlight it in the plot.
    display_string : str, optional (default=None)
        String to display
    """
    if bypass==False:
        center_vec = tuple(center_vec)
        quadric_expr = (((sp.Matrix([[x, y, z, 1]]) * sp.Matrix(A_overline)) * (sp.Matrix([[x, y, z, 1]]).T))[0,0]).expand()
        plot_implicit_3d(quadric_expr, highlight_points=[center_vec], display_string=display_string)
        return quadric_expr

def check_two_forms_centered(A_overline, A_overline_og, P_overline_tot, display_string="", bypass=True):
    if bypass==False:
        quadric_CMF = ((sp.Matrix([[x, y, z, 1]]) * sp.Matrix(A_overline) * sp.Matrix([[x], [y], [z], [1]]))[0, 0]).expand()
        A_overline_check = (np.transpose(P_overline_tot) @ A_overline_og) @ P_overline_tot
        A_overline_check = clean_near_zero(A_overline_check)
        quadric_CMF_viaTransform = ((sp.Matrix([[x, y, z, 1]]) * sp.Matrix(A_overline_check) * sp.Matrix([[x], [y], [z], [1]]))[0, 0]).expand()
        check = np.allclose(A_overline, A_overline_check, rtol=1e-10, atol=1e-10)
        if (check==False):
            raise WrongCheckException("The quadric obtained 'algebrically' and the one obtained with the transformation "
                                      "done via matrix product are not the same")
        return check

def check_two_forms_acentered(A_overline, A_overline_og, S_norm, transl_vector, display_string="", bypass=True):
    if bypass==False:
        quadric_CMF = ((sp.Matrix([[x, y, z, 1]]) * sp.Matrix(A_overline) * sp.Matrix([[x], [y], [z], [1]]))[0, 0]).expand()
        P_overline_tot = sp.BlockMatrix([[sp.BlockMatrix([sp.Matrix(S_norm), sp.Matrix([[0],[0],[0]])]).as_explicit()], [sp.Matrix([[0, 0, 0, 1]])]]).as_explicit()
        P_overline_tot = np.array(P_overline_tot, dtype=np.float64)
        A_overline_check = (np.transpose(P_overline_tot) @ A_overline_og) @ P_overline_tot
        P_overline_tot = sp.BlockMatrix([[sp.BlockMatrix([sp.eye(3,3), sp.Matrix(transl_vector)]).as_explicit()], [sp.Matrix([[0, 0, 0, 1]])]]).as_explicit()
        P_overline_tot = np.array(P_overline_tot, dtype=np.float64)
        A_overline_check = (np.transpose(P_overline_tot) @ A_overline_check) @ P_overline_tot
        A_overline_check = clean_near_zero(A_overline_check)
        quadric_CMF_viaTransform = ((sp.Matrix([[x, y, z, 1]]) * sp.Matrix(A_overline_check) * sp.Matrix([[x], [y], [z], [1]]))[0, 0]).expand()
        check = np.allclose(A_overline, A_overline_check, rtol=1e-10, atol=1e-10)
        if (check==False):
            raise WrongCheckException("The quadric obtained 'algebrically' and the one obtained with the transformation done"
                                      " via matrix product are not the same")
        return check

def plot_implicit_3d(expr, ranges=(-20, 20), points=200, threshold=0.2, alpha=0.2, cmap='viridis',
                     highlight_points=None, show_axes=True, bypass=False, display_string=None):
    """
    Creates a 3D visualization of an implicit surface defined by a symbolic expression.
    The function plots points where the expression evaluates close to zero, effectively
    visualizing the surface in 3D space.

    Parameters:
    -----------
    expr : sympy.Expression
        Symbolic expression defining the implicit surface.
        Should be an equation in variables x, y, and z.
    ranges : tuple or float, optional (default=(-20, 20))
        If tuple: (min_val, max_val) for x, y, and z axes
        If float/int: Creates symmetric range (-value, value)
    points : int, optional (default=200)
        Number of points along each axis for the grid
    threshold : float, optional (default=0.2)
        Maximum absolute value for considering a point part of the surface
    alpha : float, optional (default=0.2)
        Transparency of the surface points (0.0 to 1.0)
    cmap : str, optional (default='viridis')
        Matplotlib colormap name for surface coloring
    highlight_points : list of tuples, optional (default=None)
        List of 3D points to highlight on the surface
        Each point should be a tuple (x, y, z)
    show_axes : bool, optional (default=True)
        Whether to display coordinate axes
    bypass : bool, optional (default=True)
        If True, skips the plotting (useful for avoiding the slow-down due to the rendering, when testing)
    display_string : str, optional (default=None)
        Custom title for the plot. If None, uses 'Superficie Implicita 3D'

    Returns:
    --------
    None
        Displays the plot using matplotlib

    Notes:
    ------
    The function follows these steps:
    1. Creates a 3D grid of points
    2. Evaluates the expression at each point
    3. Selects points where the expression is close to zero (within threshold)
    4. Creates a scatter plot of these points
    5. Adds coordinate axes and highlighted points if requested
    6. Adds a colorbar showing the expression values

    Notes:
    ------
    - Large numbers of points can make the visualization slow
    - Small threshold values give more precise but potentially sparse surfaces
    - Large threshold values give denser but less precise surfaces
    """
    if bypass == False:
        if isinstance(ranges, (int, float)): # Gestione ranges
            ranges = (-ranges, ranges)
        x_vals = np.linspace(ranges[0], ranges[1], points)
        y_vals = np.linspace(ranges[0], ranges[1], points)
        z_vals = np.linspace(ranges[0], ranges[1], points)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
        f = sp.lambdify((x, y, z), expr) # Converte l'espressione SymPy in una funzione numerica
        vol = f(X, Y, Z)
        mask = np.abs(vol) < threshold # Trova i punti vicini alla superficie
        x_surface = X[mask] # Estrai i punti della superficie
        y_surface = Y[mask]
        z_surface = Z[mask]
        values = vol[mask]
        if len(values) == 0: # Verifica se abbiamo trovato punti
            print(f"Nessun punto trovato nell'intervallo {ranges} con threshold {threshold}.")
            return
        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max()) # Normalizza i valori per il colore
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        elev = 10
        azim = -60
        ax.view_init(elev=elev, azim=azim)
        camera_dist = 3
        if camera_dist is not None:
            ax.dist = camera_dist
        scatter = ax.scatter(x_surface, y_surface, z_surface,
                             c=values, cmap=plt.get_cmap(cmap),
                             alpha=alpha, norm=norm, label='Superficie')
        if show_axes: # Mostra gli assi x,y,z
            # Crea linee per gli assi
            ax.plot([-ranges[1], ranges[1]], [0, 0], [0, 0], 'k-', lw=1, alpha=0.5, label='Asse x')
            ax.plot([0, 0], [-ranges[1], ranges[1]], [0, 0], 'k-', lw=1, alpha=0.5, label='Asse y')
            ax.plot([0, 0], [0, 0], [-ranges[1], ranges[1]], 'k-', lw=1, alpha=0.5, label='Asse z')
            # Aggiungi frecce alla fine degli assi
            arrow_length = (ranges[1] - ranges[0]) * 0.05
            ax.quiver(ranges[1], 0, 0, arrow_length, 0, 0, color='k', alpha=0.5)
            ax.quiver(0, ranges[1], 0, 0, arrow_length, 0, color='k', alpha=0.5)
            ax.quiver(0, 0, ranges[1], 0, 0, arrow_length, color='k', alpha=0.5)
            # Aggiungi etichette agli assi
            ax.text(ranges[1] + arrow_length, 0, 0, 'X', fontsize=12)
            ax.text(0, ranges[1] + arrow_length, 0, 'Y', fontsize=12)
            ax.text(0, 0, ranges[1] + arrow_length, 'Z', fontsize=12)
        # Aggiungi punti evidenziati
        if highlight_points is not None:
            for i, point in enumerate(highlight_points):
                ax.scatter(*point, color='red', s=100, alpha=1.0,
                           label=f'Punto {i + 1} {point}')
                # Aggiungi linee tratteggiate dagli assi al punto
                ax.plot([point[0], point[0]], [point[1], point[1]], [0, point[2]], 'r--', alpha=0.3)
                ax.plot([point[0], point[0]], [0, point[1]], [0, 0], 'r--', alpha=0.3)
                ax.plot([0, point[0]], [0, 0], [0, 0], 'r--', alpha=0.3)
        # Imposta gli assi
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        title = display_string if display_string is not None else 'Superficie Implicita 3D'
        ax.set_title(title)
        ax.set_xlim(ranges) # Limita la visualizzazione all'intervallo specificato
        ax.set_ylim(ranges)
        ax.set_zlim(ranges)
        ax.legend() # Aggiungi legenda
        fig.colorbar(scatter) # Aggiungi una barra dei colori
        plt.show()
