import sys 
import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
import subprocess
import ast
from estimation import *

legend = ["HermiteS","HermiteS_v1","HermiteS_v2","Lagrange","CubicS"]

def parse_to_numpy_array(array_str):
    # Safely evaluate the string to a Python list of lists using ast.literal_eval
    array_list = ast.literal_eval(array_str)
    
    # Convert the list of lists into a numpy array
    array_data = np.array(array_list)
    
    return array_data

def parse_to_dict(input_string):
    """
    Convertit une chaîne représentant un dictionnaire en un dictionnaire Python.

    Args:
        input_string (str): La chaîne représentant le dictionnaire (ex: "{0: [True, 0.5], 1: [False, 0.1], 2: [False, None]}").

    Returns:
        dict: Le dictionnaire Python correspondant.
    """
    try:
        # Utiliser la fonction `eval` pour convertir la chaîne en objet Python en toute sécurité
        # (aucune exécution de code arbitraire n'est permise ici grâce à un environnement restreint).
        parsed_dict = eval(
            input_string,
            {"True": True, "False": False, "None": None},  # Définitions pour les mots-clés
            {}  # Pas d'accès aux objets globaux
        )
        return parsed_dict
    except (SyntaxError, ValueError) as e:
        print(f"Erreur lors de l'analyse de la chaîne : {e}")
        return None

def nchoosek(n, k):
    from math import comb
    return comb(n, k)

def BernsteinPolynom(N, i, t):
    assert i <= N
    return nchoosek(N, i) * t**i * (1 - t)**(N - i)

def get_BezierCoeff(Poly, index, option=0):
    assert option in [0, 1, 2, 4]
    n = Poly.shape[1]
    if index < 0 or index >= n - 1:
        raise IndexError("Index out of bounds")
    match option:
        case 0:  # Cardinal spline classique
            c = functions_used[0][1]
            tang1 = cardinal_splines(Poly, index, c)
            tang2 = cardinal_splines(Poly, index + 1, c)
        case 1:  # Cardinal spline modifiée (v1)
            alpha = functions_used[1][1]
            tang1 = cardinal_splines_v1(Poly, index, alpha)
            tang2 = cardinal_splines_v1(Poly, index + 1, alpha)
        case 2:  # Cardinal spline (v2)
            tang1 = cardinal_splines_v2(Poly, index)
            tang2 = cardinal_splines_v2(Poly, index + 1)
        case 4:
            derivees = derive_cubique(Poly)
            tang1 = derivees[:,index]
            tang2 = derivees[:,index+1]
    
    return np.array([
        Poly[:, index],
        Poly[:, index] + tang1 / 3,
        Poly[:, index + 1] - tang2 / 3,
        Poly[:, index + 1]
    ]).T

def delta_ordre1(bezier_coeff):
    return np.diff(bezier_coeff, axis=1)

def delta_ordre2(bezier_coeff):
    return np.diff(delta_ordre1(bezier_coeff), axis=1)

def get_possiblerange(u, Poly):
    n = Poly.shape[1]
    for i in range(n - 1):
        if u >= i and u <= (i + 1):
            return i
    return None

def P_prime(u, Poly, option):
    i = get_possiblerange(u, Poly)
    BezierCoeff = get_BezierCoeff(Poly, i, option)
    BezierCoeff = delta_ordre1(BezierCoeff)
    sum = np.zeros(2)
    N = len(BezierCoeff[0,:]) - 1
    for k in range(len(BezierCoeff[0,:])):
        sum += 3 * BernsteinPolynom(N, k, u - i) * BezierCoeff[:, k]
    return sum

def P_second(u, Poly, option):
    i = get_possiblerange(u, Poly)
    BezierCoeff = get_BezierCoeff(Poly, i, option)
    BezierCoeff = delta_ordre2(BezierCoeff)
    sum = np.zeros(2)
    N = len(BezierCoeff[0,:]) - 1
    for k in range(len(BezierCoeff[0,:])):
        sum += 6 * BernsteinPolynom(N, k, u - i) * BezierCoeff[:, k]
    return sum

def K(u, Poly, option):
    P_pr = P_prime(u, Poly, option)
    P_sec = P_second(u, Poly, option)
    denom = np.linalg.norm(P_pr)**3
    num = np.cross(P_pr, P_sec)  # Produit vectoriel en 2D
    return num / denom


def LagrangeCurvature(u,Poly):
    # Séparer les coordonnées x et y
    x_points = Poly[0, :]
    y_points = Poly[1, :]
    n = len(x_points)
    
    # Définir les points u associés
    u_points = np.arange(0, n)  # u = 0, 1, ..., n-1
    
    # Interpolation de Lagrange pour x(u) et y(u)
    lagrange_x = lagrange(u_points, x_points)
    lagrange_y = lagrange(u_points, y_points)
    
    # Obtenir les coefficients et leurs dérivées
    coeffs_x = lagrange_x.coef
    coeffs_y = lagrange_y.coef

    # Première et deuxième dérivées
    poly_prime_x = np.polyder(coeffs_x)
    poly_double_prime_x = np.polyder(poly_prime_x)
    poly_prime_y = np.polyder(coeffs_y)
    poly_double_prime_y = np.polyder(poly_prime_y)

    # Évaluer les dérivées en u
    x_prime = np.polyval(poly_prime_x, u)
    x_second_prime = np.polyval(poly_double_prime_x, u)
    y_prime = np.polyval(poly_prime_y, u)
    y_second_prime = np.polyval(poly_double_prime_y, u)

    # Calculer les vecteurs dérivés
    LagrangePrime = np.array([x_prime, y_prime])
    LagrangeSecondPrime = np.array([x_second_prime, y_second_prime])

    # Calcul de la courbure
    denom = np.linalg.norm(LagrangePrime)**3
    num = np.cross(LagrangePrime, LagrangeSecondPrime)
    curvature = num / denom

    return curvature


def draw_curvature(Polygon):
    if Polygon is None or len(Polygon[0,:])==0:
        sys.exit(1)
    n = len(Polygon[0,:])
    T = np.linspace(0,n-1, 100)
    for i in range(5):
        if functions_used[i][0]:
            if i in [0,1,2,4] :
                K_values = [K(u, Polygon, i) for u in T]
                plt.plot(T, K_values,label=legend[i])
                continue
            else :
                K_values = [LagrangeCurvature(u,Polygon) for u in T]
                plt.plot(T,K_values,label=legend[i])
    plt.title("Curvature Plot")
    plt.xlabel("u")
    plt.ylabel("K(u)")
    plt.legend()
    plt.draw() 
    plt.show()

if __name__=="__main__":
    result = subprocess.run(['python', 'comparaison.py'], capture_output=True, text=True)
    output = result.stdout.strip().split('\n')
    # Traiter la deuxième ligne pour récupérer le dictionnaire
    functions_used = parse_to_dict(output[1])
    # Afficher les données récupérées
    Poly = parse_to_numpy_array(output[0])
    draw_curvature(Poly)