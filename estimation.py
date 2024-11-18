import numpy as np

def cardinal_splines(Poly, index, c=0):
    n = len(Poly[0,:])
    if index < 0 or index >= n:
        raise IndexError("Index hors des limites du tableau de points.")
    # Calcul des tangentes
    if index == 0:
        return (1-c)*(Poly[:,index+1] - Poly[:,index]) 
    elif index == n - 1:
        return (1-c)*(Poly[:,index] - Poly[:,index - 1])
    return (1 - c) * (Poly[:,index + 1] - Poly[:,index - 1]) / 2

def cardinal_splines_v1(Poly, index, alpha=0.1):
    n=len(Poly[0,:])
    if alpha==0:
        raise ZeroDivisionError("On ne pourra pas diviser par 0")
    if index < 0 or index >= n:
        raise IndexError("Index hors des limites du tableau de points.")
    if index==0:
        return (Poly[:,index+1] - Poly[:,index])/alpha
    if index==n-1:
        return (Poly[:,index] - Poly[:,index-1])/alpha
    return (Poly[:,index+1] - Poly[:,index-1])/alpha

def cardinal_splines_v2(Poly, index):
    n=len(Poly[0,:])

    if index < 0 or index >= n:
        raise IndexError("Index hors des limites du tableau de points.")
    
    if index == n-1:
        return Poly[:,index]-Poly[:,index-1]
    
    if index == n-2:
        return (Poly[:,index+1]-Poly[:,index-1])/2
    
    return (4*Poly[:,index+1]-3*Poly[:,index]+Poly[:,index+2])/2

def derive_cubique(Poly):

    # Get number of columns (N>2)
    N = Poly.shape[1]
    col = np.zeros((Poly.shape[0], N))
    col[:, 0] = 3 * (Poly[:, 1] - Poly[:, 0])
    col[:, 1:N-1] = 3 * (Poly[:, 2:N] - Poly[:, 0:N-2])
    col[:, N-1] = 3 * (Poly[:, N-1] - Poly[:, N-2])
    
    # Create diagonal matrices
    # Lower diagonal
    matrice1 = np.diagflat(np.ones(N-1), -1)
    # Upper diagonal
    matrice2 = np.diagflat(np.ones(N-1), 1)
    # Main diagonal (multiplied by 4)
    matrice3 = 4 * np.eye(N)
    matrice3[0, 0] = 2
    matrice3[N-1, N-1] = 2
    
    # Sum the matrices
    A = matrice1 + matrice2 + matrice3
    
    # Solve the system and transpose
    derives = np.linalg.solve(A, col.T).T
    
    return derives
