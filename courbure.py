import numpy as np
import matplotlib.pyplot as plt
from comparaison import *

choice = {
    0 : cardinal_splines,
    1 : cardinal_splines_v1,
    2 : cardinal_splines_v2,
}

def BernsteinPolynom(N,i,t):
    assert i<=N
    return nchoosek(N,i)*t**i (1-t)**(N-i)


def get_BezierCoeff(Poly,index,option=0):
    assert option in [0,1,2],"L'option est comprise entre 0 et 2"
    n = len(Poly[0,:])
    if index < 0 or index >= n - 1:
        raise IndexError("Index out of bounds")
    tang1 = choice[option](Poly,index)
    tang2 = choice[option](Poly,index)
    return  np.array([
            Poly[:, index],  # Utiliser Poly pour accéder aux points (les coordonnées x et y)
            Poly[:, index] + tang1 / 3,
            Poly[:, index + 1] - tang2 / 3,
            Poly[:, index + 1]
        ]).T

def delta_ordre1(bezier_coeff):
    m = len(bezier_coeff[0,:])
    res  = np.ones((2,m-1))
    for i in range(m-1):
        res[:,i] = bezier_coeff[:,i+1] - bezier_coeff[:,i]
    return res

def delta_ordre2(bezier_coeff):
    previous_res = delta_ordre1(bezier_coeff)
    m = len(previous_res[0,:])
    res = np.ones((2,m-1))
    for i in range(m-1):
        res[:,i] = previous_res[:,i+1] - previous_res[:,i]
    return res

def get_possiblerange(u,Poly):
    n = len(Poly[0,:])
    for i in range(n-1):
        if u>=i and u<=(i+1):
            return i
    return None

def P_prime(u,Poly,option):
    i = get_possiblerange(u,Poly) # Possible range : [u_i,u_{i+1}] = [i,i+1]
    BezierCoeff = get_BezierCoeff(Poly,i,option)
    BezierCoeff = delta_ordre1(BezierCoeff)
    m = len(BezierCoeff)
    sum = np.ones((1,2))
    N=2
    for k in range(m):
        sum+=3*BernsteinPolynom(N,k,u)*BezierCoeff[:,k]
    return sum

def P_second(u,Poly,option):
    i = get_possiblerange(u,Poly) # Possible range : [u_i,u_{i+1}] = [i,i+1]
    BezierCoeff = get_BezierCoeff(Poly,i,option)
    BezierCoeff = delta_ordre2(BezierCoeff)
    m = len(BezierCoeff)
    sum = np.ones((1,2))
    N=1
    for k in range(m):
        sum+=6*BernsteinPolynom(N,k,u)*BezierCoeff[:,k]
    return sum

def K(u,Poly,option):
    P_pr = P_prime(u,Poly,option) 
    P_sec = P_second(u,Poly,option)
    denom = np.linalg.norm(P_pr)**3
    A = np.array([P_pr,P_sec])
    num = np.linalg.det(A)
    return num/denom


def main():
    pass



if __name__=="__main__":
    main()