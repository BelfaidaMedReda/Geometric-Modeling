import sys
import numpy as np
import matplotlib.pyplot as plt
from estimation import * 
from matplotlib.widgets import Button, Slider

# Script Mohammed Reda Belfaida
# Script fortement inspiré de celui de Mme Manon Vialle     

functions_used = {
    0 : [False,0],    # standard Hermite Interpolation
    1 : [False,0.1],  # Hermite_v1 Interpolation
    2 : [False,None], # Hermite_v2 Interpolation
    3 : [False,None], # Lagrange Interpolation
    4 : [False,None]  # Cubique Spline Interpolation 
}

xdata = 0
ydata = 0
Poly = None


#factorial n

def fact(n):
    res = 1
    for i in range(n):
        res *= (i+1)
    return res

#----------------------
# Binomial coefficient
def nchoosek(n,k):
    return fact(n)/(fact(k)*fact(n-k))

#---------------------
# Bernstein Polynomials
# N is the degree
# t = np.linspace(0,1,500)
def Bernstein(N,t):
    BNt = np.zeros((N+1, t.size))
    for i in range(N+1):
         BNt[i, :] = (nchoosek(N, i)*(t**i)*(1-t)**(N-i) )  
    return BNt

#----------------------
# plot of the Bernstein polynomials
def plotBernPoly():
    N=5
    x = np.linspace(0,1,500)
    Bern = Bernstein(N,x)
    for k in range(N+1):
        plt.plot(x,Bern[k, :])


def PlotBezierCurve(Polygon,option=0):
    N = len(Polygon[0, :])-1
    t = np.linspace(0,1,200)
    Bern = Bernstein(N, t)
    Bezier = Polygon @ Bern
    match option : 
        case 0 : 
            xdata = np.concatenate((curve.get_xdata(), Bezier[0, :]))
            ydata = np.concatenate((curve.get_ydata(), Bezier[1, :]))
            curve.set_xdata(xdata)
            curve.set_ydata(ydata)
        case 1 : 
            xdata = np.concatenate((curve1.get_xdata(), Bezier[0, :]))
            ydata = np.concatenate((curve1.get_ydata(), Bezier[1, :]))
            curve1.set_xdata(xdata)
            curve1.set_ydata(ydata)
        case 2 :
            xdata = np.concatenate((curve2.get_xdata(), Bezier[0, :]))
            ydata = np.concatenate((curve2.get_ydata(), Bezier[1, :]))
            curve2.set_xdata(xdata)
            curve2.set_ydata(ydata)
        case 4 : 
            xdata = np.concatenate((curve4.get_xdata(), Bezier[0, :]))
            ydata = np.concatenate((curve4.get_ydata(), Bezier[1, :]))
            curve4.set_xdata(xdata)
            curve4.set_ydata(ydata)
    functions_used[option][0] = True
    plt.draw()
    return     



def AcquisitionPolygone(minmax,color1,color2) :
    x = []
    y = []
    coord = 0
    while coord != []:
        coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        if coord != []:
            plt.draw()
            xx = coord[0][0]
            yy = coord[0][1]
            points.set_xdata(np.append(points.get_xdata(), xx))
            points.set_ydata(np.append(points.get_ydata(), yy))
            x.append(xx)
            y.append(yy)
            poly.set_xdata(points.get_xdata())
            poly.set_ydata(points.get_ydata())
            #plt.plot([x[-2],x[-1]],[y[-2],y[-1]],color2)
    #Polygon creation
    Polygon = np.zeros((2,len(x)))
    Polygon[0,:] = x
    Polygon[1,:] = y
    return Polygon


def AcquisitionNvxPoints(minmax,color1,color2):
    x = points.get_xdata().copy()
    y = points.get_ydata().copy()
    coord = 0
    while coord != []:
        coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        if coord != []:
            plt.draw()
            xx = coord[0][0]
            yy = coord[0][1]
            points.set_xdata(np.append(points.get_xdata(), xx))
            points.set_ydata(np.append(points.get_ydata(), yy))
            x = np.append(x, xx)
            y = np.append(y, yy)
            poly.set_xdata(points.get_xdata())
            poly.set_ydata(points.get_ydata())
            #plt.plot([x[-2],x[-1]],[y[-2],y[-1]],color2)
    #Polygon creation
    Polygon = np.zeros((2,len(x)))
    Polygon[0,:] = x
    Polygon[1,:] = y
    return Polygon


def AcquisitionRMVPoints(minmax,color1,color2):
    x = points.get_xdata().copy()
    y = points.get_ydata().copy()
    coord = 0
    while coord != []:
        coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        if coord != []:
            plt.draw()
            xx = coord[0][0]
            yy = coord[0][1]
            dist = [(xx-x[i])**2+(yy-y[i])**2 for i in range(x.size)]
            index_min = min(range(len(dist)), key=dist.__getitem__)
            points.set_xdata(np.delete(points.get_xdata(), index_min))
            points.set_ydata(np.delete(points.get_ydata(), index_min))
            x = np.delete(x, index_min)
            y = np.delete(y, index_min)
            poly.set_xdata(points.get_xdata())
            poly.set_ydata(points.get_ydata())
            #plt.plot([x[-2],x[-1]],[y[-2],y[-1]],color2)
    #Polygon creation
    Polygon = np.zeros((2,len(x)))
    Polygon[0,:] = x
    Polygon[1,:] = y
    return Polygon


def PlotHermiteSpline(Poly, c=0):
    n = len(Poly[0,:])  # Poly est un tableau 2xN, donc on prend len(Poly[0]) pour obtenir le nombre de points
    for i in range(n - 1):
        # Calcul des tangentes pour chaque point de contrôle
        tang1 = cardinal_splines(Poly, i, c)  # Utilisation de Poly pour les coordonnées
        tang2 = cardinal_splines(Poly, i + 1, c)  # Utilisation de Poly pour les coordonnées
        
        # Points de contrôle pour la courbe de Bézier (degré 3)
        bezier_points = np.array([
            Poly[:, i],  # Utiliser Poly pour accéder aux points (les coordonnées x et y)
            Poly[:, i] + tang1 / 3,
            Poly[:, i + 1] - tang2 / 3,
            Poly[:, i + 1]
        ]).T
        
        # Utilisation de De Casteljau pour interpoler les points de Bézier
        PlotBezierCurve(bezier_points)



def PlotHermiteSpline_v1(Poly, alpha=0.1):
    """
    Hermite spline en utilisant la version 1 
    pour l'estimation des tangentes
    """
    n = len(Poly[0,:])  # Poly est un tableau 2xN, donc on prend len(Poly[0]) pour obtenir le nombre de points
    for i in range(n - 1):
        # Calcul des tangentes pour chaque point de contrôle
        tang1 = cardinal_splines_v1(Poly, i, alpha)  # Utilisation de Poly pour les coordonnées
        tang2 = cardinal_splines_v1(Poly, i + 1, alpha)  # Utilisation de Poly pour les coordonnées
        
        # Points de contrôle pour la courbe de Bézier (degré 3)
        bezier_points = np.array([
            Poly[:, i],  # Utiliser Poly pour accéder aux points (les coordonnées x et y)
            Poly[:, i] + tang1 / 3,
            Poly[:, i + 1] - tang2 / 3,
            Poly[:, i + 1]
        ]).T
        
        # Utilisation de De Casteljau pour interpoler les points de Bézier
        PlotBezierCurve(bezier_points,1)


def PlotHermiteSpline_v2(Poly):
    """
    Hermite spline en utilisant la version 1 
    pour l'estimation des tangentes
    """
    n = len(Poly[0,:])  # Poly est un tableau 2xN, donc on prend len(Poly[0]) pour obtenir le nombre de points
    for i in range(n - 1):
        # Calcul des tangentes pour chaque point de contrôle
        tang1 = cardinal_splines_v2(Poly, i)  # Utilisation de Poly pour les coordonnées
        tang2 = cardinal_splines_v2(Poly, i + 1)  # Utilisation de Poly pour les coordonnées
        
        # Points de contrôle pour la courbe de Bézier (degré 3)
        bezier_points = np.array([
            Poly[:, i],  # Utiliser Poly pour accéder aux points (les coordonnées x et y)
            Poly[:, i] + tang1 / 3,
            Poly[:, i + 1] - tang2 / 3,
            Poly[:, i + 1]
        ]).T
         
        # Utilisation de De Casteljau pour interpoler les points de Bézier
        PlotBezierCurve(bezier_points,2)

def PlotLagrangeCurve(Polygon):
    """
    Trace la courbe interpolée de Lagrange en utilisant une structure similaire
    à PlotBezierCurve.

    Arguments :
    - Polygon : np.array 2xN, contenant les points de contrôle (x et y).
    """
    N = len(Polygon[0, :]) - 1
    t = np.linspace(0, N, 200)  # Résolution pour les points interpolés
    u = np.arange(N + 1)  # u_i = [0, ..., N]
    
    # Calcul des points interpolés avec Lagrange
    x_interp = np.zeros_like(t)
    y_interp = np.zeros_like(t)
    
    for i in range(N + 1):
        # Calcul du terme de base de Lagrange L_i(t)
        L_i = np.ones_like(t)
        for j in range(N + 1):
            if i != j:
                L_i *= (t - u[j]) / (u[i] - u[j])
        
        # Ajout de la contribution de L_i
        x_interp += Polygon[0, i] * L_i
        y_interp += Polygon[1, i] * L_i

    # Mise à jour des données du tracé
    xdata = np.concatenate((curve3.get_xdata(), x_interp))
    ydata = np.concatenate((curve3.get_ydata(), y_interp))
    curve3.set_xdata(xdata)
    curve3.set_ydata(ydata)
    functions_used[3][0] = True
    plt.draw()

def PlotSplineCubique(Polygon):
    """
    Trace la courbe interpolée en utilisant la démarche spéciale utilisée en
    algorithme de raccordement
    """
    n = Polygon.shape[1]  # Nombre de points de contrôle
    derivees = derive_cubique(Polygon)
    for i in range(n - 1):
        tang1 = derivees[:,i]
        tang2 = derivees[:,i+1]
        bezier_points = np.array([
            Poly[:, i],  # Utiliser Poly pour accéder aux points (les coordonnées x et y)
            Poly[:, i] + tang1 / 3,
            Poly[:, i + 1] - tang2 / 3,
            Poly[:, i + 1]
        ]).T
        PlotBezierCurve(bezier_points,4)

def DeCasteljau(poly, T):
    n = poly.shape[0] - 1
    result = []
    for t in T:
        r = poly.copy()
        for k in range(0, n):
            for i in range(0, n - k):
                r[i, :] = (1 - t) * r[i, :] + t * r[i + 1, :]

        result.append(r[0, :])
    return np.array(result)



class Index(object):

    def newPoly(self, event):
        global Poly
        Poly = AcquisitionPolygone(minmax,'or',':r')
        PlotHermiteSpline(Poly,slider.val)
        PlotLagrangeCurve(Poly)
        PlotSplineCubique(Poly)

    def addPoint(self, event):
        global Poly
        Poly = AcquisitionNvxPoints(minmax,'or',':r')
        PlotHermiteSpline(Poly,slider.val)
        PlotLagrangeCurve(Poly)
        PlotSplineCubique(Poly)
        

    def removePoint(self, event):
        global Poly
        Poly = AcquisitionRMVPoints(minmax,'or',':r')
        PlotHermiteSpline(Poly,slider.val)
        PlotLagrangeCurve(Poly)
        PlotSplineCubique(Poly)


if __name__=="__main__":
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    minmax = 10
    ax.set_xlim((-minmax,minmax))
    ax.set_ylim((-minmax,minmax))
    plt.title("Menu")
    plt.subplots_adjust(bottom=0.3)
    t = np.arange(0.0, 1.0, 0.001)
    points, = plt.plot([], [], 'bx')
    poly, = plt.plot([], 'r')
    curve, = plt.plot([], 'g',label="HermiteS")
    curve1, = plt.plot([], 'y',label="HermiteS_v1")
    curve2, = plt.plot([],'m',label="HermiteS_v2")
    curve3, = plt.plot([],'c',label="Lagrange")
    curve4, = plt.plot([],'b',label="CubicS")
    plt.legend([curve,curve1,curve2,curve3,curve4],["HermiteS","HermiteS_v1","Hermite_v2","Lagrange","CubicS"],loc="upper right")
    callback = Index()
    # Add buttons with proper spacing to avoid overlap
    axaddpoints = plt.axes([0.05, 0.05, 0.2, 0.075])  # Adjusted position
    axnewPoly = plt.axes([0.28, 0.05, 0.2, 0.075])  # Adjusted position
    axrempoint = plt.axes([0.51, 0.05, 0.2, 0.075])  # Adjusted position

    bnewPoly = Button(axnewPoly, 'New Polygon')
    baddpoints = Button(axaddpoints, 'Add Point')
    brempoints = Button(axrempoint, 'Remove Point')

    # Adjust slider position to avoid overlap
    ax_slider = plt.axes([0.03, 0.2, 0.9, 0.03], facecolor='lightgoldenrodyellow')  # Adjusted width and position
    slider = Slider(ax_slider, label='c', valmin=0, valmax=1, valinit=0)

    def update(val):
        # Efface la courbe actuelle pour éviter les superpositions
        curve.set_xdata([])
        curve.set_ydata([])
        
        # Redessine la courbe avec la nouvelle valeur du slider
        c = slider.val
        functions_used[0][1] = c
        PlotHermiteSpline(Poly, c)


    # Adjust slider position to avoid overlap
    other_ax_slider = plt.axes([0.03, 0.15, 0.9, 0.03], facecolor='black')
    other_slider = Slider(other_ax_slider, label='α', valmin=0.1, valmax=100, valinit=0.5)

    def other_update(val):
        # Efface la courbe actuelle pour éviter les superpositions
        curve1.set_xdata([])
        curve1.set_ydata([])
        
        # Redessine la courbe avec la nouvelle valeur du slider
        alpha = other_slider.val
        functions_used[1][1] = alpha
        PlotHermiteSpline_v1(Poly,alpha)

    slider.on_changed(update)
    other_slider.on_changed(other_update)

    # Callback functions for the buttons
    callback = Index()

    bnewPoly.on_clicked(callback.newPoly)
    baddpoints.on_clicked(callback.addPoint)
    brempoints.on_clicked(callback.removePoint)
    plt.show()

    # Decomment the follow lines whenever comparaison.py is executed as a subprocess of courbure.py file
    print(Poly.tolist())
    print(functions_used)