import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


# Script Mohammed Reda Belfaida
# Script fortement inspiré de celui de Mme Manon Vialle     

xdata = 0
ydata = 0
Poly = None
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
curve, = plt.plot([], 'g')
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


def PlotBezierCurve(Polygon):
    N = len(Polygon[0, :])-1
    t = np.linspace(0,1,200)
    Bern = Bernstein(N, t)
    Bezier = Polygon @ Bern
    xdata = np.concatenate((curve.get_xdata(), Bezier[0, :]))
    ydata = np.concatenate((curve.get_ydata(), Bezier[1, :]))
    curve.set_xdata(xdata)
    curve.set_ydata(ydata)
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
    print(Poly)
    for i in range(n - 1):
        print(f"{i}-ème itération")
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


def cardinal_splines_v2(Poly, index, c=0):
    pass

class Index(object):

    def newPoly(self, event):
        global Poly
        Poly = AcquisitionPolygone(minmax,'or',':r')
        PlotHermiteSpline(Poly, slider.val)

    def addPoint(self, event):
        global Poly
        Poly = AcquisitionNvxPoints(minmax,'or',':r')
        PlotHermiteSpline(Poly, slider.val)

    def removePoint(self, event):
        global Poly
        Poly = AcquisitionRMVPoints(minmax,'or',':r')
        PlotHermiteSpline(Poly, slider.val)

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
    PlotHermiteSpline(Poly, c)

slider.on_changed(update)

# Callback functions for the buttons
callback = Index()

bnewPoly.on_clicked(callback.newPoly)
baddpoints.on_clicked(callback.addPoint)
brempoints.on_clicked(callback.removePoint)

plt.show()
