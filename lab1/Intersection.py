# ExclusionV3.py
# version un peu plus optimisee

"""

Intersection of two plane Bezier curves 

Step 1 :
    an estimate of the intersection points are determined according 
    - the exclusion principle
    - and the subdivision principle
    with a recursive implementation
    This step provides initial values for Step 2

Step 2 :
    the intersection points are evaluated with the Newton-Raphson method
    in the parameter domain

The recursive function in Step 1 stops :
- when subdivided Bezier curves no more intersect in the 2D plane, 
- or when the lengths d1 and d2 of the 2 intervals containing 
a root-intersection are smaller than a given epsilon value

Due to recursivity, we possibly get several intervals containing 
the same root
==> these intersections in the parametric domain are stored
    in the lists ListRoots_u and ListRoots_v

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom

#  BEZIER FUNCTIONS


def Bernstein(n, k, x):
    coeff = binom(n, k)
    return coeff * x**k * (1 - x)**(n - k)

def BezierPlot(ax,xptcontrole,yptcontrole,color3):
    """ Display the whole Bézier curve
    """
    degree = len(xptcontrole) - 1
    nbti = 1000
    ttrace = np.linspace(0,1,nbti)
    tx = np.zeros(nbti)
    ty = np.zeros(nbti)
    for k in range(degree+1):
        poly = Bernstein(degree,k,ttrace)
        tx = (tx + xptcontrole[k]*poly)
        ty = (ty + yptcontrole[k]*poly)
    ax.plot(tx,ty,color3,lw=1)
    plt.draw()

def BezierPlotOnePoint(ax,xptcontrole,yptcontrole,t0,color7):
    """ Display value P(t0) of the Bézier curve
    """
    degree = len(xptcontrole) - 1
    x0 = 0.
    y0 = 0.
    for k in range(degree+1):
        poly = Bernstein(degree,k,t0)
        x0 = x0 + xptcontrole[k]*poly
        y0 = y0 + yptcontrole[k]*poly
    ax.plot(x0,y0,color7,ms=8)
    plt.draw()

def PolygonAcquisition(ax,color1,color2) :
    """ Acquisition of a 2D polygon in the window with subplot "ax" 
        right click stop the acquisition
    """
    x = []  # x is an empty list
    y = []
    coord = 0
    while coord != []:
        coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        # coord is a list of tuples : coord = [(x,y)]
        if coord != []:
            xx = coord[0][0]
            yy = coord[0][1]
            ax.plot(xx,yy,color1,ms=8)
            x.append(xx)
            y.append(yy)
            plt.draw()
            if len(x) > 1 :
                ax.plot([x[-2],x[-1]],[y[-2],y[-1]],color2)
    return x,y

def OneSubdivBezierCurve(xpt, ypt):
    """ One subdivision of the input control polygon according parameter 1/2
        Return the two subdivided control polygons 
            xpt1, ypt1
            xpt2, ypt2
    """
    n = len(xpt) - 1 # degree of the Bézier curve
    xp = np.zeros(2*n+1)
    yp = np.zeros(2*n+1)
    # Initialization
    for i in range(n+1):
        xp[2*i] = xpt[i]
        yp[2*i] = ypt[i]
    # Main loop
    for j in range(1,n+1):
        for i in range(n+1-j):
            k = j + 2*i
            xp[k] = ( xp[k-1] + xp[k+1] ) / 2.
            yp[k] = ( yp[k-1] + yp[k+1] ) / 2.
    # We split the subivided polygon in two control polygons
    xpt1 = xp[0:n+1]
    xpt2 = xp[n:2*n+1]
    ypt1 = yp[0:n+1]
    ypt2 = yp[n:2*n+1]  
    return xpt1, ypt1, xpt2, ypt2


# BOXES FUNCTIONS


def MinMaxBoxCP(xControlPt, yControlPt):
    """ Determination of the rectangular min-max box that contains 
        all control points
        Return the tuple Box = (Bx, By) with :
            Bx = np.array([xmin, xmax])
            By = np.array([ymin, ymax])
    """
    Bx = np.array([min(xControlPt), max(xControlPt)])
    By = np.array([min(yControlPt), max(yControlPt)])
    Box = (Bx, By)
    return Box

def IntersectBox1D(I1, I2):
    """ Intersection of the 1-dimensional boxes I1 and I2
        precisely, here I1 and I2 are intervals : 
            I1 = [a1,b1] with a1 <= b1 , I1 is an np.array
            I2 = [a2,b2] with a2 <= b2 , I2 is an np.array
        return  inter = 1 if the two intervals intersect, = 0 if not
                K = the intersection interval (np.array)
                  = np.array([]) if inter = 0
    """
    if ( I1[1] < I2[0] ) or ( I1[0] > I2[1] ) :
        K = np.array([])
        inter = 0
    else:
        K = I1
        inter = 1
        if I1[0] < I2[0] :
            K[0] = I2[0]
        if I1[1] > I2[1] :
            K[1] = I2[1]
    return inter, K

def IntersectBox2D(Box1, Box2):
    """ Intersection of the 2-dimensional boxes Box1 and Box2
        A box is a tuple Box = (Bx, By) with :
            Bx = np.array([xmin, xmax])
            By = np.array([ymin, ymax])
        Return  1 if the boxes intersect, 0 if they do not intersect
                the intersection box in case Box1 and Box2 intersect
    """
    box = ()
    interx, Kx = IntersectBox1D(Box1[0], Box2[0])
    intery, Ky = IntersectBox1D(Box1[1], Box2[1])
    inter = interx * intery
    if inter == 1 :
        box = (Kx, Ky)
    return inter, box

def PlotBox(ax, box, color4):
    """ Plot the rectangular box = (Bx, By)
            Bx = np.array([xmin, xmax])
            By = np.array([ymin, ymax])
        in the window with subplot "ax"
    """
    x1 = box[0][0]
    x2 = box[0][1]
    y1 = box[1][0]
    y2 = box[1][1]
    x = [x1, x2, x2, x1, x1]
    y = [y1, y1, y2, y2, y1]
    ax.plot(x,y,color4,lw=1)



# THE RECURSIVE FUNCTION


def BezierIntersection(ax, xp1, yp1, I1, xp2, yp2, I2, eps):
    """ Recursive intersection of 2 Bezier curves
    
    Parameters:
    ax: matplotlib Axes object for plotting
    xp1, yp1: Control points of the first Bezier curve
    xp2, yp2: Control points of the second Bezier curve
    I1, I2: Associated intervals for each curve (included in [0,1])
    eps: Epsilon value for termination condition
    
    The function finds intersections between two Bezier curves using a recursive approach.
    """
    
    # Calculate bounding boxes for both Bezier curves
    Box1 = MinMaxBoxCP(xp1, yp1)
    Box2 = MinMaxBoxCP(xp2, yp2)
    
    # Check if bounding boxes intersect
    inter, inter_box = IntersectBox2D(Box1, Box2)
    if not inter:
        return  # No intersection, terminate recursion
    
    # Check if both curve segments are small enough (termination condition)
    if (I1[1] - I1[0] < eps) and (I2[1] - I2[0] < eps):
        # Calculate intersection point
        x, y = (inter_box[0][0] + inter_box[0][1]) / 2, (inter_box[1][0] + inter_box[1][1]) / 2
        ax.plot(x, y, "go")  # Plot intersection point
        
        # Refine intersection using Newton's method
        inter_parameters = Newton2Var(xp1, yp1, I1, xp2, yp2, I2, NbOfNewtonSteps)
        u, v = inter_parameters
        
        # Store intersection parameters
        ListRoots_u.append(u)
        ListRoots_v.append(v)
        
        # Plot refined intersection points on both curves
        BezierPlotOnePoint(ax, xp1, yp1, u, "r+")
        BezierPlotOnePoint(ax, xp2, yp2, v, "bx")
        return 
    
    # Subdivision of each Bézier curve
    xp11, yp11, xp12, yp12 = OneSubdivBezierCurve(xp1, yp1)
    xp21, yp21, xp22, yp22 = OneSubdivBezierCurve(xp2, yp2)
    
    # Calculate midpoints of intervals
    center1 = (I1[0] + I1[1]) / 2
    center2 = (I2[0] + I2[1]) / 2
    
    # Create new intervals for subdivided curves
    I11 = [I1[0], center1]
    I12 = [center1, I1[1]]
    I21 = [I2[0], center2]
    I22 = [center2, I2[1]]
    
    # Uncomment the following block to visualize bounding boxes during recursion
    """
    Box11 = MinMaxBoxCP(xp11, yp11)
    Box12 = MinMaxBoxCP(xp12, yp12)
    Box21 = MinMaxBoxCP(xp21, yp21)
    Box22 = MinMaxBoxCP(xp22, yp22)

    PlotBox(ax, Box11, "--r")
    PlotBox(ax, Box12, "--r")
    PlotBox(ax, Box21, "--b")
    PlotBox(ax, Box22, "--b")
    """
    
    # Prepare data for recursive calls
    iter1 = [[xp11, yp11, I11], [xp12, yp12, I12]]
    iter2 = [[xp21, yp21, I21], [xp22, yp22, I22]]
    
    # Recursive calls for all combinations of subdivided curves
    for i in range(2):
        for j in range(2):
            BezierIntersection(ax, iter1[i][0], iter1[i][1], iter1[i][2],
                               iter2[j][0], iter2[j][1], iter2[j][2], eps)


# NEWTON FUNCTIONS


def BezierDerivative1D(PC,t0):
    """ PC : list of scalar values (control points)
        Return value P'(t0) of the Bézier function 
    """
    degree = len(PC) - 1
    PC = np.array(PC)
    # Control points of the derivative
    DPC = PC[1:] - PC[:-1]
    # evaluation
    Pprime = 0.
    # derivative of degree - 1
    for k in range(degree):
        poly = Bernstein(degree-1,k,t0)
        Pprime = Pprime + DPC[k]*poly
    return degree * Pprime

def BezierValue1D(PC,t0):
    """ PC : list of scalar values (control points)
        Return value P(t0) of the Bézier function
    """
    degree = len(PC) - 1
    Pval = 0.
    for k in range(degree+1):
        poly = Bernstein(degree,k,t0)
        Pval = Pval + PC[k]*poly
    return Pval

def Newton2Var(xp1,yp1,I1,xp2,yp2,I2,NbSteps):
    # initial values
    u0 = (I1[0] + I1[1]) / 2.
    v0 = (I2[0] + I2[1]) / 2.
    X0 = np.array([u0,v0])
    
    for k in range(NbSteps):
        u0 = X0[0]
        v0 = X0[1]
        # Inverse D of Jacobian matrix : D = J^-1
        D = np.zeros((2,2))
        D[0][0] = - BezierDerivative1D(yp2,v0)
        D[1][0] = - BezierDerivative1D(yp1,u0)
        D[0][1] = BezierDerivative1D(xp2,v0)
        D[1][1] = BezierDerivative1D(xp1,u0)
        D = D / np.linalg.det(D)
        # Second member
        b = np.zeros(2)
        b[0] = BezierValue1D(xp1,u0) - BezierValue1D(xp2,v0)
        b[1] = BezierValue1D(yp1,u0) - BezierValue1D(yp2,v0)
        # recurrence
        X0 = X0 - np.dot(D,b)
    return X0



############################################
if __name__ == '__main__':
    
    minmax = 0.002
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_xlim((-minmax,minmax))
    ax.set_ylim((-minmax,minmax))
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_title("Acquisition window") 

    # List of roots in the parametric domain    
    ListRoots_u = []
    ListRoots_v = []
    
    # Number of Newton steps
    NbOfNewtonSteps = 7

    # Bezier curve acquisition
    xp1, yp1 = PolygonAcquisition(ax,'ob','b')
    BezierPlot(ax,xp1,yp1,'b')
    xp2, yp2 = PolygonAcquisition(ax,'or','r')
    BezierPlot(ax,xp2,yp2,'r')

    # the bounding boxes
    box1 = MinMaxBoxCP(xp1, yp1)
    PlotBox(ax, box1, '--b')
    box2 = MinMaxBoxCP(xp2, yp2)
    PlotBox(ax, box2, '--r')
    
    # initialization... and go !
    #eps = 0.005
    eps = 0.002
    inter, box = IntersectBox2D(box1, box2)
    if inter :
        I1 = [0,1]
        I2 = [0,1]
        BezierIntersection(ax,xp1,yp1,I1,xp2,yp2,I2,eps)
    
    if ListRoots_u == []:
        print('--> THE CURVES DO NOT INTERSECT')
    else:
        # For the reasons exlained in the attached document, we remove the paired duplicates
        print("--> INTERSECTIONS IN PARAMETRIC DOMAIN :")
        print("list of u-intersections : ",ListRoots_u)
        print("list of v-intersections : ",ListRoots_v)
    plt.show()






