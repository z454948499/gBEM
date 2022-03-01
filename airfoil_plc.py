# -*- coding: utf-8 -*-
"""
@author: GoldWind Blade Development Team
"""

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from scipy.optimize import golden
import copy
from scipy.interpolate import interp1d


class Point(object):
    
    pId = 1
    
    "Initilization"
    def __init__(self, x, y, name = ''):
        self.x = x
        self.y = y
        self.name = name
        self.id = Point.pId
        Point.pId = Point.pId + 1
    
    # "Create by (r,theta)"
    def cylinCoord(self, r, theta, name = ''):
        theta = theta * np.pi / 180
        self.x = r * np.cos(theta)
        self.y = r * np.sin(theta)
        self.name = name
        self.id = Point.pId
        Point.id = Point.pId + 1
        
    def move(self, dx, dy):
        self.x = self.x + dx
        self.y = self.y + dy
         
    
    def rotate(self, alpha, rx0 = 0, ry0 = 0):
        alpha = alpha * np.pi / 180
        xOld = copy.deepcopy(self.x)
        yOld = copy.deepcopy(self.y)
        self.x = (xOld - rx0) * np.cos(alpha) - (yOld - ry0) * np.sin(alpha) + rx0
        self.y = (xOld - rx0) * np.sin(alpha) + (yOld - ry0) * np.cos(alpha) + ry0        
         
    
    def scale(self, xratio, yratio):
        self.x = self.x * xratio
        self.y = self.y * yratio
         
        
    def distance(self, pnt):
        return ((self.x - pnt.x)**2 + (self.y - pnt.y)**2)**0.5 
    
    def getSlope(self, pnt): 
        if (self.x - pnt.x) == 0:
            slope = float("inf")
        else:
            slope = (self.y - pnt.y) / (self.x - pnt.x)
        
        return slope
    
    def getAngle(self, pnt):
        
        slope = self.getSlope(pnt)
        
        return np.arctan(slope) * 180/ np.pi
    
    
    def getId(self):
        return self.id
    
    def setXY(self, x, y):
        self.x = x
        self.y = y
    
    def getXY(self):
        return np.array([self.x, self.y])
        
    def setName(self, newName = ''):
        self.name = newName
    
    def getName(self):
        return self.name
    
    def printMethod(self):
        
        print('Point%d %s\t' %(self.id, self.name), '(x, y) = \
              (%f, %f)' %(self.x, self.y))
        
    def printMultiPnts(self, *pnts):
        
        self.printMethod()
        for pnt in pnts:
            pnt.printMethod()
            

    def plotPnt(self):
        
        plt.plot(self.x,self.y,'*',label = '%s%i %s' \
                 %(self.__class__.__name__, self.id, self.name))  # add coordinate and name 
        plt.annotate('(%.2f, %.2f)' %(self.x, self.y), \
                      xy = (self.x, self.y), xytext = (-20, 20), \
                      textcoords = 'offset points', xycoords = 'data')
        
    def plotMultiPnts(self, *pnts):
        
        self.plotPnt()
        
        x_range = np.array([self.x, self.x])
        y_range = np.array([self.y, self.y])
        
        for pnt in pnts:
            x_range[0] = min(x_range[0], pnt.x)
            x_range[1] = max(x_range[1], pnt.x)
            y_range[0] = min(y_range[0], pnt.y)
            y_range[1] = max(y_range[1], pnt.y)
            
            pnt.plotPnt()

        plt.legend(loc = 'best')
        
        x_range = x_range.mean() + 1.5 * (x_range - x_range.mean()) 
        y_range = y_range.mean() + 1.5 * (y_range - y_range.mean())
        plt.xlim(x_range)
        plt.ylim(y_range)
        

        
class Line(object):
    
    """
    Line is composed of [[P1][P2]]
    P1, P2 are composed of [x, y], respectively.
    The line built typically can be defined as y = ax + b 
    """

    lId = 1

    def __init__(self, p1, p2, name = ''):
        
        # p1 and p2 are of point type
        if not isinstance(p1, Point):
            p1 = Point(p1[0], p1[1])
        
        if not isinstance(p2, Point):
            p2 = Point(p2[0], p2[1])
               
        self.p1 = p1
        self.p2 = p2
        self.name = name
        self.id = Line.lId
        Line.lId = Line.lId + 1
        
    def getS(self):
        
        return self.p1.distance(self.p2)
        
    def getSlope(self):

        if (self.p1.x - self.p2.x) == 0:
            slope = float("inf")
        else:
            slope = (self.p1.y - self.p2.y) / (self.p1.x - self.p2.x)
        
        return slope
    
    def getIntercept(self):
        
        if (self.p2.x - self.p1.x) == 0:
            return "vertical line, none intercept"
        else:
            intercept = (self.p1.y * self.p2.x - self.p2.y * self.p1.x) \
                /(self.p2.x - self.p1.x)   
            return intercept
    
    def getAngle(self):
        
        slope = self.getSlope()
        
        return np.arctan(slope) * 180/ np.pi
    
    def move(self, dx, dy):   # may re-define add, sub
  
        self.p1.move(dx,dy)
        self.p2.move(dx,dy) 
        
    def scale(self, xratio, yratio):
        
        self.p1.scale(xratio, yratio)
        self.p2.scale(xratio, yratio)        
         
    def rotate(self, alpha, rx0 = 0, ry0 = 0):
        
        self.p1.rotate(alpha, rx0, ry0)
        self.p2.rotate(alpha, rx0, ry0)
    
    def getPtoLDist(self, point):
        
        if abs(self.p1.x - self.p2.x) <= 1e-6:
            return abs(point[0])
        elif abs(self.p1.y - self.p2.y) < 1e-6:
            return abs(point[1])
        else:
            slope = self.getSlope()
            intercept = self.getIntercept()
            distance = abs(slope * point[0] - point[1] + intercept) \
                / (slope**2 + 1)**(0.5)        
            return distance
            
    def dotProduct(self, newLine):
        
        return (self.p2.x - self.p1.x) * (newLine.p2.x - newLine.p1.x) + \
                 (self.p2.y - self.p1.y) * (newLine.p2.y - newLine.p1.y)
            
    
    def crossProduct(self, newLine):
        
        return (self.p2.x - self.p1.x) * (newLine.p2.y - newLine.p1.y)  \
            - (self.p2.y - self.p1.y) * (newLine.p2.x - newLine.p1.x)

    def setName(self,newName = ''):
        self.name = newName
    
    def getName(self):
        return self.name
    
    def getId(self):
        return self.id
    
    def xyToPnts(x, y):
        
        p1 = Point(x[0], y[0])
        p2 = Point(x[1], y[1])   
        
        return p1, p2
        
    def getXY(self):
        x = np.array([self.p1.x, self.p2.x])
        y = np.array([self.p1.y, self.p2.y])
        return x, y
       
    def plotLine(self):
        
        x, y = self.getXY()
        plt.plot(x,y, label = '%s%i %s' %(self.__class__.__name__, self.id, self.name))
        plt.legend(bbox_to_anchor = (1,0), loc = 3)
        
    def plotMultiLines(self, *lines):
        self.plotLine()
        
        for line in lines:
            line.plotLine()
        
        
    def printMethod(self):
        
        x, y = self.getXY()
        
        print('Line%i %s\t' %(self.id, self.name), 'P1 = ', (x[0], y[0]), \
              'P2 = ', (x[1], y[1]))
        
    def printMultiLines(self, *lines):
        
        self.printMethod()
        
        for line in lines:
            line.printMethod()
      
  
class Curve(object):
    
    # discrte points
    # spline is not introduced to calculate variables like slope
    # quite different from Airfoil where spline is generally introduced
    
    cId = 1
    
    # coord are composed of variables of Point class
    
    def __init__(self, coord, name = ''):
        
        self.coord = coord
        self.name = name
        self.id = Curve.cId
        Curve.cId = Curve.cId + 1
               
    def getS(self):
        
        S = np.zeros(len(self.coord))
        for i in range(1, len(self.coord)):
            S[i] = self.coord[i].distance(self.coord[i-1])
        return S
    
    def getSlope(self):
        
        slope = np.zeros(len(self.coord))
        
        for i in range(1, len(self.coord)):       
            slope[i] = self.coord[i].getSlope(self.coord[i-1])
        
            
        # attention, improved version
        slope[0]  = slope[1]
        slope[-1] = slope[-2]
            
        return slope
        
    def getAngle(self):
        
        slope = self.getSlope()       
        angle = np.arctan(slope) * 180 / np.pi

        return angle
    
    def move(self, dx, dy):
        
        pnts = self.coord
        
        for pnt in pnts:
            pnt.move(dx, dy)            
        
    def scale(self, xratio, yratio):
        
        pnts = self.coord
        
        for pnt in pnts:
            pnt.scale(xratio, yratio)

    def rotate(self, alpha, rx0 = 0, ry0 = 0):
        
        pnts = self.coord
        
        for pnt in pnts:
            pnt.rotate(alpha, rx0, ry0)        
    
    def xyToPnts(x, y):
        
        coord = []
        
        for i in range(len(x)):
            coord.append(Point(x[i], y[i]))
            
        coord = np.array(coord)        
        return coord
    
    def getXY(self):
        
        x = []
        y = []
        
        for i in range(len(self.coord)):
            x.append(self.coord[i].x)
            y.append(self.coord[i].y)
        
        x = np.array(x)
        y = np.array(y) 
        
        return x, y
    
    def getCoord(self):
        
        x, y = self.getXY()
        coords = np.stack([x, y], axis = 1)       
        return coords
        
    def getPolyArea(self):
        
        area = np.zeros(len(self.coord))
        
        for i in range(len(self.coord) - 2):
            l1 = Line(self.coord[i+1], self.coord[0])
            l2 = Line(self.coord[i+2], self.coord[i+1])
            area[i] = abs(l1.crossProduct(l2))
            
        S = sum(area)       
        return S
    
    def savetxt(self):
        
        filename = input('Please specify the filename: \n')
        coords = self.getCoord()
        np.savetxt('%s.txt' %filename, coords, fmt = '%.4f', \
                   header = '# %s%s \n x\t y' %(self.__class__.__name__, self.name))
            
    def getId(self):
        return self.id
    
    def getName(self):
        return self.name
    
    def setName(self,newName = ''):
        self.name = newName
        
    def plotCurve(self):
        
        x, y = self.getXY()
        
        # set the first point to be the last one as well to close the curve
        x = np.insert(x, -1, x[0])
        y = np.insert(y, -1, y[0])
        
        plt.plot(x, y, label = '%s' %(self.name))
        
        plt.legend(loc = 'best')
        
    def plotMultiCurves(self, *curves):
        
        plt.figure(figsize = (6,6))
        
        self.plotCurve()
        
        for curve in curves:
            curve.plotCurve()
            
        plt.tight_layout()
                
    def printMethod(self):
        
        coords = self.getCoord()
        
        print('Curve%i %s\n' %(self.id, self.name), '(x, y) = \n', coords )
        
    def printMultiMethod(self, *curves):
         
        self.printMethod()
        
        for curve in curves:
            curve.printMethod()

    def readFile(file_path):
        # comment line starts with '#'
        coord = np.genfromtxt(file_path, dtype = float, comments = '#') 
        coord = Curve.xyToPnts(coord[:, 0], coord[:, 1])
        cur   = Curve(coord)
        return cur


class Airfoil(Curve):
    
    """
    Return an airfoil class with its geometric properties.  
    
    Bezier and CST methods are embedded for future parametric representation.
    
    Interface with the airfoil analysis is built. 
    Currently, xfoil is used to obtain the airfoil polar.
    
    Plot methods are provided.
    
    Examples:
        >>> file_dir = r'E:\\work\\2020 8 16\\mfoil\\DU\\DU00-W2-350.txt'
        >>> coord = np.genfromtxt(file_dir, skip_header = 1)
        >>> coord = np.array(coord)
        >>> af = Airfoil(coord)
        
    Grahpic Illustration:
        >>> af.plot()
        
    """
    
    afId = 1     # airfoil id
    
    def __init__(self, coord, Re = '', Ma = '', name = '', is_norm = False):
        

        """
        Parameters
        ----------
        coord : array of object
            Airfoil coordinates.
        Re    : float, optional
            Reynolds number.
        Ma    : float, optional
            Mach number.
        name  : str, optional
            Airfoil name.
            
        """      
        self.coord = coord
        self.Re    = Re
        self.Ma    = Ma
        self.name  = name
        self.id    = Airfoil.afId
        self.is_norm = is_norm
        Airfoil.afId = Airfoil.afId + 1
        if self.is_norm:
            self.normalize()
    
    def getChordBasedPara(self):
        
        """
        Returns
        -------
        samples : array_like
             Chord length based parameterized values of coordinate points.
        """
        self.s  = self.getS()
        self.t = np.cumsum(self.s)/np.sum(self.s)
        
        return self.t
    
    def getSlope(self):
        
        """
        Returns
        -------
        samples : array_like
            Slope of fitting curve using the cubic spline      
        """
        
        """This function is based on the cubic spline , which is different from 
        the linear line in the Curve class"""
        
        self.getChordBasedPara()
        
        x, y = self.getXY()
        
        csXt = interpolate.CubicSpline(self.t, x, bc_type = 'natural')
        csYt = interpolate.CubicSpline(self.t, y, bc_type = 'natural')
        
        xtDer1 = csXt(self.t, 1)
        ytDer1 = csYt(self.t, 1)
        
        self.slope = ytDer1 / xtDer1
        
        return self.slope
        
    
    def getAngle(self):
    
        """
        Returns
        -------
        samples : array_like, [deg]
            Intersection angle of the tangent line and x-axis.      
        """
        
        """This function is based on the cubic spline , which is different from 
        the lienar line in the Curve class"""
    
        self.getSlope()

        self.angle = np.arctan(self.slope) * 180 / np.pi
        
        return self.angle
    
    def getCurvature(self):
        
        """
        Returns
        -------
        samples : array_like
            Introducing the cubic spline to represent the airfoil curve. 
        """
        
        self.getChordBasedPara()
        
        csXt = interpolate.CubicSpline(self.t, self.coord[:,0], bc_type = 'natural')
        csYt = interpolate.CubicSpline(self.t, self.coord[:,1], bc_type = 'natural')
        
        xtDer1 = csXt(self.t, 1)
        ytDer1 = csYt(self.t, 1)
    
        xtDer2 = csXt(self.t, 2)
        ytDer2 = csYt(self.t, 2)
        
        yxDer2 = (xtDer1 * ytDer2 - ytDer1 * xtDer2) / xtDer1 ** 3
        
        self.cur = np.sign(yxDer2) * abs(xtDer1 * ytDer2 - ytDer1 * xtDer2) \
                      / (xtDer1**2 + ytDer1**2) ** (3/2)    
        
        return self.cur
    
    def getLe(self):
        
        """
        Returns
        -------
        samples : array_like
            Introducing the cubic spline to estimate the leading edge point. 
        """
        
        self.getChordBasedPara()
        
        x, y = self.getXY()
        
        indexMinx = np.argmin(x)    
        
        csXt = interpolate.CubicSpline(self.t, x, bc_type = 'natural')

        func = lambda t: csXt(t)
        
        #golden search method to find the minimum x value
        tLe = golden(func, brack = (self.t[indexMinx-1], self.t[indexMinx], \
                                    self.t[indexMinx + 1]),\
               tol = 1e-4)
        
        xLe = csXt(tLe)
        
        if abs(xLe - x[indexMinx]) < 1e-6:
            
            self.le = Point(x[indexMinx], y[indexMinx])

        elif  abs(xLe - x[indexMinx + 1]) < 1e-6:
            
            self.le =Point(x[indexMinx + 1], y[indexMinx + 1])
            
        else:
        
            yLe = interpolate.CubicSpline(self.t, y, bc_type = 'natural')(tLe)
            
            self.le = Point(xLe, yLe)
                               
        self.le.name = 'leading edge point'   
        
        return self.le
    
    def getTe(self):
        
        xTe = (self.coord[0].x + self.coord[-1].x) / 2
        yTe = (self.coord[0].y + self.coord[-1].y) / 2
        
        self.te = Point(xTe, yTe, name = 'trailing edge point')
        
        return self.te
    
    def getTeThick(self):
        """
        Returns
        -------
        teThic : float, [-]
            trailing edge thickness 
        """
        teThick = self.coord[0].distance(self.coord[-1])
        return teThick
    
    def getTeAngle(self):
        """
        Returns
        -------
        teAngle : float, [deg]
            Intersection angle of tangential lines at upper and lower trailing edge point 
        """
        angle = self.getAngle
        teAngle = abs(angle[-1] - angle[0])
        
        return teAngle

    def getChord(self):
        
        """
        Returns
        -------
        samples : float
            Distance between the leading edge point and trailing edge point. 
        """
        
        self.getLe()
        
        self.getTe()
        
        self.c = self.le.distance(self.te)
        
        return self.c  
    
    def getThick(self):
        
        """
        Returns
        -------
        samples : array_like
            Introducing the cubic spline to estimate the thickness. 
        """
        
        coord = self.getCoord()
        
        self.getLe()
        self.getTe()
        
        indexMinx = np.argmin(coord[:,0])
        
        # if self.le[0] not in coord[:,0]:
        if self.le.x not in coord[:,0]:
 
            coord = np.concatenate((coord[:indexMinx,:], 
                                   np.array([[self.le.x, self.le.y]]), 
                                   coord[indexMinx:,:]))
            
        csUp = interpolate.CubicSpline(coord[indexMinx::-1,0], \
                                            coord[indexMinx ::-1,1], \
                                                bc_type = 'natural')
        csLow = interpolate.CubicSpline(coord[indexMinx:,0], \
                                              coord[indexMinx:,1], \
                                                  bc_type = 'natural')
                
        xInter = np.linspace(self.le.x, self.te.x, 1000)
        
        yUp = csUp(xInter)
        
        yLow = csLow(xInter)
        
        yMidLine = (yUp + yLow) / 2
    
        self.maxThick = np.max(yUp - yLow) * 100    # maximum thickness * 100
        
        self.xMaxThick = xInter[np.argmax(yUp - yLow)]    # abscissa of maximum thickness location
        
        self.MidLine = np.array([xInter, yMidLine]).T   # mid-line
        
        self.thick = yUp - yLow
                
    def normalize(self):
        
        """
        Returns
        -------
        samples : array_like
            The normalized coordinates follows three rules:
                1. unit chord
                2. LE point is (0,0)
                3. TE point sits on x-axis
        """
        # make unit chord
        self.getChord()
        
        ratio = 1 / self.c
        
        self.scale(ratio, ratio)
        
        # move the LE point to (0,0) and move the TE point to x-axis
        
        self.getLe()
        self.getTe()

        alpha = self.le.getAngle(self.te)
        
        self.move(-self.le.x, -self.le.y)
        
        self.rotate(-alpha)
           
    def setTeThick(self, teThick):
        
        self.teThick = teThick
        
    def setMaxThick(self, maxThick):
 
        self.maxThick =  maxThick
            
    def setXY(self, newCoord):        
        
        self.coord = newCoord
        
    def setAeroCondition(self, Re, Ma):
        
        self.Re = Re
        self.Ma = Ma
    
    def savetxt(self):
        
        filename = input('Please specify the filename: \n')
        coord = self.getCoord()
        np.savetxt('%s.txt' %filename, coord, fmt = '%.4f', \
                   header = '# %s%s \n x\t y' %(self.__class__.__name__, self.name))
            
    def getStructuralProperty(self, sType):
        
        """       
        Input:
            sType :   int
                    The structural property has two types:
                        1: solid 
                        2: skin 
        """
        
        self.getThick()

        x, y = self.getXY()
        
        x = np.insert(x, len(x), x[0])
        y = np.insert(y, len(y), y[0])
        
        
        dx = np.diff(x)
        dy = np.diff(y)
        ds = (dx * dx + dy * dy) ** (1/2)
        xAvg = dx / 2 + x[:-1]
        yAvg = dy / 2 + y[:-1]
        
        xDy = sum(dy * xAvg)
        xxDy = sum(dy * xAvg ** 2)
        xxxDy = sum(dy * xAvg ** 3)
        xDs = sum(ds * xAvg)
        xxDs = sum(ds * xAvg ** 2)
        
        yDx = sum(dx * yAvg)
        yyDx = sum(dx * yAvg ** 2)
        yyyDx = sum(dx * yAvg ** 3)
        yDs = sum(ds * yAvg)
        yyDs = sum(ds * yAvg ** 2)
        
        # xMin = np.min(x)
        # xMax = np.max(x)
        # yMin = np.min(y)
        # yMax = np.max(y)
        
        area = -yDx
        sLen = sum(ds)
        
        if (area == 0):
            return
        
        xc = xxDy / (2 * xDy)
        yc = yyDx / (2 * yDx)
        
        if sType == 1:
            eixx = -yyyDx / 3 + yyDx * yc - yDx * yc **2
            eiyy =  xxxDy / 3 - xxDy * xc + xDy * xc **2
            iMid = np.argmin(x)
            aj = 0
            for i in range(1, iMid + 1):
                xmid = (x[i] + x[i-1])/2
                ymid = (y[i] + y[i-1])/2
                dx   = x[i - 1] - x[i]
                
                if xmid >= x[-1]:
                    yopp = y[-1]
                    aj = aj + abs(ymid - yopp) ** 3 * dx / 3.0
                    break
                
                if xmid <= x[iMid]:
                    yopp = y[iMid]
                    aj = aj + abs(ymid - yopp) ** 3 * dx / 3.0
                    break
                
                for j in range(len(x) - 2 , iMid - 1, -1):
                    if (xmid > x[j - 1] ) and (xmid <= x[j] ):
                        frac = (xmid - x[j-1]) / (x[j] - x[j-1])
                        yopp = y[j-1] + (y[j] - y[j-1]) * frac
                        aj = aj + abs(ymid - yopp) ** 3 * dx / 3.0
                        break     
                    
            self.struct = np.array([eixx, eiyy, aj])
             
        elif sType == 2:
            eiyyt = xxDs - xDs * xc * 2 + sLen * xc **2
            eixxt = yyDs - yDs * yc * 2 + sLen * yc ** 2
            ajt = 4.0 * area ** 2 / sLen
            self.struct = np.array([eixxt, eiyyt, ajt])
        else:
            return print("select again: 1: solid; 2: skin")
            
        return self.struct
    
    def blend_geo(self, another, thick_target, num = 400):
        
        from scipy.optimize import brentq
        
        def func_blend(ratio, coord_self, coord_other, thick_target):
            
            def mix(curve1, curve2, ratio):
 
                x_merge = np.union1d(curve1[:, 0], curve2[:, 0])
                y_merge = interp1d(curve1[:, 0], curve1[:, 1], kind='cubic', fill_value="extrapolate")(x_merge) * (1 - ratio) \
                        + interp1d(curve2[:, 0], curve2[:, 1], kind='cubic', fill_value="extrapolate")(x_merge) * ratio
 
                return np.c_[x_merge, y_merge]
                    
            indexMinx_self = np.argmin(coord_self[:,0])
            indexMinx_other = np.argmin(coord_other[:,0])

            pts_upper = mix(coord_self[:indexMinx_self + 1], coord_other[:indexMinx_other + 1], ratio)
            pts_lower = mix(coord_self[indexMinx_self:],     coord_other[indexMinx_other:], ratio)
            
            coord_blend = np.r_[pts_upper[:1:-1], pts_lower]
            coord_blend  = Curve.xyToPnts(coord_blend[:,0], coord_blend[:,1])
            af_blend     = Airfoil(coord_blend, is_norm = True)
            af_blend.getThick()
            
            return af_blend, af_blend.maxThick - thick_target
        
        def errf_blend(ratio, *args):
            
            af_blend, error = func_blend(ratio, *args)
            
            return error
        
        def pick_arange(arange, num):
            if num > len(arange):
                raise ValueError("number exceed array length")
            else:
                output =[]
                seg = len(arange) / num
                for n in range(num):
                 if int(seg * (n+1)) >= len(arange):
                  output.append(arange[-1,:])
                 else:
                  output.append(arange[int(seg * n)]) 
                return np.array(output)
        
        if not self.is_norm:
            self.normalize()
        if not another.is_norm:
            another.normalize() 
        
        self.getLe()
        another.getLe()
            
        # add (0,0) to represent the leading edge point
        coord_self = self.getCoord()
        indexMinx = np.argmin(coord_self[:,0])
        if self.le.x not in coord_self[:,0]:
 
            coord_self = np.concatenate((coord_self[:indexMinx,:], 
                               np.array([[self.le.x, self.le.y]]), 
                               coord_self[indexMinx:,:]))
            
        coord_other = another.getCoord()
        indexMinx = np.argmin(coord_other[:,0])
        if another.le.x not in coord_other[:,0]:
 
            coord_other = np.concatenate((coord_other[:indexMinx,:], 
                               np.array([[another.le.x, another.le.y]]), 
                               coord_other[indexMinx:,:]))
        
        if coord_self is None or coord_other is None:
            raise BaseException("airfoil doesn't have coords!")

        args = (coord_self, coord_other, thick_target)
        
        blend_ratio = brentq(errf_blend, 0.0, 1.0, args = args, 
                             xtol = 1e-2, maxiter = 500)
        
        af_blend, _ = func_blend(blend_ratio, *args)
        
        coords_blend = af_blend.getCoord()

        if len(coords_blend) > num:
            coords_blend = pick_arange(coords_blend, num)
            
        coords_blend  = Curve.xyToPnts(coords_blend[:, 0], coords_blend[:, 1])
        af_blend.setXY(coords_blend)
        
        return af_blend
    
    
    def init_from_file(file_path, Re = '', Ma = '', name = ''):
        # comment line starts with '#'
        coord = np.genfromtxt(file_path, dtype = float, comments = '#')
        
        if coord[0,1] < coord[-1,1]:   # coordinates start from upper TE to LE then back to lower TE
            coord = coord[::-1]
        
        coord = Curve.xyToPnts(coord[:, 0], coord[:, 1])
        return Airfoil(coord, Re, Ma, name)

