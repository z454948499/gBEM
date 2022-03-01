# -*- coding: utf-8 -*-
"""
@author: GoldWind Blade Development Team
"""

import numpy as np
import bisect
from numpy import sin, cos

def quadratic_interpolation(X, x1, x2, x3, y1, y2, y3):
    # quadratic Lagrange interpolation to obtain a smooth curve that is 
    # differentiable at all values.
    
    L1 = (X - x2) * (X - x3) / ((x1 - x2) * (x1 - x3))
    L2 = (X - x1) * (X - x3) / ((x2 - x1) * (x2 - x3))
    L3 = (X - x1) * (X - x2) / ((x3 - x1) * (x3 - x2))
    return L1 * y1 + L2 * y2 + L3 * y3

def dydx_spline(x1, x2, x3, y1, y2, y3):
    # calculates dy/DX at x = x2 by parabolic spline interpolation

    return ( -y1 * (x2 - x3)**2 + y2 * \
            (-x1 * x1 + x3 * x3 + 2 * x2 * \
           (x1 - x3)) + y3 * (x2 - x1)**2) / ( x3 * x3 * (x2 - x1) + x1 * x1 * \
                                                (x3 - x2) + x2 * x2 * \
                                                (x1 - x3) )  

def cubic_spline(X, x1, x2, x3, x4, y1, y2, y3, y4):
    # cubic spline interpolation to obtain A smooth curve that is 
    # differentiable at all values

    dydx2 = dydx_spline(x1, x2, x3, y1, y2, y3)
    dydx3 = dydx_spline(x2, x3, x4, y2, y3, y4)
    C = 1 / (x3 - x2)
    return  ( y2 * (C * (X - x3))**2 * (1 + 2 * C * (X - x2)) + y3 * \
       (C * (X - x2))**2 * (1 - 2 * C * (X - x3)) + dydx2 * ( \
            C * (X - x3))**2 * (X - x2) + dydx3 * (C * (X - x2))**2 * (X - x3) )

def interpolate_vector(x, x1, y1):
    
    if x <= x1[0]:
        return y1[0]
    elif x >= x1[-1]:
        return y1[-1]
    elif len(x1) == 2:
        return np.interp(x, x1, y1)
    elif x <= x1[1]:
        return quadratic_interpolation(x,x1[0],x1[1],x1[2],y1[0],y1[1],y1[2])
    elif x >= x1[-2]:
        return quadratic_interpolation(x,x1[-3],x1[-2],x1[-1],y1[-3],y1[-2],y1[-1])
    else:
        loc = bisect.bisect(x1, x)    # index that is just above x in x1
        return cubic_spline(x, x1[loc-2], x1[loc-1], x1[loc],x1[loc+1], 
                          y1[loc-2], y1[loc-1], y1[loc], y1[loc+1])

# offset are neglected and the quantities are of independent of positions here
# forces, velocity are independent of positions but moments are not    
def wind_to_yaw(x, yaw):
    # yaw: positive when rotating about +z axis(height direction)
    # x  : 1*3 array
    yaw = np.radians(yaw)
    cy, sy = cos(yaw), sin(yaw)
    
    w2y_matrix = np.array([[ cy, -sy, 0],
                           [ sy,  cy, 0],
                           [  0,  0,  1]])
    return np.dot(x, w2y_matrix)

def yaw_to_hub(x, tilt):
    # tilt: positive when rotor tilts up for upwind and down for downwind machine
    # x  : 1*3 array
    tilt = np.radians(tilt)
    ct, st = cos(tilt), sin(tilt)
    
    y2h_matrix = np.array([[ ct, 0, st],
                           [  0, 1,  0],
                           [-st, 0, ct]])
    return np.dot(x, y2h_matrix)

def hub_to_azimuth(x, azimuth):
    # azimuth: positive when rotating about the +x axis
    # x  : 1*3 array
    azimuth = np.radians(azimuth)
    ca, sa = cos(azimuth), sin(azimuth)
    
    h2a_matrix = np.array([[ 1,  0,   0],
                           [ 0, ca, -sa],
                           [ 0, sa,  ca]])
    return np.dot(x, h2a_matrix)

def azimuth_to_blade(x, cone):
    # cone: positive when rotating about the -y axis, 
    #       tilts toward to nacelle/tower for upwind machine
    cone = np.radians(cone)
    cc, sc = cos(cone), sin(cone)
    
    a2b_matrix = np.array([[ cc, 0, -sc],
                           [  0, 1,  0],
                           [ sc, 0,  cc]])
    return np.dot(x, a2b_matrix)

def blade_to_airfoil(x, pitch):
    # blade algined to airfoil aligned
    # pitch: positive whn rotating about the -z axis, typically twist + pitch
    #        causes the angle of attack to 
    pitch = np.radians(pitch)
    cp, sp = cos(pitch), sin(pitch)
    
    b2a_matrix = np.array([[cp,  sp, 0],
                           [-sp, cp, 0],
                           [ 0,   0, 1]])
    
    return np.dot(x, b2a_matrix)

def integration(var, s):
    
    """
    Integration from tip to hub.
    
    Parameters
    ----------
    var          :   array_like
        variables to be integrated
    s            :   array_like
        the vector along which the variables are integrated
        
    Returns
    -------
    var_s[0]     :   float
        integration result at the hub element
    var_s        :   array_like
        integration results for each element
        
    Notes
    -----
    The integration of variables are carried out along the s axis.
    
    For instance, in order to compute the blade area, it can be called:        
        area, _ = integration(c,    s)
    Instead, when the blade moment is to be computed after solving BEM equations,
    the following line can be called: 
        moment_at_hub, moment_along_blade = integration(B * pn * z_az, s)
    
    """

    NS  = len(s)
    var_s = np.zeros(NS)
    
    dx1 =  s[-2] - s[-3]
    dx2 =  s[-1] - s[-2]
    l1  =  var[-3] / dx1 / (dx1 + dx2)  
    l2  = -var[-2] / dx1 /  dx2
    l3  =  var[-1] / dx2 / (dx1 + dx2)
    aa  =  l1 + l2 + l3
    bb  = -l1 * dx2 + l2 * (dx1 - dx2) + l3 * dx1
    cc  = -l2 * dx1 * dx2
    var_s[NS-2] = (aa / 3 * (dx2)**2 + bb / 2 * dx2 + cc) * dx2
       
    i = NS - 3
    while i >= 1:
        dx = s[i+1] - s[i]
        dydx2 = 0
        
        if i == 1:
            dydx20 = dydx_spline(s[i - 1], s[i], s[i + 1],
                                    var[i - 1], var[i], var[i + 1])
        else:
            dydx20 = dydx2
            
        dydx30 =  dydx_spline(s[i], s[i+1], s[i+2], var[i],var[i+1],var[i+2])
    
        var_s[i] = var_s[i+1] + (dx*(dydx20 - dydx30)/12 + (var[i] + var[i+1])/2) * dx
        
        i = i - 1
    
    dx1 = s[1] - s[0]
    dx2 = s[2] - s[1]        
    l1  = var[0] / dx1 / (dx1 + dx2)
    l2  =-var[1] / dx1 / dx2
    l3  = var[2] / (dx1 + dx2) / dx2
    aa  = l1 + l2 + l3
    bb  =-l1 * (2 * dx1 + dx2) - l2 * (dx1 + dx2) - l3 * dx1
    cc  = l1 * dx1 * (dx1 + dx2)
    var_s[0] = var_s[1] + (aa/3 * (dx1)**2 + bb / 2 * dx1 + cc) * dx1                
    
    return var_s[0], var_s