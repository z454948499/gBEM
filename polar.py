# -*- coding: utf-8 -*-
"""
@author: GoldWind Blade Development Team
"""

from prep import quadratic_interpolation, cubic_spline
import numpy as np
from numpy import sin, cos, pi
from scipy.interpolate import UnivariateSpline as US
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import bisect


class Polar(object):
        
    def __init__(self, alpha, cl, cd, cm, relThk = None, Re = None, Ma = None):
        
        """
        Parameters:
        -------------
        alpha  :   array_like, [deg]
            angle of attack
        cl     :   array_like, [-]
            lift coefficient
        cd     :   array_like, [-]
            drag coefficient
        cm     :   array_like, [-]
            drag coefficient
        relThk :   float, [-]
            relative thickness
        Re     :   float, [-]
            Reynold numebr
        Mach   :   float, [-]
            Mach number
        -------------
        """

        self.alpha                = np.radians(alpha)    # convert degree to radian
        self.cl, self.cd, self.cm = cl, cd, cm
        self.Re, self.Ma          = Re, Ma
        self.relThk               = relThk
    
    @classmethod
    def init_from_file(cls, fname, relThk = None, Re = None, Ma = None):
        """
        Initilization from given airfoil data file
        
        Parameters
        ----------
        fname   : str
            absoulte path of the airfoil data file    
        relThk  : float, [-]
            relative thickness
        Re      : float, [-]
            Reynold number
            
        Returns
        -------
        polar : Polar 
            a Polar object
        """
        dat   = np.genfromtxt(fname, comments = '#')
        alpha = dat[:, 0]
        cl    = dat[:, 1]
        cd    = dat[:, 2]
        if dat.shape[1] > 3:
            cm    = dat[:, 3]
        else:
            cm    = np.zeros_like(alpha)
   
        return cls(alpha, cl, cd, cm, relThk, Re, Ma)
    
    def get_alfa0(self, alpha_linear_min = -5, alpha_linear_max = 5):
        # polyfit AoA ranging from [-5, 5] to evaluate the zero lift angle of attack
        
        alpha_linear_min = np.radians(alpha_linear_min)
        alpha_linear_max = np.radians(alpha_linear_max)
        
        id_linear = np.logical_and(self.alpha >= alpha_linear_min,
                     self.alpha <= alpha_linear_max)
        
        linear_fit= np.polyfit(self.alpha[id_linear], self.cl[id_linear], 1)
        
        slope     = linear_fit[0]
        intercept = linear_fit[1]
        
        # use slope at zero lift angle as the ideal slope
        if slope == 0.0:
            alfa0 = 0.0
        else:
            alfa0 = -intercept / slope
        
        return alfa0
 
    def get_clmax(self):
        # calcualtes the max cl and its angle
        alpha = self.alpha
        cl    = self.cl
        alpha_up = alpha[alpha > 0]
        cl_up    = cl[alpha > 0]
        
        idx_up = np.logical_and(cl_up[1:-1] - cl_up[:-2]>= 0, cl_up[1:-1] - cl_up[2:]>= 0)  # compare three consecutive cl and return the boolean of the maximum among them  
        idx_up = np.logical_and(idx_up,cl_up[1:-1] - np.roll(cl_up,-1)[2:]>= 0)   # compare and find the index of cl[i] - cl[i+2] > 0
        if idx_up.any():
            cl_up_extrema = cl_up[1:-1][idx_up][0]
            aoa_extrema = alpha_up[1:-1][idx_up][0]
        else:
            cl_up_extrema = cl_up[-1]
            aoa_extrema = alpha_up[-1]
            
        return cl_up_extrema, aoa_extrema
        
    def get_clmin(self):
        # calulates the min cl and its angle
        
        alpha = self.alpha
        cl    = self.cl
        alpha_low = alpha[alpha < 0]
        cl_low    = cl[alpha < 0]
 
        idx_low = np.logical_and(cl_low[1:-1] - cl_low[:-2]<= 0, cl_low[1:-1] - cl_low[2:]<= 0)
        idx_low = np.logical_and(idx_low,cl_low[1:-1] - np.roll(cl_low, 1)[:-2]<= 0.0)
        if idx_low.any():  
            cl_low_extrema = cl_low[1:-1][idx_low][-1]
            aoa_extrema = alpha_low[1:-1][idx_low][-1]
        else:
            cl_low_extrema = cl_low[0]
            aoa_extrema = alpha_low[0]
            
        return cl_low_extrema, aoa_extrema
    
    def correction3D(self, c_over_r = None, a = 2.2, h = 1.5,
                     alpha_linear_min = 0 , alpha_linear_max = 3 , 
                     alpha_corr_min = 0.0, alpha_corr_max = 50.0):
        
        """
        Corrects the 2-D Cl and Cd airfoil data by using some of the ideas 
        presented by P. K. Chaviaropoulos and M. O. L. Hansen. 
        
        Parameters
        ----------
        c_over_r   : float, [-]
            chord over radius ratio   
        a,h        : constant, [-]
            constants within the 3D correction
        alpha_linear_min  : float, [deg]
            minimum angle of attack to evaluate inviscid lift slope   
        alpha_linear_min  : float, [deg]
            minimum angle of attack to evaluate inviscid lift slope 
        alpha_corr_min    : float, [deg]
            minimum angle of attack to use the 3D correction
        alpha_corr_max    : float, [deg]
            maximum angle of attack to use the 3D correction
        
        Notes
        --------
        The basic formula are:
            
                Cx_3D = Cx_2D + a * (c/r)^h * ΔCx		x = l, d
    
                ΔCl = Cl_inv - Cl_2D	ΔCd = Cd_2D – Cd_2Dmin	 
                
                Cl_inv = 2 * pi * sin((alpha_deg - alpha0_deg)*pi/180)
    
                c = local chord
                r = local radius
                a = constant 	(estimated to 2.2)
                h = constant 	(estimated to 1.5)
                
        The codes here will follow the common practice of building the airfoil data 
        table, which is to do the 2D extrapolation first and then the 3D correction 
        if necessary.
        
        For further improvement, a full 3D CFD simulation should be implemented.         
                        
        """
        alpha_linear_min = np.radians(alpha_linear_min)
        alpha_linear_max = np.radians(alpha_linear_max)
        alpha_corr_min   = np.radians(alpha_corr_min)
        alpha_corr_max   = np.radians(alpha_corr_max)
          
        if c_over_r is None:
            raise TypeError('__init__() missing required argument chord over \
                            radius ratio ')
                            
        id_linear = np.logical_and(self.alpha >= alpha_linear_min, \
                     self.alpha <= alpha_linear_max)
        linear_fit= np.polyfit(self.alpha[id_linear], \
                               self.cl[id_linear], 1)
        alfa0     = -linear_fit[1] / linear_fit[0]
        
        id_alp    = np.logical_and(self.alpha >= alpha_corr_min, \
                                self.alpha <= alpha_corr_max)
        id_alp    = np.nonzero(id_alp)

        cl_2D_inv = 2 * pi * sin((self.alpha[id_alp] - alfa0))
        f_cl      = cl_2D_inv - self.cl[id_alp]
        
        cdmin     = np.min(self.cd)     # no interpolation, or seen as linear interpolation 
        f_cd      = self.cd[id_alp] - cdmin
             
        adj       = -0.035 * (self.alpha[id_alp]  - alpha_corr_max) + 0.15
        adj[np.nonzero(adj > 1)] = 1
        
        self.cl[id_alp] = self.cl[id_alp] + adj * a * (c_over_r)**h * f_cl
        self.cd[id_alp] = self.cd[id_alp] + adj * a * (c_over_r)**h * f_cd
                            
    def extrapolate_Thomas(self, AR = None):
        """
        extrpolates the Aoa to +/- 180 degrees with applying Viterna's method 
        within alpha_high to 90 degrees.
        
        Parameters
        ----------
        AR   : float, optional, [-]
            aspect ratio
        
        Notes
        ---------
        The extrapolation comes from the observation of the experimental results.
        Thus, fixed values and Aoa are provided beyond the experimental results.
        
        If AR is provided, the maximum drag coefficient is computed by:

        cdmax = 1.11 + 0.018*AR     AR < 50
        cdmax = 2.01                AR >= 50
        
        If extrapolation is given for Aoa beyond 90 degrees, the method should be 
        adjusted. 
        
        """
        def Viterna(B1, A2, B2, alpha, cladj):
            
            alpha = np.maximum(alpha, 0.00001)  
            # prevent denominator being zero
            CL = B1/2 * sin(2*alpha)  + A2 * (cos(alpha))**2 / sin(alpha)
            CD = B1 * (sin(alpha))**2 + B2 * cos(alpha)
            CL = CL * cladj
            
            return CL, CD
        
        if AR is None:
            cdmax = max(max(self.cd), 2.01)      # make sure the codes will not return error when AR is None
        elif AR >= 50:
            cdmax = 2.01
        elif AR < 50 :
            cdmax = 1.11 + 0.018 * AR
            
        if cdmax < 1.11:
            raise ValueError ('cdmax must be larger than 1.11')
        cdmax =  max(max(self.cd),cdmax) 
        
        # fix values at alpha = 25: cl = 1.2, cd = 0.35 
        cd_high    = 0.35     
        alpha_high = np.radians(25)
        cl_high    = 1.2
        
        alpha_high_r = alpha_high
        B1 = cdmax
        B2 = (cd_high - cdmax * (sin(alpha_high_r))**2)/cos(alpha_high_r)
        A2 = (cl_high - cdmax*sin(2*alpha_high_r)/2)*sin(alpha_high_r) \
            /(cos(alpha_high_r))**2
            
        # alpha_high <-> 90
        alpha1   = np.linspace(30,90,7)
        alpha1   = np.radians(alpha1)
        cl1, cd1 = Viterna(B1, A2, B2, alpha1, 1.0)
        cl1[-1]  = 0.1    # set cl(a = 90) = 0.1
        
        # 100 <-> 180
        alpha2   = np.linspace(100,180,9)
        alpha2   = np.radians(alpha2)
        cl2      = [-0.16,-0.415,-0.625,-0.78,-0.9,-0.9,-0.65,-0.8, -0.1]
        cd2      = [    0,     0,     0,    0,   0,   0,  0.3, 0.1,0.025]
        cd2[0:6] = cd1[-2::-1]
        
        # -30 <-> -180
        alpha3    = np.linspace(-180,-30,16)
        alpha3    = np.radians(alpha3)
        cl3       = [-0.1, 0.6, 0.35, 0.69, 0.8, 0.75, 0.625, 0.375, 0.125, -0.135, -0.385, -0.65, -0.82, -0.885, -0.885, -0.75]
        cd3       = [0.025,0.1,0.3,0,0,0,0,0,0,0,0,0,0,0,0,0]
        cd3[-1:-8:-1]  = cd1 
        cd3[-8:-14:-1] = cd2[0:6]
        
        # concatenation to get full 360 degrees data
        alpha = np.concatenate((alpha3, self.alpha, alpha1, alpha2))
        cl    = np.concatenate((cl3, self.cl, cl1, cl2))
        cd    = np.concatenate((cd3, self.cd, cd1, cd2))
        
        self.cl    = cl
        self.cd    = cd
        self.alpha = alpha
        
        # extrapolation of CM
        # -180 <-> -30 , shares the same angle of attack with cl and cd
        cm_neg = [-0.1,0.3,0.22,0.3602,0.4323,0.4581,0.4600,0.4556,0.4422,\
                  0.42,0.3889,0.3489,0.3,0.2422,0.18,0.1]
        # mirror of the negative angle of attacks
        cm_pos = -np.array(cm_neg[::-1])
        cm_pos[-1] = -cm_pos[-1]           # ensure cm is continuous
        cm = np.concatenate((cm_neg,self.cm,cm_pos))
        self.cm = cm
    
    def smooth(self):
        
        us_cl = US(self.alpha, self.cl, s = 0.1)
        self.cl = us_cl(self.alpha)
        us_cd = US(self.alpha, self.cd, s = 0.001)
        self.cd = us_cd(self.alpha)
        us_cm = US(self.alpha, self.cm, s = 0.0001)
        self.cm = us_cm(self.alpha)

    def blend_polar(self, another, weight, blend_method):
        
        def blend_scaled(self, another, weight):
            
            """
            linear blend of aerodynamic coefficients with another airfoil with 
            the given weight.
            
            Parameters
            ----------
            another    :    Polar
                another Polar object    
            weight     :    float 
                blending ratio
                
            Returns
            -------
            polar : Polar 
                a Polar object
                
            Notes
            -----
            The alpha is given by merging the existing angles of attack for two polars.
            
            """
    
            # normalization
            alfa0 = self.get_alfa0()
            alfa0_other = another.get_alfa0()
            _, aoa_clmax = self.get_clmax()
            _, aoa_clmax_other = another.get_clmax()
    
            # for upper part
            blend_alfa0 = (1 - weight) * alfa0 + weight * alfa0_other
            blend_alfa_clmax = (1 - weight) * aoa_clmax + weight * aoa_clmax_other
            
            alpha_self = self.alpha        
            alpha_up = self.alpha[alpha_self > alfa0]
            cl_up    = self.cl[alpha_self > alfa0]
            cd_up    = self.cd[alpha_self > alfa0]
            cm_up    = self.cm[alpha_self > alfa0]
            
            alpha_norm_self = (alpha_up - alfa0) / (aoa_clmax - alfa0)
    
            alpha_another = another.alpha
            alpha_another_up = another.alpha[alpha_another > alfa0_other]
            cl_another_up    = another.cl[alpha_another > alfa0_other]
            cd_another_up    = another.cd[alpha_another > alfa0_other]
            cm_another_up    = another.cm[alpha_another > alfa0_other]
    
            alpha_norm_another = (alpha_another_up - alfa0_other) / (aoa_clmax_other - alfa0_other)
            
            alpha_merge_up = np.union1d(np.rad2deg(alpha_norm_self), np.rad2deg(alpha_norm_another))
            alpha_merge_up = np.radians(alpha_merge_up)
            
            alpha_min = max(alpha_norm_self.min(), alpha_norm_another.min())
            alpha_max = min(alpha_norm_self.max(), alpha_norm_another.max())
            
            id_alpha = np.logical_and(alpha_merge_up >= alpha_min, \
                                      alpha_merge_up <= alpha_max)
            alpha_merge_up = alpha_merge_up[id_alpha]
            
            cl_self_up = np.interp( alpha_merge_up,alpha_norm_self, cl_up)
            cd_self_up = np.interp( alpha_merge_up,alpha_norm_self ,cd_up)
            cm_self_up = np.interp( alpha_merge_up,alpha_norm_self ,cm_up)
            
            cl_other_up = np.interp( alpha_merge_up,alpha_norm_another,cl_another_up)
            cd_other_up = np.interp( alpha_merge_up,alpha_norm_another,cd_another_up)
            cm_other_up = np.interp( alpha_merge_up,alpha_norm_another,cm_another_up)
            
            cl_blend_up    = (1 - weight) * cl_self_up + weight * cl_other_up       
            cd_blend_up    = (1 - weight) * cd_self_up + weight * cd_other_up
            cm_blend_up    = (1 - weight) * cm_self_up + weight * cm_other_up
    
            alpha_blend_up = alpha_merge_up * (blend_alfa_clmax - blend_alfa0) + blend_alfa0
            
            # for lower part
            _, aoa_clmin = self.get_clmin()
            _, aoa_clmin_other = another.get_clmin()
            blend_alfa_clmin = (1 - weight) * aoa_clmin + weight * aoa_clmin_other
                    
            alpha_low = self.alpha[alpha_self <= alfa0]
            cl_low    = self.cl[alpha_self <= alfa0]
            cd_low    = self.cd[alpha_self <= alfa0]
            cm_low    = self.cm[alpha_self <= alfa0]
            
            alpha_norm_self = (alpha_low - alfa0) / (aoa_clmin - alfa0)
    
            alpha_another_low = another.alpha[alpha_another <= alfa0_other]
            cl_another_low    = another.cl[alpha_another <= alfa0_other]
            cd_another_low    = another.cd[alpha_another <= alfa0_other]
            cm_another_low    = another.cm[alpha_another <= alfa0_other]
    
            alpha_norm_another = (alpha_another_low - alfa0_other) / (aoa_clmin_other - alfa0_other)
                    
            alpha_merge_low = np.union1d(np.rad2deg(alpha_norm_self), np.rad2deg(alpha_norm_another))
            alpha_merge_low = np.radians(alpha_merge_low)
            
            alpha_min = max(alpha_norm_self.min(), alpha_norm_another.min())
            alpha_max = min(alpha_norm_self.max(), alpha_norm_another.max())
            
            id_alpha = np.logical_and(alpha_merge_low >= alpha_min, \
                                      alpha_merge_low <= alpha_max)
            alpha_merge_low = alpha_merge_low[id_alpha]
            
            cl_self_low = interp1d(alpha_norm_self, cl_low)(alpha_merge_low)
            cd_self_low = interp1d(alpha_norm_self, cd_low)(alpha_merge_low)
            cm_self_low = interp1d(alpha_norm_self, cm_low)(alpha_merge_low)
                   
            cl_other_low = interp1d(alpha_norm_another,cl_another_low)(alpha_merge_low)
            cd_other_low = interp1d(alpha_norm_another,cd_another_low)(alpha_merge_low)
            cm_other_low = interp1d(alpha_norm_another,cm_another_low)(alpha_merge_low)
                    
            cl_blend_low    = (1 - weight) * cl_self_low + weight * cl_other_low      
            cd_blend_low    = (1 - weight) * cd_self_low + weight * cd_other_low
            cm_blend_low    = (1 - weight) * cm_self_low + weight * cm_other_low
    
            alpha_blend_low = alpha_merge_low * (blend_alfa_clmin - blend_alfa0) + blend_alfa0
                   
            alpha_blend = np.concatenate((alpha_blend_low[::-1], alpha_blend_up))    # np.union1d gives the ascending returns
            cl_blend    = np.concatenate((cl_blend_low[::-1], cl_blend_up))
            cd_blend    = np.concatenate((cd_blend_low[::-1], cd_blend_up))
            cm_blend    = np.concatenate((cm_blend_low[::-1], cm_blend_up))
            
            if self.Re is not None and another.Re is not None:
                Re_blend = (1 - weight) * self.Re + weight * another.Re
            else:
                Re_blend = []
                
            if self.relThk is not None and another.relThk is not None:
                relThk_blend = (1 - weight) * self.relThk + weight * another.relThk
            else:
                relThk_blend = []
            
            return type(self)(np.rad2deg(alpha_blend), cl_blend, cd_blend, cm_blend, relThk_blend, Re_blend)
        
        def blend_direct(self, another, weight):
            """
            linear blend of aerodynamic coefficients with another airfoil with 
            the given weight.
            
            Parameters
            ----------
            another    :    Polar
                another Polar object    
            weight     :    float 
                blending ratio
                
            Returns
            -------
            polar : Polar 
                a Polar object
                
            Notes
            -----
            The alpha is given by merging the existing angles of attack for two polars.
            If moment coefficients are not known in advance, remember to set them as 0 
            so as to not raise error message.
            """
            # use the degree format to avoid the floating point error due to coverting from degree to radians
            # 'np.union1d' could suffer from the floating point error, which means the function itself should be improved
            alpha_merge = np.union1d(np.rad2deg(self.alpha), np.rad2deg(another.alpha))
            alpha_merge = np.radians(alpha_merge)
            
            alpha_min = max(self.alpha.min(), another.alpha.min())
            alpha_max = min(self.alpha.max(), another.alpha.max())
            
            id_alpha = np.logical_and(alpha_merge >= alpha_min, \
                                      alpha_merge <= alpha_max)
            alpha = alpha_merge[id_alpha]
            
            cl_self = np.interp(alpha,self.alpha,self.cl)
            cd_self = np.interp(alpha,self.alpha,self.cd)
            cm_self = np.interp(alpha,self.alpha,self.cm)
            
            cl_other = np.interp(alpha,another.alpha,another.cl)
            cd_other = np.interp(alpha,another.alpha,another.cd)
            cm_other = np.interp(alpha,another.alpha,another.cm)
            
            cl_blend = (1 - weight) * cl_self + weight * cl_other
            cd_blend = (1 - weight) * cd_self + weight * cd_other
            cm_blend = (1 - weight) * cm_self + weight * cm_other
            
            # no fundamental theory for Re blending 
            if self.Re is not None and another.Re is not None:
                Re_blend = (1 - weight) * self.Re + weight * another.Re
            else:
                Re_blend = []
                 
            if self.relThk is not None and another.relThk is not None:
                relThk_blend = (1 - weight) * self.relThk + weight * another.relThk
            else:
                relThk_blend = []
            
            return type(self)(np.rad2deg(alpha), cl_blend, cd_blend, cm_blend, relThk_blend, Re_blend)
    
        if   blend_method == 'scaled':
            polar_blend = blend_scaled(self, another, weight)
        elif blend_method == 'direct':
            polar_blend = blend_direct(self, another, weight)
        else:
             return ValueError('Wrong blend method input') 
         
        return polar_blend
        
    def uniform_output(self, alpha_uni = np.linspace(-180,180,361)):
        """
        linear blend of aerodynamic coefficients with another airfoil with 
        the given weight.
        
        Parameters
        ----------
        alpha_uni    :    array_like
            uniform angle of attack for standard output   
            
        Returns
        -------
        polar : Polar 
            a Polar object
        """
        alpha_uni      = np.deg2rad(alpha_uni)
        
        interpolant_cl = interp1d(self.alpha, self.cl, kind = 'cubic')
        interpolant_cd = interp1d(self.alpha, self.cd, kind = 'cubic')
        interpolant_cm = interp1d(self.alpha, self.cm, kind = 'cubic')
        
        cl_uni         = interpolant_cl(alpha_uni)
        cd_uni         = interpolant_cd(alpha_uni)
        cm_uni         = interpolant_cm(alpha_uni)
        
        return type(self)(np.rad2deg(alpha_uni), cl_uni, cd_uni, cm_uni, self.Re)
    
    def plot_polar(self):     
        # Three sub-figures are given to show the lift, drag and moment 
        # coefficients, respectively.

        fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        fig.suptitle('Aerodynamic coefficients')
        
        ax1.plot(np.rad2deg(self.alpha),self.cl)
        ax1.set_ylabel('Cl')
        ax2.plot(np.rad2deg(self.alpha),self.cd)
        ax2.set_ylabel('Cd')
        ax3.plot(np.rad2deg(self.alpha),self.cm)
        ax3.set_xlabel('angle of attack (deg)')
        ax3.set_ylabel('Cm')
            
    def save_txt(self):
        # save the polar data to files            
        filename = input('Please specify the filename: \n')
        output = np.transpose([self.alpha,self.cl,self.cd,self.cm])
        np.savetxt('%s.txt' %filename,
                   output,
                   fmt='%8.2f %9.4f %9.4f %9.4f',
                   header = '# alpha\t cl\t cd\t cm')
        
    def calc_stall(self):
        """
        Calculate polar stall angle.
        
        Returns
        -------
        alfa_stall : float, [deg]
            stall angle 
        
        Notes
        -----
           Calculate the slope of the adjacent three points by linear regression.
           
           For three points: A(x1, y1)、B(x2,y2)、C(x3,y3), the mean values can be calculated:
               x_mean =(x1 + x2 + x3)/3
               y_mean =(y1 + y2 + y3)/3
           
           Assume the fitting line is written as: y = kx + b
               
           Then, the slope k will be: 
           
               k = ((x1 - x_mean)(y1 - y_mean) + (x2 - x_mean)(y2 - y_mean) 
                   + (x3 - x_mean)(y3 - y_mean)) / ((x1 - x_mean)**2 +
                                                    (x2 - x_mean)**2 +
                                                    (x3 - x_mean)**3 )  
            For cylindrical, it can be assumed that cl = 0 for all Aoa.
            Stall angle is 2 degrees for the cylinder here based on the algorithm here. 
                
        """
        alfa = np.rad2deg(self.alpha)
        picked = (alfa > 0) & (alfa < 30)
        alfa = alfa[picked]
        cl = self.cl[picked]
        for i in range(1, len(alfa) - 1):
            x_res = alfa[i-1:i+2] - np.mean(alfa[i-1:i+2])
            y_res = cl[i-1:i+2] - np.mean(cl[i-1:i+2])
            slope = x_res.dot(y_res) / x_res.dot(x_res)
            if slope < 0.05:
                return alfa[i]
        raise BaseException("Error polar, can't get stall angle!")
    
    def evaluate(self, alpha):
        """
        Get lift/drag coefficient at the specified angle of attack.
        
        Polar alpha should be strictly in ascending orders.
        
        Parameters
        ----------
        alpha   :     float, [rad]
            angle of attack
            
        Returns
        -------
        cl      :     float, [-]
            lift coefficient
        cd      :     float, [-]
            drag coefficient 
        """
        if self.alpha[1] < alpha < self.alpha[-2]:
            loc= bisect.bisect(self.alpha, alpha)
            cl = cubic_spline(alpha, self.alpha[loc-2], self.alpha[loc-1], self.alpha[loc],
                      self.alpha[loc+1], self.cl[loc-2], self.cl[loc-1],
                      self.cl[loc], self.cl[loc+1])
            cd = cubic_spline(alpha, self.alpha[loc-2], self.alpha[loc-1], self.alpha[loc],
                      self.alpha[loc+1], self.cd[loc-2], self.cd[loc-1],
                      self.cd[loc], self.cd[loc+1])
        elif self.alpha[-2] <= alpha < self.alpha[-1]:
            cl = quadratic_interpolation(alpha, self.alpha[-3],self.alpha[-2],self.alpha[-1],\
                                          self.cl[-3], self.cl[-2], self.cl[-1])
            cd = quadratic_interpolation(alpha, self.alpha[-3],self.alpha[-2],self.alpha[-1],\
                          self.cd[-3], self.cd[-2], self.cd[-1])
        elif self.alpha[0] < alpha <= self.alpha[1]:
            # wt4 original version is wrong, slightly modified here            
            cl = quadratic_interpolation(alpha, self.alpha[0],self.alpha[1],self.alpha[2],\
                          self.cl[0], self.cl[1], self.cl[2])
            cd = quadratic_interpolation(alpha, self.alpha[0],self.alpha[1],self.alpha[2],\
                          self.cd[0], self.cd[1], self.cd[2])   
        elif alpha >= self.alpha[-1]:
            cl = self.cl[-1]
            cd = self.cd[-1]    
        else: 
            cl     = self.cl[0]
            cd     = self.cd[0]
            
        return cl, cd    