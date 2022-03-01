
"""Verify whether it is appropriate to use interpolation to obtain the aerodynamic
data at Re another than the experimental one"""

# -*- coding: utf-8 -*-
"""
@author: GoldWind Blade Development Team
"""

from numpy import sin, cos, pi, exp, arctan
import numpy as np
import os
from matplotlib import pyplot as plt
import re
import bisect
from airfoil_plc import Airfoil
from mpl_toolkits.mplot3d import Axes3D

from inconfig import Input
from prep import integration, quadratic_interpolation, cubic_spline, interpolate_vector
from polar import Polar
from section import Section

  
class Blade(object):
    
    bID = 1
    
    def __init__(self, 
                 sections, usr_data = None, 
                 Rtip = None, Rhub = None,
                 precurve_tip = None, presweep_tip = None,
                 pitch = 0.0, rpm = None, u = None):
        """
        Parameters
        ----------
        sections:   array_like, class
            sets of sections from hub to tip, number > 1
        usr_data  :   class, optional
            input parameters defined by the user
        Rtip    :   float, optional, [m]
            blade tip radius
        Rhub    :   float, optional, [m]
            blade hub radius
        pitch   :   float, optional, [deg]  
            local radius
        rpm   :   float, optional, [r/min]
            revolution per minute
        u       :   float, optional, [m/s]
            incoming wind velocity        
        """

        self.sections = sections
        self.usr_data = usr_data
        self.pitch    = pitch
        self.rpm      = rpm
        self.u        = u
        self.rpm_factor    = 0
        self.torque_factor = 0
        self.id       = Blade.bID
        
        if Rtip is None: # if not defined directly by the user
            self.Rtip     = self.sections[-1].r
        if Rhub is None:
            self.Rhub     = self.sections[0].r
        if precurve_tip is None:
            self.precurve_tip = self.sections[-1].precurve
        if presweep_tip is None:
            self.presweep_tip = self.sections[-1].presweep

        # equivalent rotor radius, used to compute non-dimensional coefficients 
        # precurve_tip is negative
        self.rotorR = self.Rtip * cos(np.radians(self.usr_data.precone)) + \
                       self.precurve_tip * sin(np.radians(self.usr_data.precone))
        self.SR     = pi * self.rotorR**2  # regarding in the aziumth coordinate system
                              
        Blade.bID = Blade.bID + 1
    
    
    def calc_area_moment(self):
        """
        Calcuate the first, second, third and fourth moment of inertia.
        
        # Returns
        # -------
        # I0      :    float, [m^2]
        #     blade area, 0th blade area moment
        # I1      :    float, [m^3]
        #     1st blade area moment
        # I2      :    float, [m^4]
        #     2nd blade area moment
        # I3      :    float, [m^5]
        #     3rd blade area moment    
        """   
        
        geometry = self.get_geometry()   
        c = geometry['c']
        coord_az = self.define_curvature()
        z_az = coord_az['z']
        s    = coord_az['s']
        
        # correction for coning, prebend and presweep
        self.area, _   = integration(c          , s) 
        self.I0,   _   = integration(c          , z_az)
        self.I1,   _   = integration(c * z_az   , z_az)  
        self.I2,   _   = integration(c * z_az**2, z_az) 
        self.I3,   _   = integration(c * z_az**3, z_az) 
   
    @staticmethod
    def init_from_file(path_geometry, path_af, usr_data):
        """
        Initilization of blade class.
        
        Parameters
        ----------
        path_geometry : str
            absoulte path of the geometry file    
        path_af       : str
            absoulte path of the airfoil file
        usr_data      : class
            input parameters defined by the user
            
        Returns
        -------
        blade : Blade 
            a Blade object
        
        Notes:
            Planfrom will be composed of no more than 6 columns.
                # Radius Chord	Twist Thickness	precurve[optional]	presweep[optional]
            
            All geometric data for standard airfoil masterprofiles should be 
                stored with file type .geo. 
                
            All polar data for standard airfoil masterprofiles should be 
                stored with file type .txt. 
                
            Blending airfoil geometry ususally give the target thickness as
                the input, while blending polar data usually treat the weight
                as the input.
                    # blending airfoil geometry
                     geo_list[loc-1].blend_geo(geo_list[loc], thk)
                    # blending polar data
                    weight = (thk - sec_standard_af[loc-1].relThk)/\
                                (sec_standard_af[loc].relThk - sec_standard_af[loc-1].relThk)
                    polar_left.blend_polar(polar_right, weight, blend_method)
        """
        geometry = np.genfromtxt(path_geometry)
        
        r      = geometry[:,0]     # Radius
        c      = geometry[:,1]     # Chord
        twist  = geometry[:,2]     # Twist
        thick  = geometry[:,3]     # Thickness

        # precurve 
        if geometry.shape[1] > 4:
            precurve = geometry[:,4]
        else:
            precurve = np.zeros_like(r)
            
        # presweep
        if geometry.shape[1] > 5:
            presweep = geometry[:,5]
        else:
            presweep = np.zeros_like(r)

        dirs = os.listdir(path_af)
        
        # read geometry and thickness info from file
        geo_list = []
        thk_list = []
        for geo_dir in dirs:
            if os.path.splitext(geo_dir)[1] == ".geo":
                af_single_Thk = re.findall("-?\d+\.?\d*[E,e]?\d*",os.path.splitext(geo_dir)[0])
                af_single_Thk = float(af_single_Thk[0])
                geo_list.append(Airfoil.init_from_file(os.path.join(path_af,geo_dir), name = '%s' % os.path.splitext(geo_dir)[0]))
                geo_list[-1].getThick()
                thk_list.append(geo_list[-1].maxThick)   # compute the thickness based on the airfoil coordinates given
                
        geo_list = sorted(geo_list, key = lambda geo_list : geo_list.maxThick)
        thk_list = sorted(thk_list)
        
        # save the standard airfoil geometry when the plot_af is 1
        if usr_data.plot_af == 1 and len(geo_list) > 0:
            geo_list[0].plotMultiCurves(*geo_list[1:])
            plt.grid(linestyle='-.')
            plt.yticks(np.arange(-0.5, 0.55, 0.1))
            output_fig_af = os.path.dirname(os.getcwd()) +'\\case\\output' + '\\%s_%s' %(usr_data.name, usr_data.method_key) + '\\figure'
            if not os.path.exists(output_fig_af):
                # create file directory if not existing
                os.makedirs(output_fig_af) 
            path_fig_af = os.path.join(output_fig_af,'Standard_airfoil.png')
            plt.savefig(path_fig_af, bbox_inches='tight')
            
        # construct the geometry list
        sec_geos = []
        if len(geo_list) > 0:
            for thk in thick:
                loc = bisect.bisect(thk_list, thk)    # return the right index
                if loc == len(thk_list):  # relThk >= maximum airfoil thickness    
                    sec_geo = geo_list[-1].getCoord()
                elif thk == thk_list[-1] or loc == 0:            # relThk <= minimum airfoil thickness
                    sec_geo = geo_list[0].getCoord()
                else:
                    # weight = (thk - geo_list[loc-1].maxThick)/(geo_list[loc].maxThick - geo_list[loc-1].maxThick)
                    sec_geo = geo_list[loc-1].blend_geo(geo_list[loc], thk).getCoord()

                sec_geos.append(sec_geo)
        else:
            sec_geos = [None] * len(thick)
            
        # construct the polar list
        polar_list     = []
        for af_dir in dirs:
            if os.path.splitext(af_dir)[1] == ".txt":
                t_Re = re.findall("-?\d+\.?\d*[E,e]?\d*",os.path.splitext(af_dir)[0])   # the number in the filename represents the relative thickness
                af_single_Thk = float(t_Re[0])
                if len(t_Re) == 2:
                    Re  = float(t_Re[1])
                else:
                    Re  = None
                polar_list.append(Polar.init_from_file(os.path.join(path_af,af_dir), relThk = af_single_Thk, Re = Re))
        
        polar_list = sorted(polar_list, key = lambda af_polar : af_polar.relThk)    # sorted by thickness
        
        af_Thk = []
        for pol in polar_list:
            af_Thk.append(pol.relThk)
        
        if len(af_Thk) - len(np.unique(af_Thk)) > 0:

            sec_standard_af = []    # store multi Re of the same thickness as a section object
            polars = []     
            i = 0
            polars.append(polar_list[0])
            while i <= (len(af_Thk) - 2):          # construct section classes to help store the potential multi polars of any standard airfoil            
                if af_Thk[i+1] == af_Thk[i]:
                    polars.append(polar_list[i+1])
                    if i == len(af_Thk) - 2:
                        polars = sorted(polars, key = lambda af_polar : af_polar.Re)    
                        sec_standard_af.append(Section(polars, relThk = af_Thk[i+1]))  
                else:
                    polars = sorted(polars, key = lambda af_polar : af_polar.Re)
                    sec_standard_af.append(Section(polars, relThk = af_Thk[i]))
                    polars = []
                    polars.append(polar_list[i+1])
                    if i == len(af_Thk) - 2:
                        sec_standard_af.append(Section(polars, relThk = af_Thk[i+1]))
                    
                i = i + 1
            
            # compute the section polars
            af_Thk = np.unique(af_Thk)      # get rid of the duplicated thickness
            polars = []
            for thk in thick:
                loc = bisect.bisect(af_Thk, thk)    # return the right index
                if  loc == len(af_Thk):    # relThk > maximum airfoil thickness
                    polars_Re = sec_standard_af[-1].polars
                    # alpha_stall = sec_standard_af[-1].polars.calc_stall()
                elif thk == af_Thk[0] or loc == 0:            # relThk < minimum airfoil thickness
                    polars_Re = sec_standard_af[0].polars
                    # alpha_stall = sec_standard_af[0].polars.calc_stall()
                else:
                    # find maximum and minimum Re
                    Re_left  = []
                    Re_right = []
                    for j in np.arange(len(sec_standard_af[loc-1].polars)):
                        Re_left.append(sec_standard_af[loc-1].polars[j].Re)  
                    for j in np.arange(len(sec_standard_af[loc].polars)):
                        Re_right.append(sec_standard_af[loc].polars[j].Re)
  
                    Re_left  = np.array(Re_left)
                    Re_right = np.array(Re_right)
                    Re_merge = np.union1d(Re_left, Re_right)
        
                    Re_min = max(Re_left.min(), Re_right.min())
                    Re_max = min(Re_left.max(), Re_right.max())
                    
                    if Re_min <= Re_max:
                        id_Re = np.logical_and(Re_merge >= Re_min, \
                                          Re_merge <= Re_max)
                        Re = Re_merge[id_Re]         # represent the merged Re from two adjacent standard airfoils
                    else:
                        Re = [Re_max]       
                    
                    # interpolate Reynolds number
                    polars_Re = []
                    for i in np.arange(len(Re)):
                        loc_left = bisect.bisect(Re_left,Re[i])     # represents the standard airfoil thickness, that is closest below Re[i] 
                        if loc_left == len(Re_left):            # Re > maximum Re
                            polar_left       = sec_standard_af[loc-1].polars[-1]
                        elif Re[i] == Re_left[0] or loc_left == 0:           # Re < minimum Re
                            polar_left = sec_standard_af[loc-1].polars[0]
                        else:
                            weight_left = (Re[i] - sec_standard_af[loc-1].polars[loc_left - 1].Re) / \
                            (sec_standard_af[loc-1].polars[loc_left].Re - sec_standard_af[loc-1].polars[loc_left - 1].Re)
                            polar_left = sec_standard_af[loc-1].polars[loc_left-1].blend_polar(
                                sec_standard_af[loc-1].polars[loc_left], weight_left, usr_data.blend_method)
                        
                        
                        loc_right = bisect.bisect(Re_right,Re[i])   # represents the standard airfoil thickness, that is closest above Re[i] 
                        if loc_right == len(Re_right):      # Re >= maximum Re
                            polar_right = sec_standard_af[loc].polars[-1]          
                        elif Re[i] == Re_right[0] or loc_right == 0: # Re <= maximum Re
                            polar_right = sec_standard_af[loc].polars[0]
                        else:
                            weight_right = (Re[i] - sec_standard_af[loc].polars[loc_right - 1].Re) / \
                            (sec_standard_af[loc].polars[loc_right].Re - sec_standard_af[loc].polars[loc_right - 1].Re)
                            polar_right = sec_standard_af[loc].polars[loc_right-1].blend_polar(
                                sec_standard_af[loc].polars[loc_right], weight_right, usr_data.blend_method)
                        
                        # interpolate the thickness
                        weight = (thk - sec_standard_af[loc-1].relThk)/\
                            (sec_standard_af[loc].relThk - sec_standard_af[loc-1].relThk)
                        polar = polar_left.blend_polar(polar_right, weight, usr_data.blend_method)
                        polars_Re.append(polar)
                        
                polars.append(polars_Re)
                
            sections = []
            for i in range(len(polars)):
                sections.append(Section(polars[i], r = r[i], relThk = thick[i], 
                                        c = c[i], twist = twist[i],
                                        geom = sec_geos[i], precurve = precurve[i],
                                        presweep = presweep[i]))
        else:
            polars = []
            alpha_stall_list = []
            for thk in thick:
                loc = bisect.bisect(af_Thk, thk)    # return the right index
                if loc == len(af_Thk):  # relThk >= maximum af_Thk    
                    polar = [polar_list[-1]]
                    # polar[0].calc_stall
                    alpha_stall = polar_list[-1].calc_stall()
                elif thk == af_Thk[-1] or loc == 0:            # relThk <= minimum af_Thk
                    polar = [polar_list[0]]
                    # polar[0].calc_stall
                    alpha_stall = polar_list[0].calc_stall()
                else:
                    weight = (thk - polar_list[loc-1].relThk)/(polar_list[loc].relThk - polar_list[loc-1].relThk)
                    polar = [polar_list[loc-1].blend_polar(polar_list[loc], weight, usr_data.blend_method)]
                    
                    alpha_stall_1 = polar_list[loc-1].calc_stall()
                    alpha_stall_2 = polar_list[loc].calc_stall()
                    alpha_stall = (1 - weight) * alpha_stall_1 + weight * alpha_stall_2
    
                polars.append(polar)
                alpha_stall_list.append(alpha_stall)
                
        # construct the section class
            sections = []
            for i in range(len(polars)):
                sections.append(Section(polars[i], r = r[i], relThk = thick[i], 
                                        c = c[i], twist = twist[i],
                                        geom = sec_geos[i],
                                        alpha_stall = alpha_stall_list[i],
                                        precurve = precurve[i],
                                        presweep = presweep[i])) 
        # construct the blade class
        blade = Blade(sections, usr_data)
        
        return blade
    
    def distributed_aero(self):
        """
        return a dictionary containing all the aerodynamic related data derived with
        BEM theory. 
        
        Returns
        -------
        dict : dict
            a dictionary 
        """
        geometry = self.get_geometry()
        c        = geometry['c']  
        alpha, cl, cd, phi = [], [], [], []
        CP, CT, CQ, CRBM_flap = [], [], [], []
        a, a_prime = [], []
        pn, pt = [], []
        vRel   = []
        
        for i in self.sections:
            alpha.append(i.alpha)
            cl.append(i.cl)
            cd.append(i.cd)
            phi.append(i.phi)
            CP.append(i.CP)
            CT.append(i.CT)
            CQ.append(i.CQ)
            CRBM_flap.append(i.CRBM_flap)
            a.append(i.a)
            a_prime.append(i.a_prime)
            pn.append(i.pn)
            pt.append(i.pt)
            vRel.append(i.vRel)
            
        alpha = np.array(alpha)
        cl    = np.array(cl)
        cd    = np.array(cd)
        phi   = np.array(phi)
        CP    = np.array(CP)
        CT    = np.array(CT)
        CQ    = np.array(CQ)
        CRBM_flap = np.array(CRBM_flap)
        a     = np.array(a)
        a_prime = np.array(a_prime)
        pn    = np.array(pn)
        pt    = np.array(pt)
        vRel  = np.array(vRel)
        cCl   = c * cl / self.rotorR
        circulation = 0.5 * cCl * (1 - a) / sin(phi)
        
        return dict(zip(['alpha', 'cl','cd','phi','CP','CT', 'CQ', 'CRBM_flap', 'a',
                         'a_prime', 'pn', 'pt','vRel','cCl', 'circulation'],
                        [alpha, cl, cd, phi , CP, CT, CQ, CRBM_flap, a, a_prime, pn, pt,
                         vRel, cCl, circulation]))
        
    def distributed_noise(self):
        """
        Return noise distribution along the blade.  

        Returns
        -------
        noise : array_like, [dB]
            noise 
        """
        noise = []
        
        for sec in self.sections:
            noise.append(sec.noise)
            
        noise = np.array(noise)
        
        return noise
    
    def set_section_noise(self, sec_noise):
        """
        Parameters
        ----------
        sec_noise : array_like, [dBA]
            sectional noise producing the maximum blade noise, only used for
            BPM-based noise method.

        Returns
        -------
        None.

        """
        for i, sec in enumerate(self.sections):
            sec.noise = sec_noise[i]
        
    def get_geometry(self):

        r     = []
        c     = []
        twist = []
        relThk= []
        absThk= []
        precurve = []
        presweep = []
        
        for i in self.sections:
            r.append(i.r)
            c.append(i.c)
            twist.append(i.twist)
            relThk.append(i.relThk)
            precurve.append(i.precurve)
            presweep.append(i.presweep)
        
        r = np.array(r)
        c = np.array(c)
        twist = np.array(twist)
        relThk= np.array(relThk)
        absThk = c * relThk / 100.0
        precurve = np.array(precurve)
        presweep = np.array(presweep)
        
        return dict(zip(['r', 'c', 'twist', 'relThk','absThk','precurve', 'presweep'],
                        [r, c, twist, relThk, absThk, precurve, presweep]))
    
    def run_BEM(self, u = None, rpm = None, pitch = None, grad_calc = 0):
        """
        Different options have been provided for BEM calculations.
        
        Parameters
        ----------
        u           :  float, optional, [m/s]
             incoming wind speed
        rpm        :   float, optional, [r/min]
             revoluation per minute
        pitch      :   float, optional, [deg]
             pitch angle     
        grad_calc   :  int,   [-]
            if 1, calculate gradient
        """        
        
        if u is not None and rpm is not None and pitch is not None:
            # ease for usage for usrs who are more conformtable to define the variables as the input variable
            self.u, self.rpm, self.pitch = u, rpm, pitch
        
        coord_3d = self.define_curvature()
        cone     = coord_3d['cone']  # degree
        s        = coord_3d['s']
        z_az     = coord_3d['z']

        sec = self.sections[0]
        
        V = self.define_velocity_components()
        Vx, Vy = V['Vx'], V['Vy']
        
        if self.usr_data.method_key == 'wt4':
            sec = self.sections[0]
            sec.static_BEM_wt4(self.u, self.rpm, Vx[0], Vy[0], self.pitch, self.Rtip, self.Rhub, 
                            self.usr_data.precone, cone[0], self.usr_data.tilt, self.usr_data.rho, 
                            self.usr_data.mu, self.usr_data.B, self.usr_data.blend_method,
                            self.usr_data.tip_corr,self.usr_data.hub_corr,
                            self.usr_data.correction_3d, grad_calc)
            
            for j in range(1,len(self.sections)):
                sec = self.sections[j]
                sec.static_BEM_wt4(self.u, self.rpm, Vx[j], Vy[j], self.pitch, self.Rtip, self.Rhub,  
                                self.usr_data.precone, cone[j], self.usr_data.tilt, self.usr_data.rho, 
                                self.usr_data.mu, self.usr_data.B, self.usr_data.blend_method,
                                self.usr_data.tip_corr,self.usr_data.hub_corr,
                                self.usr_data.correction_3d, grad_calc, 
                                w = self.sections[j-1].w)
      
        elif self.usr_data.method_key == 'fpm': 
            sec.static_BEM(self.u, self.rpm, Vx[0], Vy[0], self.pitch, self.Rtip, self.Rhub, 
                        self.usr_data.precone, self.usr_data.tilt, self.usr_data.rho, 
                        self.usr_data.mu, self.usr_data.B, self.usr_data.blend_method,
                        self.usr_data.tip_corr,self.usr_data.hub_corr,
                        self.usr_data.correction_3d, self.usr_data.high_load,
                        grad_calc)
        
            for j in range(1,len(self.sections)):
                sec = self.sections[j]
                sec.static_BEM(self.u, self.rpm, Vx[j], Vy[j], self.pitch, self.Rtip, self.Rhub,  
                                self.usr_data.precone, self.usr_data.tilt, self.usr_data.rho, 
                                self.usr_data.mu, self.usr_data.B, self.usr_data.blend_method, 
                                    self.usr_data.tip_corr,self.usr_data.hub_corr,
                                    self.usr_data.correction_3d, self.usr_data.high_load,
                                    grad_calc)

                    
        elif self.usr_data.method_key == 'ifpm': 
            sec.static_iBEM(self.u, self.rpm, Vx[0], Vy[0], self.pitch, self.Rtip, self.Rhub, 
                        self.usr_data.precone, self.usr_data.tilt, self.usr_data.rho, 
                        self.usr_data.mu, self.usr_data.B, self.usr_data.blend_method,
                        self.usr_data.tip_corr,self.usr_data.hub_corr,
                        self.usr_data.correction_3d, self.usr_data.high_load, 
                        grad_calc)

            for j in range(1,len(self.sections)):
                sec = self.sections[j]
                sec.static_iBEM(self.u, self.rpm, Vx[j], Vy[j], self.pitch, self.Rtip, self.Rhub,  
                                self.usr_data.precone, self.usr_data.tilt, self.usr_data.rho, 
                                self.usr_data.mu, self.usr_data.B, self.usr_data.blend_method,
                                self.usr_data.tip_corr,self.usr_data.hub_corr,
                                self.usr_data.correction_3d, self.usr_data.high_load,
                                grad_calc)
                
        elif self.usr_data.method_key == 'CCBlade':
                
            sec.static_CCBlade(self.u, self.rpm, Vx[0], Vy[0], self.pitch, self.Rtip, self.Rhub, 
                            self.usr_data.precone, self.usr_data.precone, self.usr_data.rho, 
                            self.usr_data.mu, self.usr_data.B, self.usr_data.blend_method,
                            self.usr_data.tip_corr,self.usr_data.hub_corr,self.usr_data.high_load,
                            self.usr_data.correction_3d, grad_calc)

            for j in range(1,len(self.sections)):
                sec = self.sections[j]
                sec.static_CCBlade(self.u, self.rpm, Vx[j], Vy[j], self.pitch, self.Rtip, self.Rhub,  
                            self.usr_data.precone, self.usr_data.tilt, self.usr_data.rho, 
                            self.usr_data.mu, self.usr_data.B, self.usr_data.blend_method,
                            self.usr_data.tip_corr,self.usr_data.hub_corr,self.usr_data.high_load,
                            self.usr_data.correction_3d, grad_calc)
        
        else:
            raise BaseException('Unknown method key %s!' % self.usr_data.method_key)
                
        q = 0.5 * self.usr_data.rho * self.u**2
            
        out_aero         = self.distributed_aero()
        pn, pt           = out_aero['pn'], out_aero['pt']
        cone             = np.radians(cone)
        
        # thrust in each element is in different direction and should modified to be in-line with the azimuth axis
        self.T, _           = integration(self.usr_data.B * pn * cos(cone), s)     
        self.Q, _           = integration(self.usr_data.B * pt * z_az, s)
        self.flap_moment, _ = integration(self.usr_data.B * pn * z_az, s)
        self.P              = self.Q * self.rpm * pi / 30.0

        self.CP   = self.P / (q * self.SR * self.u)
        self.CT   = self.T / (q * self.SR)
        self.CQ   = self.Q / (q * self.rotorR * self.SR)
        self.CRBM_flap = self.flap_moment / (q * self.rotorR * self.SR)

        self.Mt0 = self.flap_moment / self.usr_data.B   # for a single blade, changes the negative sign into the positive 
        if self.CT < 8/9:
            a = (1 - (1 - self.CT)**(1/2)) / 2
        else:
            qq = 11/81
            rr = (-20 + 9 * self.CT + 250/27) / 54
            ss = (rr + qq**(3/2) + rr**(2))**(1/3)
            tt = rr - (qq**3 + rr**2)**(1/2)
            tt = (abs(tt))**(1/3) * abs(tt) / tt
            a  = ss + tt + 5/9
            
        self.a = a
    
    def run_noise(self):
    
        def calc_BPM_noise(self):       
            """
            Calculate noise based on BPM method. The computation of the boundary 
            layer thickness is decied by the noise_mode. Typicall, there are three 
            options: BPM-xfoil; BPM-gfoil; BPM-wt4(based on experience formula and also the 
            quickest).
            
            Notes
            -----
            total_noise    :   float, db(A)
                maximum level of noise power
            detailed_noise    :   array_like
                SPL_max is compose of three columns, i.e. [[1st col, 2nd col , 3rd col]].
                1st column ---> represent section with thickness/chord > 40%
                2nd column ---> represent the different noise frequency
                3nd column ---> represent several kinds of noise with the 1st column the frequency, i.e. 
                        Freq, SPLs,SPLp,SPLalfa,SPLTBL,SPLBL,SPLInflow, A-weighted total_noise
        
            """
            R11 = (self.usr_data.hub_height ** 2 + self.usr_data.obserx ** 2)
            IEC_corr = 10 * np.log10(4 * np.pi * R11)        
            for section in self.sections:
                if section.relThk < 40:
                    section.refresh_bl(mode = self.usr_data.noise_mode,
                              trip = self.usr_data.noise_trip,c0 = self.usr_data.v_sound)
                else:  # for very thick airfoil, use direct mode compulsorily
                    section.refresh_bl(mode = 'BPM-wt4', trip = self.usr_data.noise_trip, c0 = self.usr_data.v_sound)                    
            
            geometry    = self.get_geometry()
            r           = geometry['r']
            total_noise = 0
            for bladeangle in np.arange(0, 360, 15):
                SPL = np.zeros([len(r), 34, 8])
                for i, section in enumerate(self.sections):
                    try:
                        theta = self.pitch + section.twist  # combination of pitch and twist angle
                    except:  # if pitch not given, it is then assumed 0
                        theta = section.twist  
                        # airfoil span is set 1 in order to keep consistence with the non-BPM based methods
                    SPL[i] = section.BPM(1,
                                self.usr_data.mu, self.usr_data.v_sound,
                                self.usr_data.rho, self.usr_data.hub_height, 
                                self.usr_data.precone,self.usr_data.tilt, theta,
                                self.usr_data.obserx, self.usr_data.obsery, 
                                self.usr_data.obserz, IEC_corr, bladeangle=bladeangle)
                tmp_sec_noise = self.distributed_noise()
                noise, _  = integration(tmp_sec_noise, r)
                tmp_total_noise = 10 * np.log10(noise)
    
                if tmp_total_noise > total_noise:   # find the maximum total noise within the given blade azimuth angle
                    total_noise    = tmp_total_noise
                    detailed_noise = SPL
                    sec_noise      = tmp_sec_noise
                    
            self.noise = total_noise
            self.detailed_noise = detailed_noise
            self.set_section_noise(sec_noise)      # set the sectional noise producing the maximum total noise to each section
        
        def calc_noise(self):
            """
            Calculate the blade noise based on the empirical formula. No detailed info 
            about distribution of noise regarding the frequency will be given.
            """
            
            for sec in self.sections:
                sec.calc_noise(self.usr_data.B)
            
            geometry = self.get_geometry()
            r   = geometry['r']
            sec_noise = self.distributed_noise()
            noise, _ = integration(sec_noise, r)
            self.noise = 10 * np.log10(noise)
        
        
        if self.usr_data.noise_mode == 'wt4':   # use the emprical formula in wt4 rather than BPM
            calc_noise(self)
        elif self.usr_data.noise_mode == 'BPM-wt4' or self.usr_data.noise_mode == 'BPM-xfoil' \
            or self.usr_data.noise_mode == 'BPM-gfoil':   # use the BPM-based method to compute noise
            calc_BPM_noise(self)
        else:
            raise ValueError('Wrong noise mode')
            
    def calc_rpm_factor(self, opt):                      
        
        """
        Parameters
        ----------
        opt   : int,  [-]
            if 1,  apply optimum calculation
        """
        
        if self.usr_data.use_rpm_limiter and opt == 0:
            if self.u <= self.usr_data.rpm_limiter_table[0,0]:
                self.rpm_factor = self.usr_data.rpm_limiter_table[0,1]
            elif self.u >= self.usr_data.rpm_limiter_table[-1,0]:
                self.rpm_factor = self.usr_data.rpm_limiter_table[-1,1]
            else:
                self.rpm_factor = np.interp(self.u, self.usr_data.rpm_limiter_table[:,0], self.usr_data.rpm_limiter_table[:,1])
        else:
            self.rpm_factor = 1
     
    def define_curvature(self):
        
        x_az = np.zeros(len(self.sections))
        y_az = np.zeros(len(self.sections))
        z_az = np.zeros(len(self.sections))
        precone = self.usr_data.precone
        
        # sec coordinates in blade-aligned soordinate system
        for i, sec in enumerate(self.sections):
            x_az[i], y_az[i], z_az[i] = sec.define_curvature(precone)
        
        # compute coning angle in order to compute the relative wind velocity
        cone       = np.zeros(len(self.sections))
        cone[0]    = np.arctan(-(x_az[1]- x_az[0])/(z_az[1] - z_az[0]))
        cone[1:-1] = 0.5 * (np.arctan(-(x_az[1:-1] - x_az[0:-2])/(z_az[1:-1] - z_az[0:-2]))
                       + np.arctan(-(x_az[2:] - x_az[1:-1])/(z_az[2:] - z_az[1:-1])))
        cone[-1]   = np.arctan(-(x_az[-1] - x_az[-2])/(z_az[-1] - z_az[-2]))
        cone       = np.rad2deg(cone)
        
        # total arc length of blade used to compute thrust, torque and moment
        s          = np.zeros(len(self.sections))
        for i in np.arange(1, len(self.sections)):
            s[i] = s[i-1] + ((self.sections[i].precurve - self.sections[i-1].precurve)**2 +
                             (self.sections[i].presweep - self.sections[i-1].presweep)**2 +
                             (self.sections[i].r        - self.sections[i-1].r)**2)**(1/2)    
        
        return dict(zip(['x','y','z', 'cone','s'],[x_az, y_az, z_az, cone, s]))
    
    def define_velocity_components(self, azimuth = 0.0, yaw = 0.0):
        
        """
        azimuth    : float, [deg]
            azimuth angle
        yaw        : float, [deg]
            yaw angle
        """
        azimuth = np.radians(azimuth)
        yaw     = np.radians(yaw)
        tilt    = np.radians(self.usr_data.tilt)
        sa, ca  = sin(azimuth), cos(azimuth)
        st, ct  = sin(tilt),    cos(tilt)
        sy, cy  = sin(yaw),     cos(yaw)
        
        b2a     = self.define_curvature()
        y_az, z_az = b2a['y'], b2a['z']
        cone    = b2a['cone'] # array
        cone    = np.radians(cone)
        sc, cc  = sin(cone), cos(cone)
        
        # first rotate with azimuth angle then the tilt angle
        # height_from_hub = (y_az * sa + z_az * ca) * ct - x_az * st
        
        # velocity with shear
        # V = self.u*(1 + heightFromHub/hubHt)**shearExp 
        
        omega  = self.rpm * pi / 30
        # transform wind from wind c.s. to blade c.s.
        # Vwind = DirectionVector(V, 0*V, 0*V).windToYaw(yaw).yawToHub(tilt).hubToAzimuth(azimuth).azimuthToBlade(cone)
        # self.u and azimuth angle can also be array, then modification should be applied
        Vwind_x = self.u * ((cy*st*ca + sy*sa)*sc + cy*ct*cc)
        Vwind_y = self.u * (cy*st*sa - sy*ca)
        # u_z = self.u * (-cy*ct*sc + (sy*sa + cy*st*ca)*cc)
        
        # transform rot speed azimuth c.s. to blade c.s.
        # Vrot = -DirectionVector(omega, 0, 0).cross(az_coords).azimuthToBlade(cone)
        
        Vrot_x = -omega * y_az * sc
        Vrot_y =  omega * z_az
        # Vrot_z = -omega * y_az * cc
        
        Vx = Vwind_x + Vrot_x
        Vy = Vwind_y + Vrot_y
        
        return dict(zip(['Vx', 'Vy'], [Vx, Vy]))

    
    def calc_torque_factor(self, opt):       
        """
        Parameters
        ----------
        opt              : int,  [-]
            if 1,  apply optimum calculatoin
        """

        if self.usr_data.use_torque_limiter and opt:
            if self.u <= self.usr_data.torque_limiter_table[0,0]:
                self.torque_factor = self.usr_data.torque_limiter_table[0,1]
            elif self.u >= self.usr_data.torque_limiter_table[-1,0]:
                self.torque_factor = self.usr_data.torque_limiter_table[-1,1]
            else:
                self.torque_factor = np.interp(self.u, self.usr_data.torque_limiter_table[:,0], self.usr_data.torque_limiter_table[:,1])
        else:
            self.torque_factor = 1
                       
    def opt_pitch(self, opt):        
        """
        Parameters
        ----------
        opt              : int,  [-]
            if 1,  apply optimum calculation

        """
        omega = self.rpm * pi / 30
        tau = (5**(1/2)-1)/2

        if self.usr_data.use_pitch_table and opt == 0:
            if self.u <= self.usr_data.pitch_table[0,0]:
                self.pitch = self.usr_data.pitch_table[0,1]
            elif self.u >= self.usr_data.pitch_table[-1,0]:
                self.pitch = self.usr_data.pitch_table[-1,1]
            else:
                self.pitch = np.interp(self.u, self.usr_data.pitch_table[:,0], self.usr_data.pitch_table[:,1])
            
            self.run_BEM()
            self.losses(opt)
        else:
            al = -20 + arctan(self.u / omega / self.Rtip) * 180 / pi
            au =  10 + arctan(self.u / omega / self.Rtip) * 180 / pi
            aa = au - tau * (au - al)
            ab = al + tau * (au - al)
            
            self.pitch = aa
            self.run_BEM()
            fa = self.P
            self.pitch = ab
            self.run_BEM()
            fb = self.P
            
            while True:
                if fa > fb:
                    au = ab
                    ab = aa
                    aa = au - tau * (au - al)
                    fb = fa
                    self.pitch = aa
                    self.run_BEM()
                    fa = self.P
                else:
                    al = aa
                    aa = ab
                    ab = al + tau * (au - al)
                    fa = fb
                    self.pitch = ab
                    self.run_BEM()
                    fb = self.P
                    
                self.losses(opt)    # losses 
                if (ab - aa) < 1e-5 or (self.P_gene > self.usr_data.Pgen_max * omega/ self.usr_data.rpm_max * 30 /pi * self.torque_factor and opt == 0) :
                     break
                
        if self.P_gene > self.usr_data.Pgen_max * omega/ self.usr_data.rpm_max * 30 /pi * self.torque_factor and opt == 0:
                
            aa = self.pitch + 10
            ab = aa + 10
            self.pitch = aa
            self.run_BEM()
            self.losses(opt)
            fa = self.P_gene - self.usr_data.Pgen_max * omega / self.usr_data.rpm_max * 30 /pi * self.torque_factor
            
            while True:
                self.pitch = ab
                self.run_BEM()
                self.losses(opt)
                fb = self.P_gene - self.usr_data.Pgen_max * omega/ self.usr_data.rpm_max * 30 /pi * self.torque_factor
                ac = ab - fb * (ab - aa) / (fb - fa)
                aa = ab
                ab = ac
                fa = fb
                if (abs(fb) < 1e-4):
                    break
                
        self.run_noise()
            
        if (self.noise > self.usr_data.noise_limit) and self.u <= self.usr_data.u_noise_limit \
            and self.usr_data.noise_reduction and opt == 0:
            aa = self.pitch
            ab = aa + 1
            self.pitch = aa
            self.run_BEM()
            self.run_noise()
            fa = self.noise - self.usr_data.noise_limit
            
            while True:
                self.pitch = ab
                self.run_BEM()
                self.run_noise()
                fb = self.noise - self.usr_data.noise_limit
                ac = ab - fb * ( ab - aa) / (fb - fa)
                aa = ab
                ab = ac
                fa = fb
                if ( abs(fb) < 1e-3):
                    break
            self.losses(opt)
    
    def opt_rpm(self, opt):
        
        """
        Parameters
        ----------
        opt   : int,  [-]
            if 1,  apply optimum calculation

        """
        # optimize rpm regarding maximum power production by golden section method
        tau = (5**(1/2) - 1) / 2
        
        if self.usr_data.use_rpm_table and opt == 0:
            if self.u <= self.usr_data.rpm_table[0,0]:
                omega = self.usr_data.rpm_table[0,1] * pi / 30
            elif self.u >= self.usr_data.rpm_table[-1,0]:
                omega = self.usr_data.rpm_table[-1,1] * pi / 30
            else:
                rpm = np.interp(self.u, self.usr_data.rpm_table[:,0], self.usr_data.rpm_table[:,1])
                omega = rpm * pi/ 30
            
            if omega < self.usr_data.rpm_min * pi/ 30:
                omega = self.usr_data.rpm_min * pi / 30
            elif omega > self.usr_data.rpm_max * pi/ 30  * self.rpm_factor:
                omega = self.usr_data.rpm_max * pi/ 30 * self.rpm_factor
                
            self.rpm = omega * 30 / pi    
            self.opt_pitch(opt)
            
        else:
            
            if opt == 0:
                al = self.usr_data.rpm_min  * pi / 30
                au = self.usr_data.rpm_max  * pi / 30 * self.rpm_factor
            else:
                al = self.u / (self.rotorR)
                au = 20 * self.u / (self.rotorR)
                
            aa = au - tau * (au - al)
            ab = al + tau * (au - al)
            
            omega = aa
            self.rpm = omega * 30 / pi
            self.opt_pitch(opt)
            fa = self.P_gene
            omega = ab
            self.rpm = omega * 30 / pi
            self.opt_pitch(opt)
            fb = self.P_gene
            
            while True:
                if fa > fb:
                    au = ab
                    ab = aa
                    aa = au - tau * (au - al)
                    fb = fa
                    omega = aa
                    self.rpm = omega * 30 / pi
                    self.opt_pitch(opt)
                    fa = self.P_gene
                else:
                    al = aa
                    aa = ab
                    ab = al + tau * (au - al)
                    fa = fb
                    omega = ab
                    self.rpm = omega * 30 / pi
                    self.opt_pitch(opt)
                    fb = self.P_gene
                if ((ab - aa) * self.Rtip) < 1e-5:
                    break
                       
    def losses(self, opt):
        """
        Parameters
        -----------
        opt   : int,  [-]
            if 1,  no losses applied
        
        Notes
        ------
        Rather than use 4 kinds of losses separately(generator losses, own consumption losses
        gearbox losses and converter losses), only one table is given here to represent those losses.
        """
        if opt == 0:

            self.P_gene = self.P

            if self.P_gene <=  self.usr_data.loss_table[0,0]:
                own_consumption =  self.usr_data.loss_table[0,1]
            elif self.P_gene >=  self.usr_data.loss_table[-1,0]:
                own_consumption =  self.usr_data.loss_table[-1,1]
            elif len( self.usr_data.loss_table) == 2:
                own_consumption = np.interp(self.P_gene, self.usr_data.loss_table[:,0], 
                                            self.usr_data.loss_table[:,1]  )
                
            elif self.P_gene <=  self.usr_data.loss_table[1,0]:
                own_consumption = quadratic_interpolation(self.P_gene,  self.usr_data.loss_table[0,0],  
                                                          self.usr_data.loss_table[1,0],  self.usr_data.loss_table[2,0], 
                                                           self.usr_data.loss_table[0,1],  self.usr_data.loss_table[1,1],  
                                                           self.usr_data.loss_table[2,1])
            elif self.P_gene >=  self.usr_data.loss_table[-2,0]:
                own_consumption = quadratic_interpolation(self.P_gene,  self.usr_data.loss_table[-3,0],  
                                                          self.usr_data.loss_table[-2,0],  self.usr_data.loss_table[-1,0], 
                                                          self.usr_data.loss_table[-3,1],  self.usr_data.loss_table[-2,1],  
                                                          self.usr_data.loss_table[-1,1])
            else:
                loc= bisect.bisect(self.usr_data.loss_table[:,0], self.P_gene)
                own_consumption = cubic_spline(self.P_gene,  self.usr_data.loss_table[loc-2,0],  
                                               self.usr_data.loss_table[loc-1,0],  self.usr_data.loss_table[loc,0],
                                               self.usr_data.loss_table[loc+1,0],  self.usr_data.loss_table[loc-2,1],  
                                               self.usr_data.loss_table[loc-1,1],  self.usr_data.loss_table[loc,1],  
                                               self.usr_data.loss_table[loc+1,1])
    
            self.P_gene = self.P_gene - own_consumption
            self.loss = own_consumption
        else:
            self.P_gene = self.P
            self.loss   = 0.0
  
    def calc_opt_aero_performance(self, u = 10):
        """
        Optimum performance at specific wind velocity
        
        Parameters
        ----------
        u      :        float, [m/s]
            incoming wind velocity, default 10 m/s
        """
        opt = 1
        self.u = u
        self.opt_rpm(opt)

        opt_tsr = self.rpm * pi / 30.0 * self.rotorR / self.u
        opt_pitch = self.pitch
        opt_CP = self.CP
        opt_CT = self.CT
        opt_a  = self.a
        
        return dict(zip(['tsr','pitch','CP','CT','a'],
                        [opt_tsr, opt_pitch, opt_CP, opt_CT, opt_a]))

    def calc_static_power_curve(self): 
        """
        Calculate static power curve at given wind speed 
        (self.usr_data.u_min, self.usr_data.u_max, self.usr_data.u_step)
        """
        
        # This applies to three blades with assuming the same operating conditions.
        opt = 0
        u_min = self.usr_data.u_min
        u_max = self.usr_data.u_max + 0.1 * self.usr_data.u_step
        u_step = self.usr_data.u_step
        u   = np.arange(u_min, u_max + 0.1 * u_step, u_step)
        
        CP, CT, P, T, P_gene = [], [], [], [], []
        pitch, rpm, a, noise = [], [], [], []
        M_range, T_range     = [], []
        loss = []
        dPdpitch, dMdpitch, dTdpitch = [], [], [] 
        dPdu, dMdu, dTdu     = [], [], []
        distributed_aeros = []
        distributed_noises = []
        
        for ws in u:
            
            self.u = ws
            self.calc_rpm_factor(opt)
            self.calc_torque_factor(opt)
            self.opt_rpm(opt)
            
            CP.append(self.CP)
            P.append(self.P)
            P_gene.append(self.P_gene)
            T.append(self.T)
            CT.append(self.CT)
            pitch.append(self.pitch)
            rpm.append(self.rpm)
            a.append(self.a)
            noise.append(self.noise)
            loss.append(self.loss)
            
            gradient = self.calc_gradients()
            
            M_range.append(gradient['M_range'])
            T_range.append(gradient['T_range'])
            dPdpitch.append(gradient['dPdpitch'])
            dMdpitch.append(gradient['dMdpitch'])
            dTdpitch.append(gradient['dTdpitch'])
            dPdu.append(gradient['dPdu'])
            dMdu.append(gradient['dMdu'])
            dTdu.append(gradient['dTdu'])
            
            distributed_aero = self.distributed_aero()
            distributed_aeros.append(distributed_aero)
            distributed_noise = self.distributed_noise()
            distributed_noises.append(distributed_noise)
                    
        CP       = np.array(CP)
        P        = np.array(P)
        P_gene   = np.array(P_gene)
        CT       = np.array(CT)
        T        = np.array(T)
        pitch    = np.array(pitch)
        rpm      = np.array(rpm)
        a        = np.array(a)
        noise    = np.array(noise)
        loss     = np.array(loss)
        M_range  = np.array(M_range)
        T_range  = np.array(T_range)
        dPdpitch = np.array(dPdpitch)
        dMdpitch = np.array(dMdpitch)
        dTdpitch = np.array(dTdpitch)
        dPdu     = np.array(dPdu)
        dMdu     = np.array(dMdu)
        dTdu     = np.array(dTdu)
        
        tip_speed = rpm * pi / 30 * self.rotorR
        tsr       = tip_speed / u
        distributed_aeros = np.array(distributed_aeros)
        distributed_noises = np.array(distributed_noises)
        distributed_noises = 10 * np.log10(distributed_noises)    # atttention that noise is not equal to sum(distributed_noise) here

        return dict(zip(['u', 'tip_speed','tsr','CP', 'P', 'P_gene', 'CT', 'T', 'pitch', 'rpm', 'a', 'noise', 'loss', 'M_range', 'T_range',
                         'dPdpitch', 'dMdpitch', 'dTdpitch', 'dPdu', 'dMdu', 'dTdu','distributed_aero', 'distributed_noise' ], 
                        [u, tip_speed, tsr, CP, P, P_gene, CT, T, pitch, rpm, a, noise, loss, M_range, T_range,
                         dPdpitch, dMdpitch, dTdpitch, dPdu, dMdu, dTdu, distributed_aeros, distributed_noises]))
                                  
    def calc_turb_power_curve(self, static_curve = None):
        """
        Parameters
        ----------
        static_curve : dict, [-]
            A dictionary containing necessary static results for computing
            turbulent power curve. The default is None.

        Returns
        -------
        turb_pc   : dict, [-]
            A dictionary containing the turbulent power calculation results
            
        Notes
        -----
        The turbulence is modelled by stepping 6 standard deviations away from 
        the actual wind speed. Interpolation is then used to compute P and T.
        Thus, in order to make the interpolation reliable, the wind speed step 
        for computing the turbulent power curve should not be too large. 
        If static_curve is not given, the static power curve will be computed 
        based on a modified velocity range, where the minimum velocity is set 
        to be min(1, abs(u_min - 1)) and the u_step is set to be 
        min(1, self.usr_data.u_step).
              
        Remember, if the static curve is given as an input parameter, its velocity
        range may not be the same with those shown in the input file.
        
        """
        u_min = self.usr_data.u_min
        u_max = self.usr_data.u_max + 0.1 * self.usr_data.u_step
        u_step = self.usr_data.u_step
        u_turb   = np.arange(u_min, u_max + 0.1 * u_step, u_step)
        
        if static_curve is None:
            # the given static curve should have a wider range of incoming wind velocity
            # than those used for static curve calculation
            self.usr_data.u_min  = min(1, abs(self.usr_data.u_min - 1))
            self.usr_data.u_step = min(1, self.usr_data.u_step)
            static_curve = self.calc_static_power_curve()
            self.usr_data.u_min = u_min
            self.usr_data.u_step = u_step
            
        u_spc     = static_curve['u']
        P_gene_spc= static_curve['P_gene']
        T_spc     = static_curve['T']
        M_range   = static_curve['M_range']
        T_range   = static_curve['T_range']
        noise_spc = static_curve['noise']
            
        costi = cos(np.radians(self.usr_data.tilt))
        
        P_turb  = []
        T_turb  = []
        noise_turb = []
        CP_turb = []
        CT_turb = []
        

        for u in u_turb:
        
            TI = self.usr_data.TI
            if self.usr_data.IEC_normalized:
                TI = TI * (0.75 + 3.8 / u)
            
            if self.usr_data.rotor_averaged:
                TI = TI * 1.47 * (self.rotorR * (costi)**(1/2)) ** (-0.21)
                # TI = TI * 1.47 * (self.Rtip * cosga * (costi)**(1/2)) ** (-0.21)
                
            standard_x  = -6
            P_gene_turb =  0
            T_gene_turb =  0
            noise_gene_turb  =  0
            dx          = 0.1
            
            for standard_x in np.arange(-6, 6.001, dx):
            
                u_dx = u * (1 + standard_x * TI)
                phi    = 1 / ((2 * pi)**(1/2)) * exp(-(standard_x)**2 /2)
                P_gene_turb = P_gene_turb + phi * interpolate_vector(u_dx, u_spc, P_gene_spc)
                T_gene_turb = T_gene_turb + phi * interpolate_vector(u_dx, u_spc, T_spc)
                noise_gene_turb  = noise_gene_turb  + phi * interpolate_vector(u_dx, u_spc, noise_spc)
                
            P_gene_turb  = P_gene_turb * dx
            T_gene_turb  = T_gene_turb * dx
            noise_gene_turb   = noise_gene_turb  * dx
            q = 0.5 * self.usr_data.rho * u**2
            A = pi * self.rotorR**2  * costi**2
            CP_gene_turb = P_gene_turb / (q * A * u)
            CT_gene_turb = T_gene_turb / (q * A)
            
            P_turb.append(P_gene_turb)
            T_turb.append(T_gene_turb)
            CP_turb.append(CP_gene_turb)
            CT_turb.append(CT_gene_turb)
            noise_turb.append(noise_gene_turb)
            
        P_turb  = np.array(P_turb)
        T_turb  = np.array(T_turb)
        CP_turb = np.array(CP_turb)
        CT_turb = np.array(CT_turb)
        noise_turb = np.array(noise_turb)
            
        return dict(zip(['u_turb', 'P_turb', 'T_turb', 'CP_turb', 'CT_turb', 'M_range', 'T_range', 'noise_turb'],
                        [u_turb, P_turb, T_turb, CP_turb, CT_turb, M_range, T_range, noise_turb]))
                
    def calc_gradients(self):
        
        # Calculates the gradients of P, M and T with respect to theta and u 
        opt       = 0
        
        dpitch = 1e-6
        du     = 1e-6
    
        P_gene0 = self.P_gene
        T0      = self.T
        Mt00    = self.Mt0
        
        self.pitch = self.pitch + dpitch
        self.run_BEM(grad_calc = 1)
        self.losses(opt)
        self.pitch = self.pitch - dpitch
        dPdpitch   = (self.P_gene - P_gene0) / dpitch
        dMdpitch   = self.usr_data.B * (self.Mt0 - Mt00  ) / dpitch
        dTdpitch   = (self.T - T0) / dpitch
        
        self.u     = self.u + du
        self.run_BEM(grad_calc = 1)
        self.losses(opt)
        self.u    = self.u - du
        dPdu    = (self.P_gene - P_gene0) / du
        dMdu    = self.usr_data.B * (self.Mt0 - Mt00) / du
        dTdu    = (self.T - T0) / du
        
        
        self.run_BEM(grad_calc = 0)
        costi = cos(np.radians(self.usr_data.tilt))
        
        TI = self.usr_data.TI
        if self.usr_data.IEC_normalized:
            TI = TI * (0.75 + 3.8 / self.u)
            
        if self.usr_data.rotor_averaged:
            TI = TI * 1.47 * (self.rotorR * (costi)**(1/2)) ** (-0.21)
            # TI = TI * 1.47 * (self.Rtip * cosga * (costi)**(1/2)) ** (-0.21) 

        M_range = dMdu * TI * self.u
        T_range = dTdu * TI * self.u
        
        return dict(zip(['dPdpitch', 'dMdpitch', 'dTdpitch', 'dPdu', 'dMdu', 'dTdu', 'M_range', 'T_range'],
                        [dPdpitch, dMdpitch, dTdpitch, dPdu, dMdu, dTdu, M_range, T_range]))
        
    def calc_AEP(self, turb_curve = None, number_of_AEP_steps = 1000):
        
        u_min = self.usr_data.u_min
        u_max = self.usr_data.u_max
        
        if turb_curve is None:
            turb_pc = self.calc_turb_power_curve()
            
        u_turb  = turb_pc['u_turb']
        P_turb  = turb_pc['P_turb']
        M_range = turb_pc['M_range']
        T_range = turb_pc['T_range']

        m = 4
        AEP_step = (u_max - u_min) / number_of_AEP_steps
        
        AEP  = 0
        M_eq = 0
        T_eq = 0
        A = self.usr_data.Weibull_scale
        K = self.usr_data.Weibull_shape
        Weibull = lambda u, Weibull_scale = A, Weibull_shape = K : 1-exp(-(u/Weibull_scale)**Weibull_shape)
        
        for i in np.arange(1, number_of_AEP_steps + 1, 1):
            Wei  = Weibull(u_min+i*AEP_step)-Weibull(u_min+(i-1)*AEP_step)
            AEP  = AEP  + Wei * interpolate_vector(u_min + (i - 0.5) * AEP_step, u_turb, P_turb)        
            M_eq = M_eq + Wei * (abs(interpolate_vector(u_min + (i - 0.5) * AEP_step, u_turb, M_range )))**(m)
            T_eq = T_eq + Wei * (abs(interpolate_vector(u_min + (i - 0.5) * AEP_step, u_turb, T_range )))**(m)
        
        AEP = AEP * 8760 / 1e6   # MW/h
        M_eq = (M_eq * 20 * 8760 * 3600 / 1e7) **(1/m)
        T_eq = (T_eq * 20 * 8760 * 3600 / 1e7) **(1/m)
        
        return dict(zip(['AEP', 'M_eq', 'T_eq'],
                        [ AEP ,  M_eq ,  T_eq]))
     
    def calc_loads(self):
        
        if not hasattr(self, 'I2'):
            self.calc_area_moment()
        
        u = 14   # representative wind speed for fatigue calculation
        costi = cos(np.radians(self.usr_data.tilt))
        
        TI = self.usr_data.TI
        if self.usr_data.IEC_normalized == 1 :
            TI = TI * (0.75 + 3.8 / u)
        if self.usr_data.rotor_averaged == 1:
            TI = TI * 1.47 * (self.rotorR * (costi)**(1/2)) ** (-0.21)
        
        omega = self.usr_data.rpm_max * pi / 30
        fatigue_moments = self.usr_data.B * self.usr_data.rho * pi * TI * u * omega * self.I2
        fatigue_forces  = self.usr_data.B * self.usr_data.rho * pi * TI * u * omega * self.I1
        
        u = 17  # representative wind speed for extreme load calculation
        omega = self.usr_data.rpm_max * pi / 30 * 1.13                 
        
        NS = len(self.sections)
        pn = np.zeros(NS)
        
        coord_3d = self.define_curvature()
        cone     = coord_3d['cone']  # degree
        s        = coord_3d['s']
        z_az     = coord_3d['z']
        cone     = np.radians(cone)
        
        for i, sec in enumerate(self.sections):
            vRel = ((z_az[i] * omega )**(2) + (u * cos(cone[i]) * costi)**(2))**(1/2)  
            Re   = self.usr_data.rho * sec.c * vRel / self.usr_data.mu 
            sec.CNmax(Re, self.usr_data.blend_method)
            pn[i] = 0.5 * self.usr_data.rho * vRel**2 * sec.c * sec.CN_max
        
        extreme_forces, _              = integration(self.usr_data.B * pn * cos(cone), s) # thrust
        extreme_moments, M_flap        = integration(self.usr_data.B * pn * z_az, s)
        M_flap_single                  = M_flap / self.usr_data.B
        
        return dict(zip(['fatigue_moments', 'fatigue_forces', 'extreme_moments', 'extreme_forces','M_flap_single'],
                [fatigue_moments, fatigue_forces, extreme_moments ,  extreme_forces, M_flap_single]))
        

    def calc_CP_CT_table(self):
        
        tsr_list = np.arange(self.usr_data.tsr_min, self.usr_data.tsr_max + 
                             self.usr_data.tsr_step / 2, self.usr_data.tsr_step)
        pitch_list = np.arange(self.usr_data.pitch_min, self.usr_data.pitch_max + 
                               self.usr_data.pitch_step / 2, self.usr_data.pitch_step)
        
        n_tsr = len(tsr_list)
        n_pitch = len(pitch_list)
        
        CP_table = np.zeros([n_tsr, n_pitch])
        CT_table = np.zeros([n_tsr, n_pitch])
        
        for i in range(n_tsr):
            for j in range(n_pitch):
                self.u = 10
                self.rpm = tsr_list[i] * self.u / self.Rtip * 30 / pi
                self.pitch = pitch_list[j]
                self.run_BEM()
                CP_table[i, j] = self.CP
                CT_table[i, j] = self.CT
        
        return dict(zip(['tsr','pitch','CP','CT'],
                        [tsr_list, pitch_list, CP_table, CT_table]))
        
    def calc_CP_opt(self):
        
        """
        
        """
        opt = 1
        tsr_list = np.arange(self.usr_data.tsr_min, self.usr_data.tsr_max + 
                             self.usr_data.tsr_step / 2, self.usr_data.tsr_step)
        n_tsr = len(tsr_list)
        pitch_vector = np.zeros(n_tsr)
        CP_vector = np.zeros(n_tsr)
        CT_vector = np.zeros(n_tsr)
        a_vector = np.zeros(n_tsr)
        for i in range(n_tsr):
            self.u = 10
            self.rpm = tsr_list[i] * self.u / self.Rtip * 30 / pi
            self.opt_pitch(opt) 
            pitch_vector[i] = self.pitch
            CP_vector[i] = self.CP
            CT_vector[i] = self.CT
            a_vector[i]  = self.a
            
        return dict(zip(['tsr','pitch','CP','CT','a'],
                        [tsr_list, pitch_vector, CP_vector, CT_vector, a_vector]))
    
    def calc_stall(self):
        """
        Returns:
        --------
            stall : array_like, [-]    
        """
        stall = []
        for sec in self.sections:
            stall.append(sec.calc_stall())
        stall = np.array(stall)
        
        return stall
        
    def output(self, opt_aero = None, static_curve = None, turb_curve = None, 
                        AEP = None, loads = None, CP_CT = None, CP_opt = None ):
        
        def write(f, *args):
            for arg in args:
                f.write(str(arg))
                
        def writeln(f, *args):
            for arg in args:
                f.write(str(arg))
            f.write('\n')
        
        def table_output(table):
            # Writes CP- or CT-table to the output file
            n_pitch, n_lambda = table.shape
            write(Fil, r' lambda\theta')
            for i in range(0, n_pitch):
                write(Fil, "%11.2f" % (self.usr_data.pitch_min + i * self.usr_data.pitch_step))
            writeln(Fil)
            for j in range(0, n_lambda):
                write(Fil, "%13.2f" % (self.usr_data.tsr_min + j * self.usr_data.tsr_step))
                for i in range(0, n_pitch):
                    write(Fil, "%11.6f" % (table[i, j]))
                writeln(Fil)
            writeln(Fil)  
        
        def write_header(parameter_name):
        # Write header of detailed aerodynamic output to the output file
            Fil.write( '%10s' %parameter_name)
            Fil.write(' ')
            u   = np.arange(self.usr_data.u_min, 
                            self.usr_data.u_max + 0.1 * self.usr_data.u_step, 
                            self.usr_data.u_step)
            for ws in u:
                write(Fil, "%7.1f" % (ws), 'm/s')
            writeln(Fil)
        
        geometry = self.get_geometry()   
        output_data = os.path.dirname(os.getcwd()) +'\\case\\output' + '\\%s_%s' %(self.usr_data.name, self.usr_data.method_key) + '\\data' 
        if not os.path.exists(output_data):
            os.makedirs(output_data)
        output_path = os.path.join(output_data, self.usr_data.name + '.out' )
        Fil = open(output_path, 'w+', encoding='utf-8')
        # Write header of output file
        write(Fil, 'Output file for gBEM:           ')
        writeln(Fil, self.usr_data.name)
        writeln(Fil)
        
        # Write information about input files
        writeln(Fil, 'FILE NAMES:')
        writeln(Fil, 'Planform file name:            ', 'geometry.txt')
        writeln(Fil, 'Profile data file name         ', '     test AF')
        writeln(Fil)
        
        # Write turbine data
        writeln(Fil, 'TURBINE DATA:')
        writeln(Fil, 'Number of blades              ', "%8d"   % (self.usr_data.B))
        writeln(Fil, 'Blade precone angle           ', "%8.1f" % (self.usr_data.precone), ' deg.')
        writeln(Fil, 'Rotor tilt angle              ', "%8.1f" % (self.usr_data.tilt), ' deg.')
        write(Fil,   'Prenbend                      ')
        if np.any(geometry['precurve']) > 0:
            writeln(Fil, '     yes')
        else:
            writeln(Fil, '    no')
        writeln(Fil, 'Equivalent Rotor diameter     ', "%8.3f" % (2 * self.rotorR), ' m')
        writeln(Fil, 'Swept area                    ', "%8.0f" % (self.SR), ' m2')
        writeln(Fil, 'Specific power                ', "%8.1f" % (self.usr_data.Pgen_max / self.SR), ' W/m2')
        writeln(Fil, 'blade area                    ', "%8.2f" % (self.area), ' m2')
        writeln(Fil, 'Rotor solidity                ',
                "%8.2f" % (self.usr_data.B * self.I0 / self.SR * 100), ' %')
        writeln(Fil, '0th blade area moment         ', "%8.1f" % (self.I0), ' m2')
        writeln(Fil, '1st blade area moment         ', "%8.0f" % (self.I1), ' m3')
        writeln(Fil, '2nd blade area moment         ', "%8.0f" % (self.I2), ' m4')
        writeln(Fil, '3rd blade area moment         ', "%8.0f" % (self.I3), ' m5')
        writeln(Fil)
        
        # Write wind climate data
        writeln(Fil, 'WIND CLIMATE DATA:')
        writeln(Fil, 'Air density                   ', "%8.3f" % (self.usr_data.rho), ' kg/m3')
        write(Fil, 'Turbulence intensity          ', "%8.3f" % (self.usr_data.TI))
        if (self.usr_data.IEC_normalized == 1) and (self.usr_data.rotor_averaged == 1):
            write(Fil, ' (IEC-normalized, Rotor-averaged)')
        elif self.usr_data.IEC_normalized == 1:
            write(Fil, ' (IEC-normalized)')
        elif self.usr_data.rotor_averaged == 1:
            write(' (Rotor-averaged)')
        writeln(Fil)
        writeln(Fil, 'Weibull scale parameter       ', "%8.3f" % (self.usr_data.Weibull_scale),
                ' m/s')
        writeln(Fil, 'Weibull shape parameter       ', "%8.3f" % (self.usr_data.Weibull_shape))
        writeln(Fil)
        
        # Write calculation model options
        writeln(Fil, 'CALCULATION MODEL OPTIONS:')
        write(Fil, 'BEM equations solving            ')
        writeln(Fil, self.usr_data.method_key)
        write(Fil, 'High load correction             ')
        writeln(Fil, self.usr_data.high_load)
        write(Fil, 'Polar blend strategy            ')
        writeln(Fil, self.usr_data.blend_method)
        write(Fil, 'Automatic 3D-correction            ')
        if self.usr_data.correction_3d == 1:
            writeln(Fil, 'yes')
        else:
            writeln(Fil, ' no')
        write(Fil, 'Tip loss correction                ')
        if self.usr_data.tip_corr == 1:
            writeln(Fil, 'yes')
        else:
            writeln(Fil, ' no')
        write(Fil, 'Hub loss correction                ')
        if self.usr_data.hub_corr == 1:
            writeln(Fil, 'yes')
        else:
            writeln(Fil, ' no')
        writeln(Fil)
        
        # Write optimum aerodynamic performance data
        if self.usr_data.output_opt_aero == 1:
            opt_TSR, opt_pitch = opt_aero['tsr'], opt_aero['pitch']
            opt_CP, opt_CT, opt_a = opt_aero['CP'], opt_aero['CT'], opt_aero['a']
            writeln(Fil, 'OPTIMUM AERODYNAMIC PERFORMANCE DATA:')
            writeln(Fil, 'Optimum tip speed ratio       ', "%8.3f" % (opt_TSR))
            writeln(Fil, 'Optimum pitch angle           ', "%8.3f" % (opt_pitch),
                    ' deg.')
            writeln(Fil, 'Maximum power coefficient     ', "%8.4f" % (opt_CP))
            writeln(Fil, 'Optimum thrust coefficient    ', "%8.4f" % (opt_CT))
            writeln(Fil, 'Optimum axial induction factor', "%8.4f" % (opt_a))
            writeln(Fil)
        
        # Write operational data, power curve and other performance data etc.
        flag_oper_data = self.usr_data.output_tip_speed + self.usr_data.output_tsr + self.usr_data.output_rotor_speed+\
            self.usr_data.output_pitch_angle + self.usr_data.output_a + self.usr_data.output_CP_aero + self.usr_data.output_P_aero +\
                self.usr_data.output_loss + self.usr_data.output_Pgen + self.usr_data.output_Pgen_turb + self.usr_data.output_CP_turb +\
                    self.usr_data.output_thrust_static + self.usr_data.output_thrust_turb + self.usr_data.output_CT_static + self.usr_data.output_CT_turb +\
                        self.usr_data.output_noise + self.usr_data.output_dPdpitch + self.usr_data.output_dMdpitch +\
                        self.usr_data.output_dTdpitch + self.usr_data.output_dPdu + self.usr_data.output_dMdu + self.usr_data.output_dTdu

        if flag_oper_data > 0:
            writeln(Fil, 'OPERATIONAL DATA:')
            write(Fil, ' Wind_speed')
            oper_data = []
            fmt = "%7.1f"
            oper_data.append(static_curve['u'])
            if self.usr_data.output_tip_speed == 1:
                write(Fil, '  Tip_speed')
                oper_data.append(static_curve['tip_speed'])
                fmt += "%14.3f"
            if self.usr_data.output_tsr == 1:
                write(Fil, '     TSR')
                oper_data.append(static_curve['tsr']) 
                fmt += "%10.3f"
            if self.usr_data.output_rotor_speed == 1:
                write(Fil, '  Rotor_speed')
                oper_data.append(static_curve['rpm'])
                fmt += "%9.3f"
            if self.usr_data.output_pitch_angle == 1:
                write(Fil, '  Pitch_angle')
                oper_data.append(static_curve['pitch'])
                fmt += "%12.3f"
            if self.usr_data.output_a == 1:
                write(Fil, '    a')
                oper_data.append(static_curve['a'])
                fmt += "%12.4f"
            if self.usr_data.output_CP_aero == 1:
                write(Fil, '     CP(aero.)')
                oper_data.append(static_curve['CP'])
                fmt += "%10.4f"            
            if self.usr_data.output_P_aero == 1:
                write(Fil, '   P_aero.')
                oper_data.append(static_curve['P'] / 1000)
                fmt += "%10.3f" 
            if self.usr_data.output_loss == 1:
                write(Fil, '  losses')
                oper_data.append(static_curve['loss'] / 1000)
                fmt += "%10.3f" 
            if self.usr_data.output_Pgen == 1:
                write(Fil, '  P_gen.(static)')
                oper_data.append(static_curve['P_gene'] / 1000)
                fmt += "%10.3f"
            if self.usr_data.output_Pgen_turb == 1:
                write(Fil, '  P_gen.(turb.)')
                oper_data.append(turb_curve['P_turb'] / 1000)
                fmt += "%15.3f"
            if self.usr_data.output_CP_turb == 1:
                write(Fil, '  CP_turb.')
                oper_data.append(turb_curve['CP_turb']) 
                fmt += "%14.4f"
            if self.usr_data.output_thrust_static == 1:
                write(Fil, '  Thrust(static)')
                oper_data.append(static_curve['T'] / 1000)
                fmt += "%14.3f"
            if self.usr_data.output_thrust_turb == 1:
                write(Fil, '  Thrust(turb.)')
                oper_data.append(turb_curve['T_turb'] / 1000)
                fmt += "%15.3f"
            if self.usr_data.output_CT_static == 1:
                write(Fil, '  CT(static)')
                oper_data.append(static_curve['CT'])
                fmt += "%13.4f"
            if self.usr_data.output_CT_turb == 1:
                write(Fil, '  CT(turb.)')
                oper_data.append(turb_curve['CT_turb'])
                fmt += "%11.4f"
            if self.usr_data.output_noise == 1:
                write(Fil, '      Noise')
                oper_data.append(static_curve['noise'])
                fmt += "%13.3f"
            if self.usr_data.output_dPdpitch == 1:
                write(Fil, '    dP/dpitch')
                oper_data.append(static_curve['dPdpitch'] / 1000)
                fmt += "%13.3f"
            if self.usr_data.output_dMdpitch == 1:
                write(Fil, '    dM/dpitch')
                oper_data.append(static_curve['dMdpitch'] / 1000)
                fmt += "%13.3f"
            if self.usr_data.output_dTdpitch == 1:
                write(Fil, '    dT/dpitch')
                oper_data.append(static_curve['dTdpitch'] / 1000)
                fmt += "%13.3f"
            if self.usr_data.output_dPdu == 1:
                write(Fil, '        dP/du')
                oper_data.append(static_curve['dPdu'] / 1000)
                fmt += "%13.3f"
            if self.usr_data.output_dMdu == 1:
                write(Fil, '        dM/du')
                oper_data.append(static_curve['dMdu'] / 1000)
                fmt += "%13.3f"
            if self.usr_data.output_dTdu == 1:
                write(Fil, '        dT/du')
                oper_data.append(static_curve['dTdu'] / 1000)
                fmt += "%13.3f"

            writeln(Fil)
            write(Fil, '   [m/s]')
            if self.usr_data.output_tip_speed == 1:
                write(Fil, '       [m/s]')
            if self.usr_data.output_tsr == 1:
                write(Fil, '       [-]')
            if self.usr_data.output_rotor_speed == 1:
                write(Fil, '     [rpm]')
            if self.usr_data.output_pitch_angle == 1:
                write(Fil, '       [deg.]')
            if self.usr_data.output_a == 1:
                write(Fil, '      [-]')
            if self.usr_data.output_CP_aero == 1:
                write(Fil, '       [-]')
            if self.usr_data.output_P_aero == 1:
                write(Fil, '       [kW]')
            if self.usr_data.output_loss == 1:
                write(Fil, '     [kW]')
            if self.usr_data.output_Pgen == 1:
                write(Fil, '       [kW]')
            if self.usr_data.output_Pgen_turb == 1:
                write(Fil, '            [kW]')
            if self.usr_data.output_CP_turb == 1:
                write(Fil, '          [-]')
            if self.usr_data.output_thrust_static == 1:
                write(Fil, '         [kN]')
            if self.usr_data.output_thrust_turb == 1:
                write(Fil, '           [kN]')
            if self.usr_data.output_CT_static == 1:
                write(Fil, '           [-]')
            if self.usr_data.output_CT_turb == 1:
                write(Fil, '        [-]')
            if self.usr_data.output_noise == 1:
                write(Fil, '       [dB(A)]')
            if self.usr_data.output_dPdpitch == 1:
                write(Fil, '    [kW/deg.]')
            if self.usr_data.output_dMdpitch == 1:
                write(Fil, '   [kNm/deg.]')
            if self.usr_data.output_dTdpitch == 1:
                write(Fil, '    [kN/deg.]')
            if self.usr_data.output_dPdu == 1:
                write(Fil, '   [kW/(m/s)]')
            if self.usr_data.output_dMdu == 1:
                write(Fil, '  [kNm/(m/s)]')
            if self.usr_data.output_dTdu == 1:
                write(Fil, '   [kN/(m/s)]')

            writeln(Fil)
            oper_data = np.transpose(oper_data)
            np.savetxt(Fil, oper_data, delimiter='\t', fmt = fmt)
            self.oper_data = oper_data
            writeln(Fil)
        
        # Write Annual energy production
        if self.usr_data.output_AEP == 1:    
            
            writeln(Fil, 'ANNUAL ENERGY PRODUCTION:')
            writeln(Fil, 'Annual energy production    ', AEP['AEP'], ' MWh')
            writeln(Fil)
        
        # Write loads
        if self.usr_data.output_loads == 1:
            # begin
            writeln(Fil, 'ROTOR LOADS:')
            writeln(Fil, 'Fatigue moments             ', loads['fatigue_moments'] / 1000,
                    ' kNm')
            writeln(Fil, 'Fatigue forces              ', loads['fatigue_forces'] / 1000,
                    ' kN')
            writeln(Fil, 'Extreme moments             ', loads['extreme_moments'] / 1000,
                    ' kNm')
            writeln(Fil, 'Extreme forces              ', loads['extreme_forces']/ 1000,
                    ' kN')
            writeln(Fil, 'Equivalent moments          ', AEP['M_eq'] / 1000, ' kNm')
            writeln(Fil, 'Equivalent forces           ', AEP['T_eq'] / 1000, ' kN')
            writeln(Fil)
            writeln(Fil, 'BLADE LOADS:')
            writeln(Fil, ' Radius  M_flap(extr.)')
            writeln(Fil, '    [m]       [kNm]')
            for i in range(0, len(self.sections)):
                writeln(Fil, "%7.3f" % (self.sections[i].r), '    % .2f' %(loads['M_flap_single'][i] / 1000))
            writeln(Fil)
        
        if (self.usr_data.output_CP_table == 1) or (self.usr_data.output_CT_table == 1):
            
            writeln(Fil, 'CP- AND CT-TABLE FORMAT:')
            writeln(Fil, 'Number of pitch columns     ', '%10d' % len(CP_CT['pitch']))
            writeln(Fil, 'Number of lambda rows       ', '%10d' % len(CP_CT['tsr']))
            writeln(Fil)
            
            if self.usr_data.output_CP_table == 1:
                writeln(Fil, 'CP-TABLE:')
                table_output(CP_CT['CP'])
            if self.usr_data.output_CT_table == 1:
                writeln(Fil, 'CT-TABLE:')
                table_output(CP_CT['CT'])
               
        # Write CP-optimum curves
        if self.usr_data.output_CP_opt == 1:
            
            writeln(Fil, 'CP-OPTIMUM CURVES:')
            writeln(Fil, 'Number of lambda rows       ', '%10d' % len(CP_opt['tsr']))
            writeln(Fil)
            writeln(Fil, ' lambda   pitch        CP        CT         a')
            writeln(Fil, '    [-]  [deg.]       [-]       [-]       [-]')
            for j in range(0, len(CP_opt['tsr'])):
                writeln(Fil, "%7.2f" % (self.usr_data.tsr_min + j * self.usr_data.tsr_step),
                        "%8.3f" % (CP_opt['pitch'][j]), "%10.6f" % (CP_opt['CP'][j]),
                        "%10.6f" % (CP_opt['CT'][j]), "%10.6f" % (CP_opt['a'][j]))
        writeln(Fil)
            
        # Write detailed aerodynamic data for blade sections
        if (self.usr_data.det_output_alfa == 1) or (self.usr_data.det_output_CL == 1) \
            or (self.usr_data.det_output_CD == 1) or (self.usr_data.det_output_CP == 1) \
            or (self.usr_data.det_output_CT == 1) or (self.usr_data.det_output_a == 1) \
            or (self.usr_data.det_output_a_prime == 1) or (self.usr_data.det_output_v == 1) \
            or (self.usr_data.det_output_noise == 1):
            
            dic_aeros = {}
            for _ in static_curve['distributed_aero']:
                for k,v in _.items():
                    dic_aeros.setdefault(k,[]).append(v)
                       
            r = geometry['r']
            writeln(Fil, 'DETAILED AERODYNAMIC BLADE DATA:')
            if self.usr_data.det_output_alfa == 1:
                write_header('   alfa')
                det_alfa = np.c_[r, np.transpose(dic_aeros['alpha'])]
                det_alfa[:,1:] = np.rad2deg(det_alfa[:,1:])   # convert from rad to deg
                np.savetxt(Fil, det_alfa, fmt = '%10.3f', delimiter= '')
                writeln(Fil)
            
            if self.usr_data.det_output_phi == 1:
                write_header('   phi')
                det_phi = np.c_[r, np.transpose(dic_aeros['phi'])]
                det_phi[:,1:] = np.rad2deg(det_phi[:,1:])    # convert from rad to deg
                np.savetxt(Fil, det_phi, fmt = '%10.3f', delimiter= '')
                writeln(Fil)
                
            if self.usr_data.det_output_cl == 1:
                write_header('   cl')
                det_cl = np.c_[r, np.transpose(dic_aeros['cl'])]
                np.savetxt(Fil, det_cl, fmt = '%10.3f', delimiter= '')
                writeln(Fil)
                
            if self.usr_data.det_output_cCl == 1:
                write_header('   cCl')
                det_cCl = np.c_[r, np.transpose(dic_aeros['cCl'])]
                np.savetxt(Fil, det_cCl, fmt = '%10.3f', delimiter= '')
                writeln(Fil)
            
            if self.usr_data.det_output_cd == 1:
                write_header('   cd')
                det_cd = np.c_[r, np.transpose(dic_aeros['cd'])]
                np.savetxt(Fil, det_cd, fmt = '%10.3f', delimiter= '')
                writeln(Fil) 
            
            if self.usr_data.det_output_CP == 1:
                write_header('   CP')
                det_CP = np.c_[r, np.transpose(dic_aeros['CP'])]
                np.savetxt(Fil, det_CP, fmt = '%10.3f', delimiter= '')
                writeln(Fil)
                
            if self.usr_data.det_output_CT == 1:
                write_header('   CT')
                det_CT = np.c_[r, np.transpose(dic_aeros['CT'])]
                np.savetxt(Fil, det_CT, fmt = '%10.3f', delimiter= '')
                writeln(Fil)
                
            if self.usr_data.det_output_a == 1:
                write_header('   a')
                det_a = np.c_[r, np.transpose(dic_aeros['a'])]
                np.savetxt(Fil, det_a, fmt = '%10.3f', delimiter= '')
                writeln(Fil) 
            
            if self.usr_data.det_output_a == 1:
                write_header('   a_prime')
                det_a_prime = np.c_[r, np.transpose(dic_aeros['a_prime'])]
                np.savetxt(Fil, det_a_prime, fmt = '%10.3f', delimiter= '')
                writeln(Fil) 
            
            if self.usr_data.det_output_vRel == 1:
                write_header('  vRel')
                det_vRel = np.c_[r, np.transpose(dic_aeros['vRel'])]
                np.savetxt(Fil, det_vRel, fmt = '%10.3f', delimiter= '')
                writeln(Fil)
                
            if self.usr_data.det_output_tau == 1:
                write_header('  tau')
                det_circ = np.c_[r, np.transpose(dic_aeros['circulation'])]
                np.savetxt(Fil, det_circ, fmt = '%10.3f', delimiter= '')
                writeln(Fil)   
            
            if self.usr_data.det_output_noise == 1:
                write_header('  noise')
                det_noise = np.c_[r, np.transpose(static_curve['distributed_noise'])]
                np.savetxt(Fil, det_noise, fmt = '%10.3f', delimiter= '')
                writeln(Fil)

        Fil.close()
    
    def plot_geometry(self, path_fig = 'Blade_geomtry.png'):
        # plot blade geometry and save it to file
        
        geom = self.get_geometry()
        plt.figure(figsize=(10, 10), dpi=100)
        for i, var, ylabel in [[1, 'c', 'chord [m]'], 
                               [2, 'twist', 'twist [deg]'], 
                               [3, 'relThk', 'relative Thk [%]'],
                               [4, 'absThk', 'absoulte Thk [m]'],
                               [5, 'precurve', 'precurve [m]'],
                               [6, 'presweep', 'presweep [m]']]:
            plt.subplot(3, 2, i)
            plt.plot(geom['r'], geom[var], 'black')
            plt.xlabel('station [m]')
            plt.ylabel(ylabel)
            plt.grid(linestyle='-.')
            plt.xlim([0, None])
            plt.tight_layout()
        plt.savefig(path_fig, bbox_inched='tight')   
        
    def plot_operational_data(self, static_curve, turb_curve, path_fig = 'Operational_data.png'):
        # plot the operational data and save it to file
        # merge static and turbulent power curves
        oper_data = static_curve.copy()
        oper_data.update(turb_curve)
        oper_data['P']      = oper_data['P']      / 1000
        oper_data['P_gene'] = oper_data['P_gene'] / 1000
        oper_data['P_turb'] = oper_data['P_turb'] / 1000
        oper_data['T']      = oper_data['T']      / 1000
        oper_data['T_turb'] = oper_data['T_turb'] / 1000
        oper_data['loss']   = oper_data['loss']   / 1000
        
        plot_list =[[1,  'tip_speed', 'tip speed [m/s]'], 
                    [2,  'tsr', 'tip speed ratio'], 
                    [3,  'rpm', 'rpm [r/min]'],
                    [4,  'pitch', 'pitch [deg]'],
                    [5,  'a',     'a'],
                    [6,  'CP', 'CP(aero.)'],
                    [7,  'CP_turb', 'CP(turb.)'],
                    [8,  'P', 'P_aero [KW]'],
                    [9,  'loss',  'losses [KW]'],
                    [10, 'P_gene', 'P_gen.(static) [KW]'],
                    [11, 'P_turb', 'P_gen.(turb) [KW]'],
                    [12, 'CT', 'CT'],
                    [13, 'CT_turb', 'CT(turb.)'],
                    [14, 'T', 'Thrust(static) [KN]'],
                    [15, 'T_turb', 'Thrust(turb) [KN]'],
                    [16, 'noise', 'noise [dB(A)']]
        

        if self.usr_data.plot_oper_all == 1:
            # plot all the operational data
            # this is preparation for quick automatic plot option
            fig, _ = plt.subplots(4, 4, figsize = (16, 9), dpi = 100)
            fig.suptitle('Operational data -- %s' %self.usr_data.method_key, fontsize = 16, fontweight = 'bold', fontstyle='italic')
            for i, var, ylabel in plot_list:
                plt.subplot(4, 4, i)
                plt.plot(oper_data['u'], oper_data[var], 'black')
                plt.xlabel('wind speed [m/s]')
                plt.ylabel(ylabel)
                plt.grid(linestyle='-.')
            fig.tight_layout()
            fig.subplots_adjust(top = 0.92)
            fig.savefig(path_fig, bbox_inched='tight')   
        else:
            # plot these chosen variables given in the input file
            plot_vars = []
            if self.usr_data.plot_tip_speed == 1:
                plot_vars.append(1)
            if self.usr_data.plot_tsr       == 1:
                plot_vars.append(2)
            if self.usr_data.plot_rpm       == 1:
                plot_vars.append(3)   
            if self.usr_data.plot_pitch     == 1:
                plot_vars.append(4)
            if self.usr_data.plot_a         == 1:
                plot_vars.append(5)
            if self.usr_data.plot_CP        == 1:
                plot_vars.append(6)
            if self.usr_data.plot_CP_turb   == 1:
                plot_vars.append(7)
            if self.usr_data.plot_P         == 1:
                plot_vars.append(8)
            if self.usr_data.plot_loss      == 1:
                plot_vars.append(9)
            if self.usr_data.plot_P_gene    == 1:
                plot_vars.append(10)
            if self.usr_data.plot_P_turb    == 1:
                plot_vars.append(10)   
            if self.usr_data.plot_CT        == 1:
                plot_vars.append(12)
            if self.usr_data.plot_CT_turb   == 1:
                plot_vars.append(13)
            if self.usr_data.plot_T         == 1:
                plot_vars.append(14)
            if self.usr_data.plot_T_turb    == 1:
                plot_vars.append(15)
            if self.usr_data.plot_noise     == 1:
                plot_vars.append(16)
                
            plot_vars = np.array(plot_vars)   # now it can represent the sub-index
            fig, axs = plt.subplots(int((len(plot_vars))**(1/2)) + 1, int((len(plot_vars))**(1/2)) + 1, 
                                    figsize=(16, 9), dpi = 100)
            fig.suptitle('Operational data -- %s' %self.usr_data.method_key, fontsize = 16, fontweight = 'bold', fontstyle='italic')
            count = 0
            for i in plot_vars:
                count = count + 1
                plt.subplot(int((len(plot_vars))**(1/2)) + 1, int((len(plot_vars))**(1/2)) + 1, count)
                var_str = plot_list[i-1][1]   # find the key
                plt.plot(oper_data['u'], oper_data[var_str], 'black')
                plt.xlabel('wind speed [m/s]')
                plt.ylabel(plot_list[i-1][2])
                plt.grid(linestyle='-.')
            fig.tight_layout()
            fig.subplots_adjust(top = 0.92)
            fig.savefig(path_fig, bbox_inched='tight')  
            
        if self.usr_data.plot_P_comp == 1:
            plt.figure(figsize = (6, 6), dpi = 100)
            plt.title('Comparison of power output -- %s' %self.usr_data.method_key, fontsize = 12, fontweight = 'bold', fontstyle='italic')
            plt.plot(oper_data['u'], oper_data['P_gene'])
            plt.plot(oper_data['u'], oper_data['P_turb'])
            plt.xlabel('wind speed [m/s]')
            plt.ylabel('Power [KW]')
            plt.grid(linestyle='-.')
            legends = ['static', 'turbulent']
            plt.legend(legends, loc = 'lower right')
            output_P_comp = os.path.dirname(path_fig)
            path_P_comp = os.path.join(output_P_comp,'P_comp.png')
            plt.savefig(path_P_comp, bbox_inched='tight')
            
    def plot_detailed_data(self, static_curve, path_fig = 'detailed_data.png'):
        
        geom = self.get_geometry()
        
        dic_aeros = {}
        for _ in static_curve['distributed_aero']:
            for k,v in _.items():
                dic_aeros.setdefault(k,[]).append(v)
        
        noise_dict = {'noise':static_curve['distributed_noise']} # add the distributed noise
        dic_aeros.update(noise_dict)
        
        dic_aeros['alpha'] = np.rad2deg(dic_aeros['alpha'])
        dic_aeros['phi'] = np.rad2deg(dic_aeros['phi'])
        
        plot_list =[[1,  'alpha', 'alpha [deg]'], 
                    [2,  'phi',  'phi [deg]'],
                    [3,  'cl',   'cl'], 
                    [4,  'cd',   'cd'],
                    [5,  'CP',   'CP'],
                    [6,  'CT',   'CT'],
                    [7,  'a',    'a'],
                    [8,  'a_prime', 'a_prime'],
                    [9,  'vRel',    'vRel [m/s]'],
                    [10, 'noise',   'noise [dB(A)]'],
                    [11, 'circulation', 'circulation'],
                    [12, 'cCl',     'cCl']]
        
        if self.usr_data.plot_detail_all == 1:
            # plot all the operational data
            # this is preparation for quick automatic plot option
            fig, _ = plt.subplots(4, 3, figsize = (16, 9), dpi = 100)
            fig.suptitle('Detailed data -- %s' %self.usr_data.method_key, fontsize = 16, fontweight = 'bold', fontstyle='italic')
            for i, var, ylabel in plot_list:
                plt.subplot(4, 3, i)
                dic_aeros_var = np.transpose(dic_aeros[var])
                for col in np.arange(0, len(static_curve['u'])):
                    plt.plot(geom['r'], dic_aeros_var[:,col])
                plt.xlabel('station [m]')
                plt.ylabel(ylabel)
                plt.grid(linestyle='-.')
                if i == 1:
                    if len(static_curve['u']) % 2 == 0:
                        # even number
                        num_col = int(len(static_curve['u'])/2)
                    else:
                        # odd number
                        num_col = int(len(static_curve['u'])/2) + 1
                    legends = static_curve['u']
                    fig.legend(legends, bbox_to_anchor=(0.25, 1.02, 2.75, 0.5), loc= 'upper left',
                        ncol = num_col , mode="expand")
            fig.subplots_adjust(wspace = 0.25)
            fig.tight_layout()
            plt.subplots_adjust(top = 0.92)
            fig.savefig(path_fig, bbox_inched='tight')
        else:
            # plot these chosen variables given in the input file
            plot_vars = []
            if self.usr_data.plot_detail_alfa   == 1:
                plot_vars.append(1)
            if self.usr_data.plot_detail_phi    == 1:
                plot_vars.append(2)
            if self.usr_data.plot_detail_cl     == 1:
                plot_vars.append(3)   
            if self.usr_data.plot_detail_cd     == 1:
                plot_vars.append(4)
            if self.usr_data.plot_detail_CP     == 1:
                plot_vars.append(5)
            if self.usr_data.plot_detail_CT     == 1:
                plot_vars.append(6)
            if self.usr_data.plot_detail_a      == 1:
                plot_vars.append(7)
            if self.usr_data.plot_detail_a_prime== 1:
                plot_vars.append(8)
            if self.usr_data.plot_detail_vRel   == 1:
                plot_vars.append(9)
            if self.usr_data.plot_detail_noise  == 1:
                plot_vars.append(10)
            if self.usr_data.plot_detail_tau    == 1:
                plot_vars.append(10)   
            if self.usr_data.plot_detail_cCl    == 1:
                plot_vars.append(12)
                
            plot_vars = np.array(plot_vars)   # now it can represent the sub-index
            fig, _ = plt.subplots(int((len(plot_vars))**(1/2)) + 1, int((len(plot_vars))**(1/2)),
                                  figsize=(16, 9), dpi = 100)
            fig.suptitle('Detailed data -- %s' %self.usr_data.method_key, fontsize = 16, fontweight = 'bold', fontstyle='italic')
            count = 0
            for i in plot_vars:
                count = count + 1
                plt.subplot(int((len(plot_vars))**(1/2)) + 1, int((len(plot_vars))**(1/2)), count)
                var_str = plot_list[i-1][1]   # find the key
                dic_aeros_var = np.transpose(dic_aeros[var_str])
                for col in np.arange(0, len(static_curve['u'])):
                    plt.plot(geom['r'], dic_aeros_var[:,col])
                plt.xlabel('station [m]')
                plt.ylabel(plot_list[i-1][2])
                plt.grid(linestyle='-.')
                if i == 1:
                    if len(static_curve['u']) % 2 == 0:
                        num_col = int(len(static_curve['u'])/2)
                    else:
                        num_col = int(len(static_curve['u'])/2) + 1
                    legends = static_curve['u']
                    fig.legend(legends, bbox_to_anchor=(0.25, 1.02, 2.75, 0.5), loc= 'upper left',
                        ncol = num_col,mode="expand")
            fig.subplots_adjust(wspace = 0.25)
            fig.tight_layout()
            plt.subplots_adjust(top = 0.92)
            fig.savefig(path_fig, bbox_inched='tight')    
                
    def plot_3D_shape(self, path_fig = '3d_shape.png'):
        # function to be improved later and can be used to output the 3D coordinates
        # after the pitch axis is given
        figure = plt.figure()
        ax = Axes3D(figure)
        for sec in self.sections:
            z = sec.r
            x = sec.geom[:,0]
            y = sec.geom[:,1]
            # set the first point to be the last one as well to close the curve
            x = np.insert(x, -1, x[0])
            y = np.insert(y, -1, y[0])
            ax.plot3D(x,y,z)
        plt.savefig(path_fig, bbox_inched='tight')    

    def plot_contour(self, CP_CT, opt_aero = None, path_fig = 'contour.png'):
        
        tsr_list   = np.arange(self.usr_data.tsr_min, self.usr_data.tsr_max + 
                              self.usr_data.tsr_step / 2, self.usr_data.tsr_step)
        pitch_list = np.arange(self.usr_data.pitch_min, self.usr_data.pitch_max + 
                              self.usr_data.pitch_step / 2, self.usr_data.pitch_step)
        
        X, Y = np.meshgrid(tsr_list, pitch_list)
        tot = self.usr_data.plot_contour_CP + self.usr_data.plot_contour_CT
        fig, axs = plt.subplots(1, tot, figsize=(16,9), dpi = 200)  
        fig.suptitle('Contour map -- %s' %self.usr_data.method_key, fontsize = 16, fontweight = 'bold', fontstyle='italic')
        
        if self.usr_data.plot_contour_CP == 1:
            axs[0].set_title('CP contour', fontsize=12)
            CP_table = CP_CT['CP']
            CS  = axs[0].contour(X, Y, CP_table.T, levels=np.arange(0.001, 0.51, 0.02))
            plt.gca().clabel(CS, inline=True, fontsize=8)
            axs[0].set_xlabel('tip speed ratio')
            axs[0].set_ylabel('pitch [deg]')
            axs[0].set(xlim= (self.usr_data.tsr_min,self.usr_data.tsr_max), 
                      ylim = (self.usr_data.pitch_min,self.usr_data.pitch_max))
            if self.usr_data.plot_CP_opt == 1:
                # add the optimum CP point
                axs[0].scatter(opt_aero['tsr'], opt_aero['pitch'], marker = 'o', c = 'r', label = 'Opt CP point') 
                axs[0].legend()
            
        if self.usr_data.plot_contour_CT == 1:
            axs[1].set_title('CT contour', fontsize=12)
            CT_table = CP_CT['CT']
            CS  = axs[1].contour(X, Y, CT_table.T, levels=np.arange(0.001, 1.0, 0.05))
            plt.gca().clabel(CS, inline=True, fontsize=8)
            axs[1].set_xlabel('tip speed ratio')
            axs[1].set_ylabel('pitch [deg]')
            axs[1].set(xlim= (self.usr_data.tsr_min,self.usr_data.tsr_max), 
                    ylim = (self.usr_data.pitch_min,self.usr_data.pitch_max))
            if self.usr_data.plot_CP_opt == 1:
                # add the optimum CP point
                axs[1].scatter(opt_aero['tsr'], opt_aero['pitch'], marker = 'o', c = 'r', label = 'Opt CP point') 
                axs[1].legend()
            
        fig.tight_layout()
        plt.subplots_adjust(top = 0.88)
        plt.savefig(path_fig, bbox_inched='tight')
        
    def plot(self, static_curve = None, turb_curve = None, CP_CT = None, opt_aero = None):

        
        output_fig = os.path.dirname(os.getcwd()) +'\\case\\output' + '\\%s_%s' %(self.usr_data.name, self.usr_data.method_key) + '\\figure' 
        # create directory if .\output\figure does not exist
        if not os.path.exists(output_fig):
            os.makedirs(output_fig)
        
        if self.usr_data.plot_geometry == 1 and static_curve is not None:  
            path = os.path.join(output_fig,'geometry.png')
            self.plot_geometry(path_fig = path)
        
        if self.usr_data.plot_3D_shape == 1:
            path = os.path.join(output_fig,'3d_shape.png')
            self.plot_3D_shape(path_fig = path)
        
        flag_plot_oper = self.usr_data.plot_oper_all + self.usr_data.plot_tip_speed + self.usr_data.plot_tsr +\
            self.usr_data.plot_rpm + self.usr_data.plot_pitch + self.usr_data.plot_a + self.usr_data.plot_CP +\
                self.usr_data.plot_CP_turb + self.usr_data.plot_P + self.usr_data.plot_loss + self.usr_data.plot_P_gene +\
                    self.usr_data.plot_P_turb + self.usr_data.plot_CT + self.usr_data.plot_CT_turb + self.usr_data.plot_T +\
                        self.usr_data.plot_T_turb + self.usr_data.plot_noise + self.usr_data.plot_P_comp
        if flag_plot_oper > 0 and static_curve is not None and turb_curve is not None:
            path = os.path.join(output_fig,'Operational_data.png')
            self.plot_operational_data(static_curve, turb_curve, path_fig = path)
        
        flag_plot_detail = self.usr_data.plot_detail_all + self.usr_data.plot_detail_alfa +\
            self.usr_data.plot_detail_phi + self.usr_data.plot_detail_cl + self.usr_data.plot_detail_cl +\
                self.usr_data.plot_detail_cd + self.usr_data.plot_detail_CP + self.usr_data.plot_detail_CT +\
                    self.usr_data.plot_detail_a + self.usr_data.plot_detail_a_prime + self.usr_data.plot_detail_vRel +\
                        self.usr_data.plot_detail_noise + self.usr_data.plot_detail_tau + self.usr_data.plot_detail_cCl
        if flag_plot_detail > 0 and static_curve is not None:
            path = os.path.join(output_fig,'detailed_data.png')
            self.plot_detailed_data(static_curve, path_fig = path)
                
        if self.usr_data.plot_contour_CP == 1 or self.usr_data.plot_contour_CT == 1:
            path = os.path.join(output_fig,'contour.png')
            self.plot_contour(CP_CT, opt_aero = opt_aero, path_fig = path) 
        
        #add new feathers later
        # zoom-in plot can use ax.margin to achieve that, see
        # https://matplotlib.org/gallery/subplots_axes_and_figures/axes_margins.html#sphx-glr-gallery-subplots-axes-and-figures-axes-margins-py
     
      