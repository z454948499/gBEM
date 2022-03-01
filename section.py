# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:25:40 2021

@author: Administrator
"""

import numpy as np
from prep import blade_to_airfoil, azimuth_to_blade, hub_to_azimuth, yaw_to_hub
from numpy import sin, cos, pi, exp, sqrt, arctan
from scipy.optimize import brentq
import warnings
import BPM_functions as Bf
import bisect


class Section(object):
    
    sID = 1
    
    # polars are composed of a list of Polar class objects
    def __init__(self, polars, r = None, relThk = None, c = None, twist = None,
                 geom = None, name = None, alpha_stall = None, precurve = 0.0,
                 presweep = 0.0):
        """
        Parameters
        ----------
        polars  :   array_like, [-]
            list used to store potential multiple polar objects, number >= 1
        r       :   float, optional, [m]  
            local radius
        relThk  :   float, optional, [-]
            local relative thickness
        c       :   float, optional, [m]  
            local chord length
        twist   :   float, optional, [deg]
            local twsit angle
        geom    :   array_like, optional
            2D geometric coordinates
        name    :   str, optional, [-]
            section name
        alpha_stall : float, [deg]
            stall angle
        precurve    : float, [m]
            prebend length
        presweep    : float, [m]
            presweep length
        """
        self.polars = polars        
        self.r      = r
        self.relThk = relThk
        self.c      = c
        self.twist  = twist
        self.geom   = geom
        self.name   = name
        self.alpha_stall = alpha_stall
        self.id     = Section.sID
        self.precurve = precurve
        self.presweep = presweep
        
        Section.sID = Section.sID + 1
            
    def define_curvature(self, precone):
            
        x = [self.precurve, self.presweep, self.r]
        x_az, y_az, z_az = azimuth_to_blade(x, -precone)
        
        return x_az, y_az, z_az    
    
    def static_BEM(self,
                  u,
                  rpm,
                  Vx,
                  Vy,
                  pitch,
                  Rtip,
                  Rhub, 
                  precone,
                  tilt,
                  rho,
                  mu,
                  B, 
                  blend_method,
                  tip_corr,
                  hub_corr,
                  correction_3d,
                  high_load_corr,
                  grad_calc = 0,
                  tol = 1e-8,
                  max_iter = 1000):
        """
        Parameters
        ----------
         u          :   float, [m/s] 
             incoming wind speed
         rpm        :   float, [r/min]
             revoluation per minute
         Vx         :   float, [m/s]
             velocity in x-direction in blade coordinate system
         Vy         :   float, [m/s]
             velocity in y-direction in blade coordinate system
         pitch      :   float, [deg]
             pitch angle
         Rtip       :   float, [m] 
             tip radius
         Rhub       :   float, [m] 
             hub radius
         precone       :   float, [deg]
            local precone angle
         tilt       :   float, [deg]
            local tilt angle
         rho        :   float, [kg/m^3]
             air density
         mu         :   float, [kg/(m*s)]
             air dynamic viscosity
         B          :   int,   [-]
             blade number
         blend_method:  str,   [-]
             polar blending strategy
         tip_corr   :   int,   [-]
             if true, tip correction is applied 
         hub_corr   :   int,    [-]
             if true, hub correction is applied
         correction_3d:  int, [-]
             if true, apply 3d correction 
         grad_calc  :   int,    [-]
             if true, calculate gradient  
         tol        :   float, [-]
             tolerance for termination 
         max_iter    :   int,   [-]
             maximum allowable iteration limit, default = 1000
        """
        # Step 1: initial guess for a, a_prime
        a           = 0.0
        a_prime     = 0.0
        
        if grad_calc:
            a       = self.a
            a_prime = self.a_prime
                    
        rf          = 1.0       # Under-relaxation factor
        if high_load_corr   == 'Buhl':
            ac      = 0.4       # crit axial induction to apply high load correction
        elif high_load_corr == 'Spera':
            ac      = 1/3
        else:
            raise ValueError('Wrong high load correction model name')
        
        k_crit      = ac / ( 1 - ac)
        step        = 0
        pitch       = np.radians(pitch)
        twist       = np.radians(self.twist)
        sigma       = (B * self.c) / (2 * pi * self.r)
        omega       = rpm * pi / 30
        _, _ , z_az = self.define_curvature(precone)
        flag        = 0     # convergence flag
        
        # Step 3: check 3D correction
        if correction_3d:
            for polar in self.polars:
                polar.correction3D(c_over_r = self.c / self.r)

        while step <= max_iter:
            # Step 2: compute phi and local AoA
            phi   = np.arctan( (1 - a) * Vx / ((1 + a_prime) * Vy))
            vRel  = ((Vx * (1 - a))**2 + ((1 + a_prime) * Vy)**2 )**(0.5)
            Re   = rho * vRel * self.c / mu
            
            alpha = phi - twist - pitch    # radians

            # Step 4: load Cl and Cd (table reading)
            cl, cd = self.evaluate(Re, alpha, blend_method)
            
            # Step 5: compute Cn and Ct
            cphi, sphi = cos(phi), sin(phi)
            Cn = cl*cphi + cd*sphi
            Ct = cl*sphi - cd*cphi
            
            # tip loss correction
            Ftip = 1
            if self.r == Rtip:
                a_new       = 0     # a = 1 used by AeroDyn v15.04, which may causes singularity when computing gradient here
                a_prime_new = 0
                flag        = 1
                break
            elif tip_corr and sphi > 1e-3:
                ftip = (B/2.0)*(Rtip - self.r)/(self.r*sphi)
                Ftip = (2.0/pi)*np.arccos(exp(-ftip))
            
            # hub loss correction
            Fhub = 1
            if self.r == Rhub:
                a_new       = 0     # a = 1 used by AeroDyn v15.04, which may causes singularity when computing gradient here
                a_prime_new = 0
                flag        = 1
                break
            elif hub_corr and sphi > 1e-3:
                fhub = (B/2.0)*(self.r - Rhub)/(self.r*sphi)
                Fhub = (2.0/pi)*np.arccos(exp(-fhub))
    
            F = Ftip * Fhub

            k  = sigma*Cn/4.0/F/sphi/sphi
            kp = sigma*Ct/4.0/F/sphi/cphi
            # compute axial inudction factor
            if (phi > 0): # momentum or empirical region 
                if (k <= k_crit): # momentum state        
                    a_new = k/(1+k)           
                else: 
                    if high_load_corr == 'Buhl':  # Glauert (Buhl) correction
                        g1 = 2*F*k - (10/9-F)
                        g2 = 2*F*k - (4/3-F)*F
                        g3 = 2*F*k - (25/9-2*F)        
                        if (np.abs(g3) < 1e-6):
                            a_new = 1 - 1/2*np.sqrt(g2)
                        else:
                            a_new = (g1 - np.sqrt(g2)) / g3
                    elif high_load_corr == 'Spera':  # Spera correction
                        K = 1/k
                        a_new = 0.5 * (2 + K *(1- 2*ac)-sqrt((K*(1-2*ac)+2)**2\
                                                        +4*(K*ac**2-1)))
            else: # propeller brake region 
                if (k > 1):
                    a_new = k/(k-1)
                else:
                    a_new = 0
            
            a_prime_new = kp/(1-kp)

            a_new       = rf * a_new       + (1.0 - rf) * a
            a_prime_new = rf * a_prime_new + (1.0 - rf) * a_prime
            
            res_a       = abs(a_new-a)
            res_a_prime = abs(a_prime_new - a_prime)

            if step == max_iter or (res_a < tol and res_a_prime < tol) or grad_calc:
                a           = a_new
                a_prime     = a_prime_new
                
            if (res_a < tol and res_a_prime < tol) or grad_calc:
                step = max_iter + 1    
                flag = 1
            else:
                step        = step + 1
                a           = a_new
                a_prime     = a_prime_new               
        
        self.cl = cl
        self.cd = cd
        self.alpha   = alpha
        
        self.flag    = flag
            
        self.a       = a
        self.a_prime = a_prime
        self.phi     = phi
        self.Re      = Re
        self.vRel    = vRel
        self.pn = 0.5 * rho * self.vRel**2 * self.c * Cn
        self.pt = 0.5 * rho * self.vRel**2 * self.c * Ct
        self.CT   = B * self.pn         / (rho * u**2 * pi * z_az)  # local thrust coefficient
        self.CP   = B * self.pt * omega / (rho * u**3 * pi) 
        self.CQ   = B * self.pt         / (rho * u**2 * pi * z_az)  # local in-plane moment
        self.CRBM_flap = B * self.pn    / (rho * u**2 * pi * z_az)  # local out-of-plane bending moment            
    
    def static_iBEM(self,
                  u,
                  rpm,
                  Vx,
                  Vy,
                  pitch,
                  Rtip,
                  Rhub, 
                  precone,
                  tilt,
                  rho,
                  mu,
                  B,
                  blend_method,
                  tip_corr,
                  hub_corr,
                  correction_3d,
                  high_load_corr,
                  grad_calc = 0,
                  tol = 1e-8,
                  Steffesen_accel = True,
                  max_iter     = 800,
                  step_rf_lim = 250):
        """
        Parameters
        ----------
         u          :   float, [m/s] 
             incoming wind speed
         rpm        :   float, [r/min]
             revoluation per minute
         Vx         :   float, [m/s]
             velocity in x-direction in blade coordinate system
         Vy         :   float, [m/s]
             velocity in y-direction in blade coordinate system
         pitch      :   float, [deg]
             pitch angle
         Rtip       :   float, [m] 
             tip radius
         Rhub       :   float, [m] 
             hub radius
         precone    :   float, [deg]
            local precone angle
         tilt       :   float, [deg]
            local tilt angle
         rho        :   float, [kg/m^3]
             air density
         mu         :   float, [kg/(m*s)]
             air dynamic viscosity
         B          :   int,   [-]
             blade number
         blend_method:  str,   [-]
             polar blending strategy
         tip_corr   :   int,   [-]
             if true, tip correction is applied 
         hub_corr   :  int,    [-]
             if true, hub correction is applied
         correction_3d:  int, [-]
             if true, apply 3d correction 
         grad_calc  : int,    [-]
             if true, calculate gradient   
         tol        : float,  [-]
             tolerance for termination
         Steffesen_accel: bool, [-]
             if true, apply Steffesen accelration
         max_iter    :   int,   [-]
             maximum allowable iteration limit, default = 800
         step_rf_limit : int,  [-]
             iteration limit per relaxation factor, defauly = 250 
        """
        # Step 1: initial guess for a, a_prime
        a           = 0.0
        a_prime     = 0.0
        
        if grad_calc:
            a       = self.a
            a_prime = self.a_prime
            Steffesen_accel = False
             
        rf          = 1.0       # Under-relaxation factor
        if high_load_corr == 'Buhl':
            ac      = 0.4       # crit axial induction to apply high load correction
        elif high_load_corr == 'Spera':
            ac      = 1/3
        else:
            raise ValueError('Wrong correction model name')
            
        CT_crit     = 4 * ac * (1 - ac)
        step        = 0
        pitch       = np.radians(pitch)
        twist       = np.radians(self.twist)
        sigma       = (B * self.c) / (2 * pi * self.r)

        flag        = 0   # convergence flag
        res_a       = 1
        res_a_prime = 1
        
        # Step 3: check 3D correction
        if correction_3d:
            for polar in self.polars:
                polar.correction3D(c_over_r = self.c / self.r)
         
        while step <= max_iter:
            # Step 2: compute phi and local AoA 
            phi   = np.arctan((1 - a) * Vx / ((1 + a_prime) * Vy))
            vRel  = ((Vx * (1 - a))**2 + ((1 + a_prime) * Vy)**2 )**(0.5)
            Re   = rho * vRel * self.c / mu
            alpha = phi - twist - pitch    # radians

            # Step 4: load Cl and Cd (table reading)
            cl, cd = self.evaluate(Re, alpha, blend_method)
            
            # Step 5: compute Cn and Ct
            cphi, sphi = cos(phi), sin(phi)
            Cn = cl*cphi + cd*sphi
            Ct = cl*sphi - cd*cphi
            
            # tip loss correction
            Ftip = 1
            if self.r == Rtip:
                Ftip = 0
            elif tip_corr and sphi > 1e-3:
                ftip = (B/2.0)*(Rtip - self.r)/(self.r*sphi)
                Ftip = (2.0/pi)*np.arccos(exp(-ftip))
            
            # hub loss correction
            Fhub  = 1
            if self.r == Rhub:
                Fhub = 0
            elif hub_corr and sphi > 1e-3:
                fhub = (B/2.0)*(self.r - Rhub)/(self.r*sphi)
                Fhub = (2.0/pi)*np.arccos(exp(-fhub))
    
            F = Ftip * Fhub

            if F == 0:    # hub or tip section
                a_new       = 0     # a = 1 used by AeroDyn v15.04, which may causes singularity when computing gradient here
                a_prime_new = 0
                flag        = 1
                break
            else:
                # %%
                CT = (1 - a)**2 *Cn*sigma/sphi**2
                if CT > CT_crit * F:   
                    if high_load_corr == 'Buhl':  
                        k  = sigma*Cn/4.0/F/sphi/sphi
                        g1 = 2*F*k - (10/9-F)
                        g2 = 2*F*k - (4/3-F)*F
                        g3 = 2*F*k - (25/9-2*F)        
                        if (np.abs(g3) < 1e-6):
                            a_new = 1 - 1/2*np.sqrt(g2)
                        else:
                            a_new = (g1 - np.sqrt(g2)) / g3
                    elif high_load_corr == 'Spera':
                        K = 4 * F * sphi**2 / (sigma * Cn)
                        a_new = 0.5 * (2 + K *(1- 2* ac)-sqrt((K*(1-2*ac)+2)**2+4*(K*ac**2-1)))
                else:
                    a_new = CT/ 4 / F/ (1 - a)
                        
                a_prime_new = 1/((4*F*sphi*cphi/(sigma*Ct))-1)      
    # %%            
                if Steffesen_accel:
                    a2 = a_new
                    a_prime2 = a_prime_new
                                        
                    phi2   = arctan( (1 - a2) * Vx / ((1 + a_prime2) * Vy))
                    vRel2  = ((Vx * (1 - a2))**2 + ((1 + a_prime2) * Vy)**2 )**(0.5)
                    Re2   = rho * vRel2 * self.c / mu
                    
                    # Step 3: compute local angle of attack
                    alpha2 = phi2 - twist - pitch    # radians
                    
                    # Step 4: load Cl and Cd (table reading)
                    cl2, cd2 = self.evaluate(Re2, alpha2, blend_method)
                    
                    # Step 5: compute Cn and Ct
                    cphi2, sphi2 = cos(phi2), sin(phi2)
                    Cn2 = cl2*cphi2 + cd2*sphi2
                    
                    Ftip2 = 1
                    if tip_corr and sphi2 > 1e-3:
                        ftip = (B/2.0)*(Rtip - self.r)/(self.r*sphi2)
                        Ftip2 = (2.0/pi)*np.arccos(exp(-ftip))
    
                    # hub loss correction
                    Fhub2 = 1
                    if hub_corr and sphi2 > 1e-3:
                        fhub = (B/2.0)*(self.r - Rhub)/(self.r*sphi2)
                        Fhub2 = (2.0/pi)*np.arccos(exp(-fhub))
                    F2 = Ftip2 * Fhub2                   
                    
                    CT2 = (1 - a2)**2 *Cn2*sigma/sphi2**2  
                    if CT2 > CT_crit * F2:
                        if high_load_corr == 'Buhl':
                            k2  = sigma*Cn2/4.0/F2/sphi2/sphi2 
                            g1 = 2*F2*k2 - (10/9-F2)
                            g2 = 2*F2*k2 - (4/3-F2)*F2
                            g3 = 2*F2*k2 - (25/9-2*F2)
                            if (np.abs(g3) < 1e-6):
                                a_new2 = 1 - 1/2*np.sqrt(g2)
                            else:
                                a_new2 = (g1 - np.sqrt(g2)) / g3 
                        elif high_load_corr == 'Spera':
                            K = 4*F2*sphi2**2 / (sigma * Cn2)
                            a_new2 = 0.5 * (2 + K *(1- 2* ac)-sqrt((K*(1-2*ac)+2)**2+4*(K*ac**2-1)))
                    else:
                        a_new2 = CT2/ 4 / F2/ (1 - a2)
                                    
                    a_new = a - (a_new - a)**2/(a - 2 * a_new + a_new2)
    
                a_new       = rf * a_new       + (1.0 - rf) * a
                a_prime_new = rf * a_prime_new + (1.0 - rf) * a_prime
                                    
                res_a       = abs(a_new-a)
                res_a_prime = abs(a_prime_new - a_prime)
                
                if np.isnan(a_new) or np.isnan(a_prime_new):                   
                    if Steffesen_accel:
                        Steffesen_accel = False
                        rf = 1.0
                        a = 0
                        a_prime = 0
                        max_iter = max_iter + step
                        continue
                    else:
                        step = max_iter + 1
                        break
    
                if step == max_iter or (res_a < tol and res_a_prime < tol) or grad_calc:
                    a           = a_new
                    a_prime     = a_prime_new
                    
                if (res_a < tol and res_a_prime < tol) or grad_calc:
                    step = max_iter + 1
                    flag = 1
                    break
                else:
                    step        = step + 1
                    a           = a_new
                    a_prime     = a_prime_new     
    
                if step % step_rf_lim == 0:
                    rf = rf / 2
                    
                if step > max_iter and Steffesen_accel:
                    Steffesen_accel = False
                    rf = 1.0
                    a = 0.0
                    a_prime = 0.0
                    max_iter = max_iter + step 
        
        _, _ , z_az = self.define_curvature(precone)
        omega = rpm * pi/ 30.0
                     
        self.cl = cl
        self.cd = cd
        self.alpha   = alpha
        
        self.flag    = flag
            
        self.a       = a
        self.a_prime = a_prime
        self.phi     = phi
        self.Re      = Re
        self.vRel    = vRel
        self.pn = 0.5 * rho * self.vRel**2 * self.c * Cn
        self.pt = 0.5 * rho * self.vRel**2 * self.c * Ct
        self.CT   = B * self.pn         / (rho * u**2 * pi * z_az)  # local thrust coefficient
        self.CP   = B * self.pt * omega / (rho * u**3 * pi) 
        self.CQ   = B * self.pt         / (rho * u**2 * pi * z_az)  # local in-plane moment
        self.CRBM_flap = B * self.pn    / (rho * u**2 * pi * z_az)  # local out-of-plane bending moment                   
     
    def static_CCBlade(self,
                  u,
                  rpm,
                  Vx,
                  Vy,
                  pitch,
                  Rtip,
                  Rhub,
                  precone,
                  tilt,
                  rho,
                  mu,
                  B,
                  blend_method,
                  tip_corr,
                  hub_corr,
                  high_load_corr,
                  correction_3d,
                  grad_calc = 0,
                  tol = 1e-8,
                  max_iter = 100):
        """
        # Parameters:
        --------------
         u          :   float, [m/s] 
             incoming wind speed
         rpm        :   float, [r/min]
             revoluation per minute 
         Vx         :   float, [m/s]
             velocity component in x-direction
         Vy         :   float, [m/s]
             velocity component in y-direction
         pitch      :   float, [deg]
             pitch angle
         Rtip       :   float, [m] 
             tip radius
         Rhub       :   float, [m] 
             hub radius
         precone       :   float, [deg]
            local precone angle
         tilt       :   float, [deg]
            local tilt angle 
         B          :   int,   [-]
             blade number
         blend_method:  str,   [-]
             polar blending strategy
         mu         :   float, [kg/(m*s)]
             air dynamic viscosity
         rho        :   float, [kg/m^3]
             air density
         tip_corr   :   int,   [-]
             if true, tip correction is applied 
         hub_corr   :  int,    [-]
             if true, hub correction is applied
         correction_3d:  int, [-]
             if true, apply 3d correction 
         grad_calc  : int,    [-]
             if true, calculate gradient  
         tol        : float, optional, [-]
             tolerance for termination
         max_iter    :   int,   [-]
             maximum allowable iteration limit, default = 100
        """
        def func(phi,
                sec,
                cl,
                cd,
                Vx,
                Vy,
                Rtip,
                Rhub, 
                B,
                tip_corr,
                hub_corr,
                high_load_corr,
                correction_3d):
            
            if high_load_corr == 'Buhl':
                ac      = 0.4       # crit axial induction to apply high load correction
            elif high_load_corr == 'Spera':
                ac      = 1/3
            else:
                raise ValueError('Wrong correction model name')    
            k_crit      = ac / ( 1 - ac)

            sigma = (B * sec.c) / (2 * pi * sec.r)
            sphi, cphi = sin(phi), cos(phi)
        
            Cn = cl*cphi + cd*sphi
            Ct = cl*sphi - cd*cphi
            
            # tip loss correction
            if tip_corr and sphi > 1e-3:
                ftip = (B/2.0)*(Rtip - sec.r)/(sec.r*sphi)
                Ftip = (2.0/pi)*np.arccos(exp(-ftip))
            else:
                Ftip = 1
            # hub loss correction
            if hub_corr and sphi > 1e-3:
                fhub = (B/2.0)*(sec.r - Rhub)/(sec.r*sphi)
                Fhub = (2.0/pi)*np.arccos(exp(-fhub))
            else:
                Fhub = 1
            F = Ftip * Fhub
            
            # BEM parameters
            k = sigma*Cn/4.0/F/sphi/sphi
            kp = sigma*Ct/4.0/F/sphi/cphi
            
            # compute axial inudction factor
            if (phi > 0): # momentum or empirical region 
                if (k <= k_crit): # momentum state        
                    a = k/(1+k)
                else: 
                    if high_load_corr == 'Buhl':  # Glauert (Buhl)  correction                    
                        g1 = 2*F*k - (10/9-F)
                        g2 = 2*F*k - (4/3-F)*F
                        g3 = 2*F*k - (25/9-2*F)        
                        if (np.abs(g3) < 1e-6):
                            a = 1 - 1/2*np.sqrt(g2)
                        else:
                            a = (g1 - np.sqrt(g2)) / g3
                    elif high_load_corr == 'Spera':   # Spera correction
                        K = 4* F*sphi**2 / (sigma * Cn)
                        a = 0.5 * (2 + K *(1- 2* ac)-sqrt((K*(1-2*ac)+2)**2+4*(K*ac**2-1)))
            else: # propeller brake region 
                if (k > 1):
                    a = k/(k-1)
                else:
                    a = 0 

            # compute the tangential induction factor 
            ap = kp/(1-kp)
            
            # wake ratation swith is bypassed here 
            # error function 
            lambda_r = Vy/Vx
            if (phi > 0):
                fzero = sphi/(1-a) - cphi/lambda_r*(1-kp)
            else:
                fzero = sphi*(1-k) - cphi/lambda_r*(1-kp)
            
            return fzero, a, ap
        
        def relative_wind(phi, a, a_prime, Vx, Vy, c, twist, pitch, rho, mu):
             
            # avoid numerical errors when angle is close to 0 or 90 deg
            # and other induction factor is at some ridiculous value
            # this only occurs when iterating on Reynolds number
            # during the phi sweep where a solution has not been found yet
            if abs(a) > 10 :
                vRel = Vy*(1+a_prime)/cos(phi)
            elif abs(a_prime) > 10:
                vRel = Vx*(1-a)/sin(phi)
            else:
                vRel = sqrt((Vx*(1-a))**2 + (Vy*(1+a_prime))**2)
            
            Re = rho * vRel * c / mu
            
            pitch       = np.radians(pitch)
            twist       = np.radians(twist)
            alpha       = phi - twist - pitch # rad
            
            return alpha, vRel, Re
    
        def run_BEM(phi, sec, Vx, Vy, 
                        pitch, Rtip, Rhub, rho, mu, B, blend_method, tip_corr, hub_corr, high_load_corr, correction_3d):
            
            a       = 0
            a_prime = 0
            
            twist = sec.twist
            c     = sec.c
            alpha, vRel, Re = relative_wind(phi, a, a_prime, Vx, Vy, c, twist, pitch, rho, mu)
            
            cl, cd = sec.evaluate(Re, alpha, blend_method)
            
            args = (sec, cl, cd, Vx, Vy, Rtip, Rhub, B, tip_corr, hub_corr, high_load_corr, correction_3d)
            
            fzero, a, ap = func(phi, *args)
            
            return fzero, a, ap
            
        def errf(phi, sec, Vx, Vy,
                      pitch, Rtip, Rhub, rho, mu, B, blend_method, 
                        tip_corr, hub_corr, high_load_corr, correction_3d):
            
            fzero, _, _ = run_BEM(phi, sec, Vx, Vy,
                                  pitch, Rtip, Rhub, rho, mu, B, blend_method, tip_corr, hub_corr, high_load_corr, correction_3d)
            
            return fzero
        
        if grad_calc:
            a       = self.a
            ap      = self.a_prime
            phi     = self.phi
        
        flag = 0
        if correction_3d:
            for polar in self.polars:
                polar.correction3D(c_over_r = self.c / self.r)

        if self.r == Rhub:
            a       = 0
            ap      = 0
            phi     = np.arctan(Vx/Vy)
            flag    = 1
            self.flag = flag    
        elif self.r == Rtip:
            a       = 0
            ap      = 0
            phi     = np.arctan(Vx/Vy)
            flag    = 1
            self.flag = flag
        elif not grad_calc:    
            sec     = self
            epsilon = 1e-6
            phi_lower = epsilon
            phi_upper = pi/2
            
            args = (sec, Vx, Vy,
                     pitch, Rtip, Rhub, rho, mu, B, blend_method, tip_corr, hub_corr, high_load_corr, correction_3d)
    
            if errf(phi_lower, *args) * errf(phi_upper, *args) > 0:  # an uncommon but possible case
                if errf(-pi/4, *args) < 0 and errf(-epsilon, *args) > 0:
                    phi_lower = -pi/4
                    phi_upper = -epsilon
                else:
                    phi_lower = pi/2
                    phi_upper = pi - epsilon
    
            try:
                phi, r_converge = brentq(errf, phi_lower, phi_upper, args=args, 
                                         xtol = 1e-8, max_iter = max_iter, full_output= True)    
                if (r_converge.flag):
                    self.flag = 1
            except ValueError:
                warnings.warn('error.  check input values.')
                phi = 0.0
    
            fzero, a, ap  = run_BEM(phi, sec,
                         Vx, Vy, pitch, Rtip, Rhub, rho, mu, B, blend_method, tip_corr, hub_corr, high_load_corr, correction_3d)
        
        
        alpha_rad, vRel, Re = relative_wind(phi, a, ap, Vx, Vy, self.c, self.twist, 
                                            pitch, rho, mu)
        cl, cd = self.evaluate(Re, alpha_rad, blend_method)
        cphi = cos(phi)
        sphi = sin(phi)
        Cn = cl*cphi + cd*sphi  # these expressions should always contain drag
        Ct = cl*sphi - cd*cphi
        
        _, _ , z_az = self.define_curvature(precone)
        omega = rpm * pi/ 30.0
        
        self.cl = cl
        self.cd = cd
        self.alpha   = alpha_rad
            
        self.a       = a
        self.a_prime = ap
        self.phi     = phi
        self.vRel    = vRel
        self.Re      = Re
        self.pn      = 0.5 * rho * self.vRel**2 * self.c * Cn
        self.pt      = 0.5 * rho * self.vRel**2 * self.c * Ct
        
        self.CT   = B * self.pn         / (rho * u**2 * pi * z_az)  # local thrust coefficient
        self.CP   = B * self.pt * omega / (rho * u**3 * pi) 
        self.CQ   = B * self.pt         / (rho * u**2 * pi * z_az)  # local in-plane moment
        self.CRBM_flap = B * self.pn         / (rho * u**2 * pi * z_az)  # local out-of-plane bending moment
       
    def static_BEM_wt4(self,
                      u,
                      rpm,
                      Vx,
                      Vy,
                      pitch,
                      Rtip,
                      Rhub,
                      precone,
                      cone,
                      tilt,
                      rho,
                      mu,
                      B,
                      blend_method,
                      tip_corr,
                      hub_corr,
                      correction_3d,
                      grad_calc = 0,
                      w   = 0,
                      tol = 1e-12,
                      max_iter = 100):
        """ 
        # Parameters:
        ---------------
         u          :   float, [m/s] 
             incoming wind speed
         rpm        :   float, [r/min]
             revoluation per minute 
         Vx         :   float, [m/s]
             velocity in x-direction in blade coordinate system
         Vy         :   float, [m/s]
             velocity in y-direction in blade coordinate system
         pitch      :   float, [deg]
             pitch angle
         Rtip       :   float, [m] 
             tip radius
         Rhub    :   float, [m]
             position of blade bearing
         precone    :   float, [deg]
            local precone angle
         cone       :   float, [deg]
             local cone angle
         tilt       :   float, [deg]
            local tilt angle 
         rho        :   float, [kg/m^3]
             air density
         mu         :   float, optional [kg/(m*s)]
             air dynamic viscosity
         B          :   float, optional
             blade number
         blend_method:  str,   [-]
             polar blending strategy
         correction_3d   : bool, optional
             if true, apply 3D correction
         grad_calc  :  int, optional
             acts as a flag to control the gradients calculation
                 if 1 (in wt4: 0)   -----> gradient calculation
                 if 0 (in wt4: 1)  -----> normal BEM calculation
         w          :   float, optional, [m/s]
             induced velocity
         tol        : flaot, optional, [-]
             tolerance for termination
         max_iter    :   int,   [-]
             maximum allowable iteration limit, default = 100
             
        Notes:
        ---------------
        The solution method ignores drag during solving BEM equations, thus not 
        that accurate for elements in the root region.
        The hub correction model is introduced apart from the tip loss model in the 
        primary wt4 version.
        The calculation of normal and tangential forces, however, will take the 
        drag into consideration.
        The final local CP, CT, CQ, CRBM_flap output will not follow the definition in 
        the original codes and adapt to the prebend or presweep. Therefore, 
        be careful about the difference between the gBEM version and the primary one.
        """
        if grad_calc:
            w = self.w
        
        # costi = cos(np.radians(tilt))
        cosga = cos(np.radians(cone))
        uy    = 0            # yaw anlge = 0
        w0    = []
        dw    = []
        dw0   = []
        # max_iter = 100      # change of max_iter may result in extreme minor difference compared with wt4
        step    = 0
        omega = rpm * pi / 30
        pitch = np.radians(pitch)
        twist = np.radians(self.twist)
        _, _ , z_az = self.define_curvature(precone)
        
        flag = 0
        
        # Step 3: check 3D correction
        if correction_3d:
            for polar in self.polars:
                polar.correction3D(c_over_r = self.c / self.r)

        while step <= max_iter:    
            vRel = ((uy - Vy) ** 2 + Vx * Vx - w**2)**(1/2)     # relative velocity, assuming drag neglected 
            phi  = np.arctan(Vx /(Vy - uy)) - np.arctan(w/vRel)      # add yaw later 
            sint = sin(phi)
            cost = cos(phi)
            wz   = -w    * cost
            # vy   = -vRel * cost
            # vz   =  vRel * sint
            Re   = rho * vRel * self.c / mu
            # Step 3: compute local angle of a 
            if self.r >= Rhub:
                alpha = phi - pitch - twist # radians 
            else:
                alpha = phi - twist

            # Step 4: load Cl and Cd (table reading)
            cl, cd = self.evaluate(Re, alpha, blend_method)     

            Q = 1/2 * rho * vRel**2    # attention. slightly differ from available power
            L = Q * self.c * cl     
            a = -wz * cosga / u
            
            # tip loss correction
            Ftip = 1
            if tip_corr:
                if self.r == Rtip:
                    Ftip = 0
                elif sint > 1e-3:
                    ftip = (B/2.0)*(Rtip - self.r)/(self.r*sint)
                    Ftip = (2.0/pi)*np.arccos(exp(-ftip))

            # hub loss correction
            Fhub = 1
            if hub_corr:
                if self.r == Rhub:
                    Fhub = 0
                elif sint > 1e-3:
                    fhub = (B/2.0)*(self.r - Rhub)/(self.r*sint)
                    Fhub = (2.0/pi)*np.arccos(exp(-fhub))
                    
            F = Ftip * Fhub

            # massflow correction
            if a * F < 1/3:
                fw = 1
            else:
                fw = 1.25 - 0.75 * a * F
            
            if F == 0:
                wny = w
            else:
                wny = B * L / (4 * F * rho * pi * self.r *cosga**2 * u * (1 - fw*a))   # wt4 version
            
            dw = wny - w
            delta = abs(dw/u)
            
            # iteration criterium
            if (step == max_iter) or (delta < tol) or grad_calc:
                D  = Q * self.c * cd
                pt = L * sint - D * cost
                pn = L * cost + D * sint
                CT   = B * pn         / (rho * u**2 * pi * z_az)    # simplified by B*pn*dr/(1/2 * rho * U**2 * 2 * pi * r * dr)
                # CP = B * pt * omega / (rho * u**3 * pi * cosga)   # initial version
                CP   = B * pt * omega / (rho * u**3 * pi)
                CQ   = B * pt         / (rho * u**2 * pi * z_az)  # local in-plane moment
                CRBM_flap = B * pn         / (rho * u**2 * pi * z_az)  # local out-of-plane bending moment
            
            if(delta < tol) or grad_calc:
                flag = 1
                step = max_iter + 1    # stop iteration
            else:
                step = step + 1
                if step > 1:
                    wny = w0 - (w - w0)/(dw - dw0) * dw0
                if abs((wny - w)/u) > 0.5:
                    wny = w + dw
                if ((wny - w)/u) > 1:      # a > 1
                    wny = w + u         
                if ((wny - w)/u) < -1:     # a < -1
                    wny = w - u
            
                w0  = w
                w   = wny
                dw0 = dw
        
        self.flag = flag
        self.cl = cl
        self.cd = cd
        self.cl2cd = self.cl / self.cd
        self.alpha   = alpha
        self.w       = w       # induced resultant velocity 
        
        self.a       = a
        self.a_prime = w * sint / (2 * omega * self.r)    
        self.phi     = phi
        self.Re      = Re
        self.vRel    = vRel
        self.CP      = CP
        self.CT      = CT
        self.pt      = pt
        self.pn      = pn
        self.CQ      = CQ
        self.CRBM_flap    = CRBM_flap
        
    @classmethod
    def init_from_thickness(cls, p1, p2, thickness, blend_method):
        
        if p1.relThk is None or p2.relThk is None:
            raise ValueError("the relativeness thickness must be given before interpolation")
        else:
            weight = (thickness - p1.relThk) / (p2.relThk - p1.relThk)
 
        return [cls(p1.blend_polar(p2, weight, blend_method), relThk = thickness)]
    
    def get_XY(self):
        """
        Get airfoil geometry.

        Returns
        -------
        x : array_like
            x coordiates
        y : array_like
            y coordinates

        """
        x = self.geom[:,0]
        y = self.geom[:,1]
        return x, y
    
    def get_polar(self):
        return self.polars
    
    def get_name(self):
        return self.name
    
    def get_ID(self):
        return self.id
    
    def set_polar(self, newPolar):
        self.polars = newPolar
        
    def set_twist(self, newTwist):
        self.twist  = newTwist

    def set_chord(self, newC):
        self.c = newC
                
    def get_CP_CT(self):
        return dict(zip(['CP','CT'],[self.CP, self.CT]))
    
    def get_oper_point(self):
        return dict(zip(['alpha', 'cl', 'cd', 'phi'],
                        [self.alpha, self.cl, self.cd, self.phi]))
    
    def refresh_bl(self, mode = 'BPM-wt4', trip = 1, c0 = 337.7559):
        '''
        calculate boundary layer thickness
        
        Input:
        ------
        mode   :   str, [-]
            noise_calculation mode
        trip   :   int, optional, [-]
            if 0, untripped;
            if 1, tripped
        c0     :   float, [m/s]
            sound speed
        '''
        alfa = np.rad2deg(self.alpha)
        Re = self.Re
        Ma = self.vRel / c0
        if mode in ['BPM-xfoil', 'BPM-gfoil']:
            self.Deltas_star, self.Deltap_star = Bf.calculate_boundarylayer_thickness(
                self.geom, Re, Ma, alfa, self.c, trip, mode)
        elif mode == 'BPM-wt4':
            self.Deltas_star, self.Deltap_star = Bf.boundarylayer_thickness(Re, alfa, self.c, trip)
        else:
            raise BaseException('Invalid mode %s!' % mode)
   
    def BPM(self, Airfoil_span, mu, c0, rho, HubHeight, precone, tilt, theta,
            obserx, obsery, obserz, IEC_corr, bladeangle=0):
        """
        Parameters
        ----------
        
        Returns
        -------
        results   :   array_like
            array used to store various kinds of noise types and the total noise
            after taking the inflow atmospheric noise and  A-weighted noise 
            into consideration.
            The array will be shown as :
                [[Freq SPLs  SPLp  SPLalfa  SPLTBL SPLBL  SPLInflow  noise_total]]
        The size of results is (len(Freq), 8).
        """
    
        def te(geom, c):
            if geom is not None: 
                TE_thick =  (np.linalg.norm(geom[0] - geom[-1])) * c
                theta_low = np.arctan((geom[-2,1] - geom[-1,1]) / (geom[-2,0] - geom[-1,0]))
                theta_up  = np.arctan((geom[1,1] - geom[0,1]) / (geom[1,0] - geom[0,0]))
                TE_angle  = min(abs(np.rad2deg(theta_up - theta_low)), 14)
            else:
                TE_thick = 0.0003 * c
                TE_angle = 14
            
            return TE_thick, TE_angle
        
        alfa = np.rad2deg(self.alpha)
        # airfoil parameters：----------------
        Vlocal = self.vRel  # freestream velocity m/s
        chord  = self.c     # airfoil chord lenght/m
        relThk = self.relThk / 100  # airfoil reference thickness
        TE_thick, TE_angle = te(self.geom, self.c)
        # directiontivity setting：--------------
        P_sec =yaw_to_hub(hub_to_azimuth(azimuth_to_blade([self.precurve, self.presweep, self.r], \
                                                          -precone), -bladeangle), -tilt)\
                    + [0, 0, HubHeight]
        # unit_chord denotes the unit chord vector from the blade element leading edge to the blade element trailing edge in airfoil aligned coordinate system
        vec_chord  = -yaw_to_hub(hub_to_azimuth(azimuth_to_blade(blade_to_airfoil([0, 1, 0], -theta), -precone), -bladeangle), -tilt)
        # unit_radial denotes the unit chord vector in spanwise direction in blade aligned coordinate system
        vec_radial = -yaw_to_hub(hub_to_azimuth(azimuth_to_blade([0, 0, 1], -precone), -bladeangle), -tilt)
        
        Distance, Fie, Sitae = Bf.get_direct_paras(P_sec, vec_chord, vec_radial, obserx, obsery, obserz)
       
        Ma = Vlocal / c0
        Re = chord * Vlocal / mu
        Deltas_star = self.Deltas_star
        Deltap_star = self.Deltap_star

        Freq = np.array([
            10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
            315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
            5000, 6300, 8000, 10000, 12500, 16000, 20000
        ])
        AdB = np.array([
            -70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5,
            -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0, 0.6,
            1, 1.2, 1.3, 1.2, 1, 0.5, -0.1, -1.1, -2.5, -4.3, -6.6, -9.3
        ])
        AATM = Bf.AATM(Distance, Freq)
        Dh = Bf.Dh(Sitae, Fie, Ma)
        Dl = Bf.Dl(Sitae, Fie, Ma)

        # noise calculation-------------------------
        result = np.zeros((len(Freq), 8))
        result[:, 0] = Freq
        for k in range(len(Freq)):
            Strouhals = Freq[k] * Deltas_star / Vlocal
            Strouhalp = Freq[k] * Deltap_star / Vlocal
            Strouhal_TE = Freq[k] * TE_thick / Vlocal

            # SPLs[k],SPLp[k],SPLalfa[k],SPLTBL[k]
            result[k, 1:5] = Bf.SPLTBL(Ma, Deltas_star, Deltap_star, Strouhalp,
                                       Strouhals, Re, alfa,
                                       Airfoil_span, Dh, Distance)
            result[k, 5] = Bf.SPLBL(Ma, Deltas_star, Deltap_star, Strouhal_TE,
                                    TE_thick, relThk, TE_angle,
                                    Airfoil_span, Dh, Distance)
            result[k, 6] = Bf.SPLInflow(HubHeight, self.r, Airfoil_span,
                                     Freq[k], self.c, Vlocal, bladeangle, Distance, Ma, Dl, rho)
        result = np.clip(result, -100, None)   # replace the element in result which is smaller than -100 to -100
        # calcuate the dB(A) noise
        result[:, 7] = 10 * np.log10(np.sum(10 ** (result[:, 4:7] / 10), axis=1)) + AdB - AATM + IEC_corr
        self.noise  = np.sum(10 ** (result[:,7] / 10))
        return result
    
    def calc_noise(self, B):
        """
        Calculate the noise generated by an annular element.

        Parameters
        ----------
        B : int, [-]
            blade number
        """        
        alpha_deg = np.rad2deg(self.alpha)
        
        if alpha_deg  < -3:
            f = -0.01 * (alpha_deg  + 3) ** 2 - 1.771
        elif alpha_deg  < 25:
            f = 0.00001464*(alpha_deg +3)**4 + 0.005194*(alpha_deg +3)**2-1.771
        else:
            f = -0.001157 * (alpha_deg  - 25)**2 + 0.131 * (alpha_deg  -
                                                             25) + 11.3
            
        self.noise = B * self.vRel**5 * 10**(f/10)
            
    def calc_stall(self):
        """
        Returns
        -------
        alpha_stall : float, [deg]
            stall angle of attack
        """
        
        if len(self.polars) < 2:
            alpha_stall = self.polars[0].calc_stall()
        else:
            Re_sec = []
            for i in np.arange(len(self.polars)):
                Re_sec.append(self.polars[i].Re)
            loc = bisect.bisect(Re_sec, self.Re)    # return the right index
            if loc == len(Re_sec):  # Re >= maximum Re    
                alpha_stall = self.polars[-1].calc_stall()
            elif self.Re == Re_sec[0] or loc == 0:            # Re <= Re
                alpha_stall = self.polars[0].calc_stall()    
            else:   # use weighted interpolation of two Re
                weight = (self.Re - self.polars[loc - 1].Re) / \
                        (self.polars[loc].Re - self.polars[loc - 1].Re)     
                alpha_stall_1 = self.polars[loc-1].calc_stall()
                alpha_stall_2 = self.polars[loc].calc_stall()
                alpha_stall = (1 - weight) * alpha_stall_1 + weight * alpha_stall_2
        
        return alpha_stall

    def CNmax(self, Re, blend_method):
             
            tau =((5)**(1/2) - 1) / 2 
            
            al = np.deg2rad(0)
            au = np.deg2rad(20)
            aa = au - tau * (au - al)
            ab = al + tau * (au - al)
            
            alpha = aa
            cl, cd = self.evaluate(Re, alpha, blend_method)
            fa = cl * cos(alpha) + cd * sin(alpha)
            alpha = ab
            cl, cd = self.evaluate(Re, alpha, blend_method)
            fb = cl * cos(alpha) + cd * sin(alpha)
            
            while True:
                if (fa > fb):
                    au = ab
                    ab = aa
                    aa = au - tau * (au - al)
                    fb = fa
                    alpha = aa
                    cl, cd = self.evaluate(Re, alpha, blend_method)
                    fa = cl * cos(alpha) + cd * sin(alpha)
                else:
                    al = aa
                    aa = ab
                    ab = al + tau * (au - al)
                    fa = fb
                    alpha = ab
                    cl, cd = self.evaluate(Re, alpha, blend_method)
                    fb = cl * cos(alpha) + cd * sin(alpha)
                    
                if (np.rad2deg(ab - aa) < 1e-5 ) :
                    break
                
            self.CN_max = cl * cos(alpha) + cd * sin(alpha)
     
    def evaluate(self, Re, alpha, blend_method):
        """
        Get lift/drag coefficient at the specified Reynolds number and 
        angle of attack with linear interpolation.
        
        Parameters
        ----------
        Re    : float, [-]
            Reynolds number
        alpha : float [rad]
            angle of attack
                        
        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient
        """
        if len(self.polars) < 2:
            cl, cd = self.polars[0].evaluate(alpha)
        else:
            Re_sec = []
            for j in np.arange(len(self.polars)):
                Re_sec.append(self.polars[j].Re)
            loc = bisect.bisect(Re_sec,Re)     # represents the standard airfoil thickness, that is closest below Re
            if loc == len(Re_sec):            # Re > maximum Re
                polar = self.polars[-1]
            elif Re == Re_sec[0] or loc == 0:           # Re < minimum Re
                polar = self.polars[0]
            else:
                weight = (Re -self.polars[loc - 1].Re) / \
                    (self.polars[loc].Re - self.polars[loc - 1].Re)
                polar = self.polars[loc - 1].blend_polar(self.polars[loc], weight, blend_method)
                
            cl, cd = polar.evaluate(alpha)

        return cl, cd    
        
    def print_sec(self):
        
        if self.id is not None and self.name is not None:
            print('Section%d %s\t' %(self.id, self.name))
        else:
            raise ValueError("section name or id is not given")