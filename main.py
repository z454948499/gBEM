# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:18:32 2021

@author: Goldwind Blade Development Team
"""
import os
from matplotlib import pyplot as plt
import time

from inconfig import Input
from blade import Blade

# get the absolute path of current working directory
root_dir  = os.path.abspath('.')        
# locate the input file
input_dir = os.path.dirname(os.getcwd()) + "\\case\\input"  
# define the geometry file path
path_geometry = input_dir + "\\geometry_prebend.txt"     
# define airfoil data folder
path_af  = input_dir + "\\test AF"   
# Initilization of input parameters including turbine configuration
usr_data = Input(input_file = 'config.ini')  
# Initilization of Blade object   
blade    = Blade.init_from_file(path_geometry, path_af, usr_data)



# example 1 
opt    = blade.calc_opt_aero_performance(u = 10)    
spc    = blade.calc_static_power_curve()
tpc    = blade.calc_turb_power_curve()

# %% compute the static noise or turbulent noise
# u      = spc['u']
# noise_s = spc['noise']
# noise_t = tpc['noise_turb']
# plt.figure()
# plt.plot(u, noise_s, 'r.-', label = 'static')
# plt.plot(u, noise_t, 'b.-', label = 'turbulent')
# plt.xlabel('wind speed [m/s]')
# plt.ylabel('noise')
# plt.grid(linestyle='-.')
# plt.legend()
# %%

CP_CT  = blade.calc_CP_CT_table()
AEP    = blade.calc_AEP()
loads  = blade.calc_loads()
CP_opt = blade.calc_CP_opt()
blade.plot(static_curve = spc, turb_curve = tpc, \
            CP_CT = CP_CT, opt_aero = opt)
blade.output(opt_aero = opt, static_curve = spc, \
              turb_curve = tpc, AEP = AEP, \
              CP_CT = CP_CT, CP_opt = CP_opt, loads = loads )

# example 2
# blade.usr_data.method_key = 'wt4'
# blade.usr_data.noise_mode = 'BPM-wt4'
# blade.u     = 9
# blade.rpm   = 10.5
# blade.pitch = 0
# geometry    = blade.get_geometry()
# r = geometry['r']
# blade.run_BEM()
# blade.run_noise()
# noise2 = blade.distributed_noise()
# plt.figure()
# plt.plot(r, noise2)