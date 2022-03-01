# -*- coding: utf-8 -*-
"""
@author: GoldWind Blade Development Team
"""

import os
import configparser
import numpy as np

class Input(object):
    
    '''
    Input class for the turbine and environment parameters.
    '''   
    def __init__(self, input_file = 'config.ini'):
        
        root_dir = os.path.dirname(os.getcwd()) + "\\case\\input"
        if not os.path.exists(root_dir):
            os.makedir(root_dir)       # create file directory if not found 
        cp = configparser.ConfigParser()
        abs_path = root_dir + '\\' + input_file
        if os.path.exists(abs_path):
            cp.read(abs_path)            
            self._sections          = cp.sections()    # store all the section names, variables with prefix '_' 
        else:
            raise IOError(' Check input file name')
   
        self.name                   = cp.get('Turbine_config','name')
        self.B                      = cp.getint('Turbine_config','B')
        self.precone                = cp.getfloat('Turbine_config','precone')
        self.tilt                   = cp.getfloat('Turbine_config','tilt')
        self.hub_height             = cp.getfloat('Turbine_config','hub_height')
        self.rpm_max                = cp.getfloat('Turbine_config','rpm_max')
        self.rpm_min                = cp.getfloat('Turbine_config','rpm_min')
        self.Pgen_max               = cp.getfloat('Turbine_config','Rated_Power') 
        self.Pgen_max               = self.Pgen_max * 1000   # convert kw to w
        
        self.u_min                  = cp.getfloat('Power_curve','u_min')
        self.u_max                  = cp.getfloat('Power_curve','u_max')
        self.u_step                 = cp.getfloat('Power_curve','u_step')
        self.use_pitch_table        = cp.getint('Power_curve','use_pitch_table')
        self.pitch_table            = eval(cp.get('Power_curve','pitch_table'))
        self.use_rpm_table          = cp.getint('Power_curve','use_rpm_table')
        self.rpm_table              = eval(cp.get('Power_curve','rpm_table'))
        self.pitch_table            = np.array(self.pitch_table)   # convert list to array
        self.rpm_table              = np.array(self.rpm_table)       # convert list to array
        
        self.rho                    = cp.getfloat('Wind_climate','rho')
        self.v_sound                = cp.getfloat('Wind_climate','v_sound')
        self.mu                     = cp.getfloat('Wind_climate','mu')
        
        self.TI                     = cp.getfloat('Turbulence_config','TI')
        self.Weibull_scale          = cp.getfloat('Turbulence_config','weibull_scale')
        self.Weibull_shape          = cp.getfloat('Turbulence_config','weibull_shape')
        self.IEC_normalized         = cp.getint('Turbulence_config','IEC_normalized')
        self.rotor_averaged         = cp.getint('Turbulence_config','Rotor_averaged')

        self.pitch_max              = cp.getfloat('CP_CT_table','pitch_max')
        self.pitch_min              = cp.getfloat('CP_CT_table','pitch_min')
        self.pitch_step             = cp.getfloat('CP_CT_table','pitch_step')
        self.tsr_max                = cp.getfloat('CP_CT_table','tsr_max')
        self.tsr_min                = cp.getfloat('CP_CT_table','tsr_min')
        self.tsr_step               = cp.getfloat('CP_CT_table','tsr_step')

        self.use_loss               = cp.getint('Loss', 'use_loss')
        self.loss_table             = eval(cp.get('Loss','loss_table'))
        self.loss_table             = np.array(self.loss_table) # convert list to array
        self.loss_table             = self.loss_table * 1000    # covert kw to w
        
        self.noise_reduction        = cp.getint('Noise','Noise_reduction')
        self.noise_limit            = cp.getfloat('Noise','Noise_limit')
        self.u_noise_limit          = cp.getfloat('Noise','U_Noise_limit')
        self.obserx                 = cp.getfloat('Noise','obserx')
        self.obsery                 = cp.getfloat('Noise','obsery')
        self.obserz                 = cp.getfloat('Noise','obserz')
        self.noise_mode             = cp.get('Noise','noise_mode')
        self.noise_trip             = cp.getint('Noise','noise_trip')

        self.use_rpm_limiter        = cp.getint('Limiter','use_rpm_limiter')
        self.rpm_limiter_table      = eval(cp.get('Limiter','rpm_limiter_table'))
        self.use_torque_limiter     = cp.getint('Limiter','use_torque_limiter')
        self.torque_limiter_table   = eval(cp.get('Limiter','torque_limiter_table'))
        
        self.correction_3d          = cp.getint('BEM','correction_3D')
        self.tip_corr               = cp.getint('BEM', 'tip_corr')
        self.hub_corr               = cp.getint('BEM', 'hub_corr')
        self.method_key             = cp.get('BEM', 'method_key')
        self.high_load              = cp.get('BEM', 'high_load')
        
        self.blend_method           = cp.get('Miscellaneous','blend_method')
        
        self.output_opt_aero        = cp.getint('Output_operational_data','output_opt_aero')
        self.output_tip_speed       = cp.getint('Output_operational_data','output_tip_speed')
        self.output_tsr             = cp.getint('Output_operational_data','output_tsr')
        self.output_rotor_speed     = cp.getint('Output_operational_data','output_rotor_speed')
        self.output_pitch_angle     = cp.getint('Output_operational_data','output_pitch_angle')
        self.output_a               = cp.getint('Output_operational_data','output_a')
        self.output_CP_aero         = cp.getint('Output_operational_data','output_tsr')
        self.output_P_aero          = cp.getint('Output_operational_data','output_P_aero')
        self.output_loss            = cp.getint('Output_operational_data','output_loss')
        self.output_Pgen            = cp.getint('Output_operational_data','output_Pgen')
        self.output_Pgen_turb       = cp.getint('Output_operational_data','output_Pgen_turb')
        self.output_CP_turb         = cp.getint('Output_operational_data','output_CP_turb')
        self.output_thrust_static   = cp.getint('Output_operational_data','output_thrust_static')
        self.output_thrust_turb     = cp.getint('Output_operational_data','output_thrust_turb')
        self.output_CT_static       = cp.getint('Output_operational_data','output_CT_static')
        self.output_CT_turb         = cp.getint('Output_operational_data','output_CT_turb')
        self.output_noise           = cp.getint('Output_operational_data','output_noise')
        self.output_dPdpitch        = cp.getint('Output_operational_data','output_dPdpitch')
        self.output_dMdpitch        = cp.getint('Output_operational_data','output_dMdpitch')
        self.output_dTdpitch        = cp.getint('Output_operational_data','output_dTdpitch')
        self.output_dPdu            = cp.getint('Output_operational_data','output_dPdu')
        self.output_dMdu            = cp.getint('Output_operational_data','output_dMdu')
        self.output_dTdu            = cp.getint('Output_operational_data','output_dTdu')
        self.output_loads           = cp.getint('Output_operational_data','output_loads')
        self.output_CP_table        = cp.getint('Output_operational_data','output_CP_table')
        self.output_CT_table        = cp.getint('Output_operational_data','output_CT_table')
        self.output_CP_opt          = cp.getint('Output_operational_data','output_CP_opt')
        self.output_AEP             = cp.getint('Output_operational_data','output_AEP')
        
        self.det_output_alfa        = cp.getint('Detailed_output','det_output_alfa')
        self.det_output_phi         = cp.getint('Detailed_output','det_output_phi')
        self.det_output_cl          = cp.getint('Detailed_output','det_output_cl')
        self.det_output_cCl         = cp.getint('Detailed_output','det_output_cCl')
        self.det_output_cd          = cp.getint('Detailed_output','det_output_cd')
        self.det_output_CP          = cp.getint('Detailed_output','det_output_CP')
        self.det_output_CT          = cp.getint('Detailed_output','det_output_CT')
        self.det_output_a           = cp.getint('Detailed_output','det_output_a')
        self.det_output_a_prime     = cp.getint('Detailed_output','det_output_a_prime')
        self.det_output_vRel        = cp.getint('Detailed_output','det_output_vRel')
        self.det_output_tau         = cp.getint('Detailed_output','det_output_tau')
        self.det_output_noise       = cp.getint('Detailed_output','det_output_noise')
                
        self.plot_af                = cp.getint('Plot_geometry','plot_af')
        self.plot_geometry          = cp.getint('Plot_geometry','plot_geometry')
        self.plot_3D_shape          = cp.getint('Plot_geometry','plot_3D_shape')
              
        self.plot_oper_all          = cp.getint('Plot_operational_data','plot_oper_all')   
        self.plot_tip_speed         = cp.getint('Plot_operational_data','plot_tip_speed')
        self.plot_tsr               = cp.getint('Plot_operational_data','plot_tsr')
        self.plot_rpm               = cp.getint('Plot_operational_data','plot_rpm')
        self.plot_pitch             = cp.getint('Plot_operational_data','plot_pitch')
        self.plot_a                 = cp.getint('Plot_operational_data','plot_a')
        self.plot_pitch             = cp.getint('Plot_operational_data','plot_pitch')
        self.plot_CP                = cp.getint('Plot_operational_data','plot_CP')
        self.plot_CP_turb           = cp.getint('Plot_operational_data','plot_CP_turb')
        self.plot_P                 = cp.getint('Plot_operational_data','plot_P')
        self.plot_loss              = cp.getint('Plot_operational_data','plot_loss')
        self.plot_P_gene            = cp.getint('Plot_operational_data','plot_P_gene')
        self.plot_P_turb            = cp.getint('Plot_operational_data','plot_P_turb')
        self.plot_CT                = cp.getint('Plot_operational_data','plot_CT')
        self.plot_CT_turb           = cp.getint('Plot_operational_data','plot_CT_turb')
        self.plot_T                 = cp.getint('Plot_operational_data','plot_T')
        self.plot_T_turb            = cp.getint('Plot_operational_data','plot_T_turb')
        self.plot_noise             = cp.getint('Plot_operational_data','plot_noise')
        self.plot_P_comp            = cp.getint('Plot_operational_data','plot_P_comp')
        
        self.plot_detail_all        = cp.getint('Plot_detail','plot_detail_all')
        self.plot_detail_alfa       = cp.getint('Plot_detail','plot_detail_alfa')
        self.plot_detail_phi        = cp.getint('Plot_detail','plot_detail_phi')
        self.plot_detail_cl         = cp.getint('Plot_detail','plot_detail_cl')
        self.plot_detail_cd         = cp.getint('Plot_detail','plot_detail_cd')
        self.plot_detail_CP         = cp.getint('Plot_detail','plot_detail_CP')
        self.plot_detail_CT         = cp.getint('Plot_detail','plot_detail_CT')
        self.plot_detail_a          = cp.getint('Plot_detail','plot_detail_a')
        self.plot_detail_a_prime    = cp.getint('Plot_detail','plot_detail_a_prime')
        self.plot_detail_vRel       = cp.getint('Plot_detail','plot_detail_vRel')
        self.plot_detail_noise      = cp.getint('Plot_detail','plot_detail_noise')
        self.plot_detail_tau        = cp.getint('Plot_detail','plot_detail_tau')
        self.plot_detail_cCl        = cp.getint('Plot_detail','plot_detail_cCl')
               
        self.plot_contour_CP        = cp.getint('Plot_contour','plot_contour_CP')
        self.plot_contour_CT        = cp.getint('Plot_contour','plot_contour_CT')
        self.plot_CP_opt            = cp.getint('Plot_contour','plot_CP_opt')
        
    def print_sections(self):
        print([sec_name for sec_name in self._sections])
        
    