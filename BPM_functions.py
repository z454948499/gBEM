#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import sin, cos, tan, pi, sqrt, exp, arctan, arccos, log, log10, abs, deg2rad
# from numba import njit
import os


def Amax(a):
    if a < 0.13:
        Amax = sqrt(67.552 - 886.788 * a * a) - 8.219
    elif a >= 0.13 and a <= 0.321:
        Amax = -15.901 * a + 1.098
    else:
        Amax = -4.669 * a**3 + 3.491 * a**2 - 16.699 * a + 1.149
    return Amax


def Amin(a):
    if a < 0.204:
        Amin = sqrt(67.552 - 886.788 * a**2) - 8.219
    elif a >= 0.204 and a <= 0.244:
        Amin = -32.665 * a + 3.981
    else:
        Amin = -142.795 * a**3 + 103.656 * a**2 - 57.757 * a + 6.006
    return Amin


def Bmax(b):
    if b < 0.10:
        Bmax = sqrt(16.888 - 886.788 * b**2) - 4.109
    elif b >= 0.10 and b <= 0.187:
        Bmax = -31.330 * b + 1.854
    else:
        Bmax = -80.541 * b**3 + 44.174 * b**2 - 39.381 * b + 2.344
    return Bmax


def Bmin(b):
    if b < 0.13:
        Bmin = sqrt(16.888 - 886.788 * b**2) - 4.109
    elif b >= 0.13 and b <= 0.145:
        Bmin = -83.607 * b + 8.138
    else:
        Bmin = -817.810 * b**3 + 355.210 * b**2 - 135.024 * b + 10.619
    return Bmin


def Dh(Sitae, Fie, Ma):
    # Mc=0.8Ma
    Dh = 2 * (sin(Sitae / 2))**2 * (sin(Fie))**2 / (1 + Ma * cos(Sitae)) / (
        1 + 0.2 * Ma * cos(Sitae))**2
    return Dh


def Dl(Sitae, Fie, Ma):
    Dl = (sin(Sitae))**2 * (sin(Fie))**2 / (1 + Ma * cos(Sitae))**4
    return Dl


# 湍流边界层尾缘噪声，公式来自《Airfoil Self-Noise and Prediction》
# 20200513赵雄


def SPLTBL(Ma, Deltas_star, Deltap_star, Strouhalp, Strouhals, Reynold_chord,
           Alpha, Airfoil_span, Dh, Distance):
    St1 = 0.02 * Ma**(-0.6)
    if Alpha < 1.33:
        St2 = St1
    elif Alpha >= 1.33 and Alpha <= 12.5:
        St2 = St1 * 10**(0.0054 * (Alpha - 1.33)**2)
    else:
        St2 = St1 * 4.72

    St12 = (St1 + St2) / 2
    if Reynold_chord < 2.47 * 10**5:
        K1 = -4.31 * log10(Reynold_chord) + 156.3
    elif Reynold_chord >= 2.47 * 10**5 and Reynold_chord <= 8 * 10**5:
        K1 = -9 * log10(Reynold_chord) + 181.6
    else:
        K1 = 128.5
    dK1 = 10 * log10(
        Deltap_star / Deltas_star)  # △K1表达式，参考柏宝红《翼型湍流尾缘噪声半经验预测公式改进》

    Gamma = 27.094 * Ma + 3.31
    Gamma0 = 23.43 * Ma + 4.651
    beta = 72.65 * Ma + 10.74
    beta0 = -34.19 * Ma - 13.82

    if Alpha < (Gamma0 - Gamma):
        K2 = K1 - 1000
    elif Alpha >= (Gamma0 - Gamma) and Alpha <= (Gamma0 + Gamma):
        K2 = K1 + sqrt(beta**2 - (beta / Gamma)**2 *
                       (Alpha - Gamma0)**2) + beta0
    else:
        K2 = K1 - 12

    if Reynold_chord < 9.52 * 10**4:
        a0 = 0.57
        b0 = 0.3
    elif Reynold_chord >= 9.52 * 10**4 and Reynold_chord <= 8.57 * 10**5:
        a0 = (-9.57 * 10**(-13)) * (Reynold_chord - 8.57 * 10**5)**2 + 1.13
        b0 = (-4.48 * 10**(-13)) * (Reynold_chord - 8.57 * 10**5)**2 + 0.56
    else:
        a0 = 1.13
        b0 = 0.56
# 压力面噪声SPLp：
    a = abs(log10(Strouhalp / St1))
    Amina = Amin(a)
    Amaxa = Amax(a)
    Amina0 = Amin(a0)
    Amaxa0 = Amax(a0)
    Ara0 = (-20 - Amina0) / (Amaxa0 - Amina0)
    Aa = Amina + Ara0 * (Amaxa - Amina)
    SPLp = 10 * log10(Deltap_star * Ma**5 * Airfoil_span * Dh /
                      Distance**2) + Aa + (K1 - 3) + dK1
    # 吸力面噪声SPLs:
    a = abs(log10(Strouhals / St12))
    Amina = Amin(a)
    Amaxa = Amax(a)
    Amina0 = Amin(a0)
    Amaxa0 = Amax(a0)
    Ara0 = (-20 - Amina0) / (Amaxa0 - Amina0)
    Aa = Amina + Ara0 * (Amaxa - Amina)
    SPLs = 10 * log10(
        Deltas_star * Ma**5 * Airfoil_span * Dh / Distance**2) + Aa + (K1 - 3)
    # 分离噪声SPLALFA：
    b = abs(log10(Strouhals / St2))
    Bminb = Bmin(b)
    Bmaxb = Bmax(b)
    Bminb0 = Bmin(b0)
    Bmaxb0 = Bmax(b0)
    Brb0 = (-20 - Bminb0) / (Bmaxb0 - Bminb0)
    Bb = Bminb + Brb0 * (Bmaxb - Bminb)
    SPLalfa = 10 * log10(
        Deltas_star * Ma**5 * Airfoil_span * Dh / Distance**2) + Bb + K2
    # 湍流边界层尾缘噪声SPLTBL：
    SPLTBL = 10 * log10(10**(SPLp / 10) + 10**(SPLs / 10) + 10**(SPLalfa / 10))
    return (SPLs, SPLp, SPLalfa, SPLTBL)


# 钝尾缘噪声，公式来自《2016-zhu-Improvement of airfoil trailingedge bluntness noise model》
# 20200512赵雄 朱卫军在BPM原公式的基础上进行了修正


def SPLBL(Ma, Deltas_star, Deltap_star, Strouhal_TE, TE_thick,
          Airfoil_refthick, TE_angle, Airfoil_span, Dh, Distance):

    Delta_avg = 0.5 * (Deltas_star + Deltap_star)
    a = TE_thick / Delta_avg

    if a >= 0.2:
        Stpeak = 0.149 / (1 + 0.235 / a - 0.0132 / a**2)
    else:
        Stpeak = 0.1 * a + 0.06
    fi = log10(Strouhal_TE / Stpeak)
    h = Airfoil_refthick
    S2 = 654.43 * h**3 - 652.26 * h**2 + 58.77 * h

    a_14 = a
    fi_14 = fi
    G5_14 = G5fun(a_14, fi_14)
    a_0 = 6.724 * a**2 - 4.019 * a + 1.107
    fi_0 = fi
    G5_0 = G5fun(a_0, fi_0)
    G5 = G5_0 + 0.07143 * TE_angle * (G5_14 - G5_0)  # 在尾缘角度0°和14°之间插值计算

    if a >= 0.2:
        K0 = 150 - 20.0 * (a - 0.2)**(0.25)
    else:
        K0 = 150

    SPLBL = 10 * log10(2 * TE_thick * Ma**5.7 * Airfoil_span * Dh / Distance **
                       2) + 20 * (1 + Ma**2) * log10(a) + G5 + S2 + K0

    if SPLBL < -100:
        SPLBL = -100
    return (SPLBL)


def G5fun(a, fi):
    if a < 0.25:
        miu = 0.1221
    elif a >= 0.25 and a <= 0.62:
        miu = -0.2175 * a + 0.1755
    elif a >= 0.62 and a < 1.15:
        miu = -0.0308 * a + 0.0596
    else:
        miu = 0.0242

    if a <= 0.02:
        mm = 0
    elif a > 0.02 and a <= 0.5:
        mm = 68.724 * a - 1.35
    elif a > 0.5 and a <= 0.62:
        mm = 308.475 * a - 121.23
    elif a > 0.62 and a <= 1.15:
        mm = 224.811 * a - 69.35
    elif a > 1.15 and a <= 1.2:
        mm = 1583.28 * a - 1631.59
    else:
        mm = 268.344

    if mm < 0:
        mm = 0
    else:
        mm = mm

    fi0 = -sqrt(mm**2 * miu**4 / (6.25 + mm**2 * miu**2))
    kk = 2.5 * sqrt(1 - (fi0 / miu)**2) - 2.5 - mm * fi0

    if fi < fi0:
        G5 = mm * fi + kk
    elif fi > fi0 and fi < 0:
        G5 = 2.5 * sqrt(1 - (fi / miu)**2) - 2.5
    elif fi >= 0 and fi < 0.03616:
        G5 = sqrt(1.5625 - 1194.99 * fi**2) - 1.25
    else:
        G5 = -155.543 * fi + 4.375

    return (G5)
      
def get_direct_paras(P_sec, vec_chord, vec_radial, obserx, obsery, obserz):
    
    # coord of observer
    P_obser  = np.array([obserx, obsery, obserz])
    # distance between section and observer
    Distance = np.linalg.norm(P_sec - P_obser)
    Sitae    = arccos((P_obser - P_sec) @ vec_chord / (Distance))
    
    skalar = (P_obser - P_sec) @ vec_chord        # Distance * cos(Sitae)
    P_obs_vec_radial = skalar * vec_chord + P_sec - P_obser    # subtract the projected vector
    
    Fie = arccos((P_obs_vec_radial @ vec_radial) / np.linalg.norm(P_obs_vec_radial))

    return (Distance, Fie, Sitae)


# 叶尖涡噪声，不计算


def SPLTIPfun(Att, Distance, Dh, FreQ, TipChord, Ma):
    l = TipChord * 0.008 * Att
    MaMAX = Ma * (1 + 0.036 * Att)
    UMAX = 340 * MaMAX
    St = FreQ * l / UMAX

    SPLTIP_value = 10 * log10(Ma**2 * MaMAX**3 * l**2 * Dh /
                              Distance**2) - 30.5 * (log10(St) + 0.3)**2 + 126

    return max(SPLTIP_value, -100)


# 来流湍流噪声，公式来自《2004zhu的硕士论文《Modelling Of Noise From Wind Turbines》 20200511赵雄——计算叶片使用


def SPLInflow(HubHeight, RSpan, Airfoil_span, FreQ, Chord, Vlocal, bladeangle,
              Distance, Ma, Dl, rho):
    z = HubHeight + RSpan * cos(deg2rad(bladeangle))
    z0 = 0.05
    rr = 0.24 + 0.096 * log(z0) + 0.016 * log(z0)**2
    ii = rr * log10(30 / z0) / log10(z / z0)
    LL = 25 * z**0.35 * z0**(-0.063)
    kk = pi * FreQ * Chord / Vlocal

    result = 10 * log10(Dl * rho**2 * 340**2 * LL * Airfoil_span * Ma**3 *
                        ii**2 / Distance**2 * kk**3 /
                        (1 + kk**2)**(7 / 3)) + 58.4  # zx改：去掉Dh*换成Dl
    if FreQ < 1000:  # 当频率小于1000Hz时，加入低频修正因子LFC
        beta2 = 1 - Ma**2
        ss2 = (2 * pi * kk / beta2 + (1 + 2.4 * kk / beta2)**(-1))**(
            -1)  # 可压缩Sears函数
        LFC = 10 * ss2 * Ma * kk**2 / beta2

        result = result + 10 * log10(LFC / (1 + LFC))
    return max(result, -100)


# 层流边界层涡脱落噪声，不计算
# def SPLINFLOW()
# def SPLTIP()
# def SPLLBL()


# 大气损失
def AATM(Distance, FreQ):

    Pa = 89150  # 大气压
    Pr = 101325  # 参考大气压
    Ta = 15  # 摄氏度，环境温度
    TaK = Ta + 273.15  # 开
    Tr = 273.16
    hrel = 20  # 相对湿度

    #Va = -6.8346 * (273.16 / (273.15 + Ta)) ** 1.261 + 4.6151
    Va = 10.79586 * (1 - 273.16 / TaK) - 5.02808 * log10(
        TaK / 273.16) + 1.50474 * 0.0001 * (
            1 - 10**(-8.29692 * (TaK / 273.16 - 1))) + 0.42873 * 0.001 * (
                -1 + 10**(4.76955 * (1 - 273.16 / TaK))) - 2.2195983
    PsatdPr = 10**Va
    h = hrel * PsatdPr * (Pa / Pr)**(-1)
    frN = (TaK / Tr) ** (-1 / 2) * (Pa / Pr) * \
        (9 + 280 * h * exp(-4.17 * ((TaK / Tr) ** (-1 / 3) - 1)))
    fro = (Pa / Pr) * (24 + ((4.04 * 10**4 * h) * (0.02 + h) /
                             (0.391 + h)))  # 0.0001应该是10**4，看错了个负号

    p1 = 1.84 * 10**(-11) * (Pa / Pr)**(-1) * (TaK / Tr)**(1 / 2)
    p2 = 0.01275 * exp(-2239.1 / TaK) * (fro / (fro**2 + FreQ**2))
    p3 = 0.1068 * exp(-3352 / TaK) * (frN / (frN**2 + FreQ**2))

    ALFAa = 8.686 * FreQ**2 * (p1 + (TaK / Tr)**(-5 / 2) * (p2 + p3))

    AATM = ALFAa * Distance

    return AATM


def calculate_boundarylayer_thickness(airfoil, Re, Ma, alfa, c, trip=0, mode='BPM-gfoil'):
    '''
    calculate boundary layer thickness by xfoil\gfoil
    '''
    # 生成Xfoil执行命令文件---------------------------------------------
    xfoilinput_filename = 'xfoil.input'
    if mode == 'BPM-xfoil':
        xfoil_input_template = '\n'.join([
            'LOAD', '{airfoil_name}.geo', '', 'PLOP', 'G', '', 'PPAR', 'N 180',
            'P 1', 'T 1.500000e-01', 'R 2.000000e-01', 'XT 1.000000 1.000000',
            'XB 1.000000 1.000000', '', '', 'OPER', 'VISC {Re}', 'MACH {Ma}',
            'ITER 400', 'VPAR', 'XTR {XTR} {XTR}', 'N 7.000000', 'LAG 5.600000 0.9',
            'GB  6.750000 0.950000', 'CTR 1.800000 3.300000', '', 'ALFA {alfa}',
            'DUMP', '{airfoil_name}.bl', '', 'quit', ''
        ])
    elif mode == 'BPM-gfoil':
        xfoil_input_template = '\n'.join([
            'LOAD', '{airfoil_name}.geo', '', '', 'PPAR', 'N 180', 'P 1',
            'T 1.500000e-01', 'R 2.000000e-01', 'XT 1.000000 1.000000',
            'XB 1.000000 1.000000', '', '', 'PLOP', 'G', '', '', 'OPER', 'VISC {Re}', 'MACH {Ma}',
            'ITER 400', 'VPAR', 'XTR {XTR} {XTR}', 'N 7.000000', 'LAG 5.600000 0.9', '',
            'GB  6.750000 0.950000', 'CTR 1.800000 3.300000', '', '', '',
            'alfa {alfa}', 'DUMP', '{airfoil_name}.bl', '', 'quit', ''
        ])
    else:
        raise BaseException('Invalid mode %s!' % mode)
    if trip == 0:
        XTR = 1
    else:
        XTR = -0.02
    airfoil_name = 'temp'
    np.savetxt(airfoil_name + '.geo', airfoil)
    with open(xfoilinput_filename, 'w+') as file:
        file.write(
            xfoil_input_template.format(airfoil_name=airfoil_name,
                                        alfa=alfa,
                                        Re=Re,
                                        Ma=Ma,
                                        XTR=XTR))
    os.system(r'xfoil < xfoil.input')

    [dss, dps] = np.genfromtxt(airfoil_name + '.bl', usecols=(4, ))[[0, 179]]
    os.remove(xfoilinput_filename)
    os.remove(airfoil_name + '.geo')
    os.remove(airfoil_name + '.bl')
    return c * dss, c * dps


def boundarylayer_thickness(Re, alfa, c, trip=1):
    alfa_ABS = abs(alfa)
    # UNTRIPPED
    if trip == 0:
        delta_star_0 = 10**((3.0187 - 1.5397 * log10(Re) + 0.1059 * log10(Re)**2)) * c
    # TRIPPED
    else:
        if Re <= 300000:
            delta_star_0 = 0.0601 * Re **(-0.114) * c
        else:
            delta_star_0 = 10**((3.411 - 1.5397 * log10(Re) + 0.1059 * log10(Re)**2)) * c

    delta_star_P = 10**((-0.0432 * alfa_ABS + 0.00113 * alfa_ABS**2)) * delta_star_0
    # UNTRIPPED
    if trip == 0:
        if alfa_ABS <= 7.5:
            delta_star_S = 10**((0.0679 * alfa_ABS)) * delta_star_0
        elif (alfa_ABS <= 12.5):
            delta_star_S = 0.0162 * 10**((0.3066 * alfa_ABS)) * delta_star_0
        elif (alfa_ABS <= 25):
            delta_star_S = 52.42 * 10**((0.0258 * alfa_ABS)) * delta_star_0
        # above AoA of 25 degree calculate the shadow area thickness of the airfoil and use as boundary layer thickness
        else:
            delta_star_S = c * sin(alfa_ABS * pi / 180)

    # TRIPPED
    else:
        # begin
        if alfa_ABS <= 5:
            delta_star_S = 10**((0.0679 * alfa_ABS)) * delta_star_0
        elif (alfa_ABS <= 12.5):
            delta_star_S = 0.381 * 10**((0.1516 * alfa_ABS)) * delta_star_0
        elif (alfa_ABS <= 25):
            delta_star_S = 14.296 * 10**((0.0258 * alfa_ABS)) * delta_star_0
        # above AoA of 25 degree calculate the shadow area thickness of the airfoil and use as boundary layer thickness
        else:
            delta_star_S = c * sin(alfa_ABS * pi / 180)
    return delta_star_S, delta_star_P