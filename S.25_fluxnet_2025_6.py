import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import os
import seaborn as sns
import numpy as np
import ee
import scipy.stats as st
import math
def calculate_monthly_et(t_mean, t_max, t_min, Rs, Rnl,td_mean, u2, altitude):
    """
    使用FAO Penman-Monteith方程计算月参考蒸散量(ET0)

    参数:
    t_mean -- 月平均气温(摄氏度)
    t_max -- 月最高气温(摄氏度)
    t_min -- 月最低气温(摄氏度)
    rh_mean -- 月平均相对湿度(%)
    u2 -- 2米高处月平均风速(m/s)
    altitude -- 站点海拔高度(米)

    返回:
    et0_monthly -- 月参考蒸散量(mm/month)
    """
    # 1. 计算饱和水汽压(kPa)
    e0_tmax = 0.6108 * np.exp((17.27 * t_max) / (t_max + 237.3))
    e0_tmin = 0.6108 * np.exp((17.27 * t_min) / (t_min + 237.3))
    es = (e0_tmax + e0_tmin) / 2

    # 2. 计算实际水汽压(kPa)
    ea = 0.6108 * np.exp((17.27 * td_mean) / (td_mean + 237.3))

    # 3. 计算饱和水汽压曲线斜率(kPa/°C)
    delta = 4098 * (0.6108 * np.exp((17.27 * t_mean) / (t_mean + 237.3))) / ((t_mean + 237.3) ** 2)

    # 4. 计算大气压(kPa)
    P = 101.3 * ((293 - 0.0065 * altitude) / 293) ** 5.26

    # 5. 计算心理常数(kPa/°C)
    gamma = 0.000665 * P

    # 7. 计算净辐射(MJ/m²/day)
    Rns = 0.85 * Rs  # 净短波辐射
    Rn = Rns + Rnl
    # 8. 计算土壤热通量(G) - 月尺度通常忽略
    G = 0

    # 9. 计算Penman-Monteith ET0 (mm/day)
    et0_daily = (0.408 * delta * (Rn - G) + gamma * (900 / (t_mean + 273)) * u2 * (es - ea)) / (
                delta + gamma * (1 + 0.34 * u2))

    # 10. 转换为月尺度(mm/month)
    et0_monthly = et0_daily * 30
    et0_monthly = et0_monthly*0.94 # 转换为w/m2
    return et0_monthly

def cal_ETa_PM(Rn, Ts, Ta):

    Rn = Rn#*(1-0.23)
    ra = 200  # s/m
    Pa = 101 * 1000  # Pa
    rhoa = Pa / (287.05 * (Ta + 273.15))  # presure unit: Pa; Ta unit: K; [kg m-3] (Garratt, 1994)
    es = 2.1718e10 * np.exp(-4157. / (Ta + 273.15 - 33.91));
    # specific heat of dry air
    Cp = 1005 + ((Ta + 273.15) - 250) ** 2 / 3364  # [J kg-1 K-1] (Garratt, 1994)

    H = rhoa * Cp * (Ts - Ta) / ra
    # H[H<0]=0
    G = Rn*0.15
    LE = Rn - H - G
    # LE[LE<0]=np.nan
    lmt = 24*3600*30/(2.45*10**6)
    ET = LE/lmt
    # ensure numpy array
    ET = np.array(ET, dtype=float)
    return ET/0.72

def cal_ETa(Rn, Ts, Ta, ETref):

    ra = 200  # s/m
    Pa = 101 * 1000  # Pa
    rhoa = Pa / (287.05 * (Ta + 273.15))  # presure unit: Pa; Ta unit: K; [kg m-3] (Garratt, 1994)

    # specific heat of dry air
    Cp = 1005 + ((Ta + 273.15) - 250) ** 2 / 3364  # [J kg-1 K-1] (Garratt, 1994)
    # specific heat of air
    psy = rhoa * Cp / (ra * Rn)
    c = 1.
    ET_frc = 1-psy*(Ts-c*Ta)
    # ET_frc[ET_frc>1]=1
    # ET_frc[ET_frc < 0] = 0
    ETa = ETref*ET_frc # calculated ET; openET is reference
    # ETa[ETa>250]=np.nan
    return ETa

if __name__ == '__main__':
    final_df = pd.read_csv(os.path.join('..',  '2_Output', 'flux_and_rs.csv'))

    sites = list(set(final_df['site']))
    plt.figure()

    df = []
    output = []
    ET0s = []
    LEs = []
    for site in sites:
        if site == 'US-WestPhoenix': continue
        df_site = final_df[final_df['site']==site]
        Rs0 = df_site['sw_down'].values # w/m2
        Rs = Rs0*84600/10**6 # MJ/m2/day
        Ts = df_site['lstD'].values*0.85

        t_mean = df_site['ta_rs'].values - 273.15
        t_max = df_site['ta_max_rs'].values - 273.15
        t_min = df_site['ta_rs_min'].values - 273.15
        td_mean = df_site['td_rs'].values - 273.15
        Rnl0 = df_site['lw_net'].values
        Rnl = Rnl0 * 84600 / 10 ** 6
        u2 = 2.0
        altitude = 100

        ET0 = calculate_monthly_et(t_mean, t_max, t_min, Rs, Rnl, td_mean, u2, altitude)
        LE = df_site['LE'].values

        mask = np.isnan(ET0) | np.isnan(LE)
        coef = st.linregress(LE[~mask],ET0[~mask]).rvalue
        slope = st.linregress(LE[~mask], ET0[~mask]).slope
        dt = np.nanpercentile(Ts+ 273.15 - df_site['ta_rs'],50)
        ndvi = df_site['ndvi'].mean()
        print(coef,slope,site)

        if (coef>0.5) and (slope>0):
            ETa = cal_ETa(Rs0 * 0.77 + Rnl0, Ts, t_mean, ET0)  # cal_ETa_PM(Rs, Ts, t_mean)
            output.append([coef, slope, dt, ndvi])

            df.append(df_site)
            plt.plot(ET0*dt/6, df_site['LE'].values, 'o')
            ET0s = ET0s+list(ET0)
            LEs = LEs + list(df_site['LE'].values)

    output = np.array(output)
    ET0s = np.array(ET0s)
    mask = np.isnan(ET0s) | np.isnan(LEs)
    st.linregress(np.array(LEs)[~mask],np.array(ET0s)[~mask])
    # plt.figure(); plt.plot(output[:,-2],output[:,1],'o')






