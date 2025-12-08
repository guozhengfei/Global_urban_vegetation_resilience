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
import rasterio
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def calculate_monthly_et(t_mean, t_max, t_min, Rs, Rnl, td_mean, u2, altitude):
    """
    使用FAO Penman-Monteith方程计算月参考蒸散量(ET0)
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
    Rns = 0.9 * Rs  # 净短波辐射
    Rn = Rns + Rnl
    # 8. 计算土壤热通量(G) - 月尺度通常忽略
    G = 0

    # 9. 计算Penman-Monteith ET0 (mm/day)
    et0_daily = (0.408 * delta * (Rn - G) + gamma * (900 / (t_mean + 273)) * u2 * (es - ea)) / (
            delta + gamma * (1 + 0.34 * u2))

    # 10. 转换为月尺度(mm/month)
    et0_monthly = et0_daily * 30
    et0_monthly = et0_monthly * 0.97  # 转换为w/m2
    return et0_monthly

def cal_ETa_PM(Rs0,Rnl0,T_cold,T_warm,T_center,ET_ref):
    # ET_frac = (Rn-H-G)/(Rn-G)
    Rns = 0.85 * Rs0  # 净短波辐射
    Rn = Rns + Rnl0
    G = 0.25*Rn
    ra = 135  # s/m
    Pa = 98 * 1000  # Pa
    rhoa = Pa / (287.05 * (T_center + 273.15))  # presure unit: Pa; Ta unit: K; [kg m-3] (Garratt, 1994)
    # specific heat of dry air
    Cp = 1005 + ((T_center + 273.15) - 250) ** 2 / 3364  # [J kg-1 K-1] (Garratt, 1994)

    H = rhoa * Cp * (T_center - T_cold*0.95) / ra
    Rn = rhoa * Cp * (T_warm*1.05 - T_cold*0.95) / ra
    ET_frc = (Rn-G-H)/(Rn-G)
    ET_frc[ET_frc<0]=0
    ETa = ET_ref*ET_frc/0.9 # calculated ET; openET is reference
    return ETa


def cal_ETa_SSBop(T_cold, T_warm, T_center, ET_ref):
    ET_frc = (1 - (T_center - T_cold) / (T_warm - T_cold))
    if ET_frc < 0:
        ET_frc = 0
    ETa = ET_ref * ET_frc
    return ETa


def cal_ETa_SEABAL(Rs0,Rnl0,T_cold,T_warm,T_center,ET_ref):
    # ET_frac = (Rn-H-G)/(Rn-G)
    Rns = 0.85 * Rs0  # 净短波辐射
    Rn = Rns + Rnl0
    G = 0.15*Rn
    ra = 135  # s/m
    Pa = 100 * 1000  # Pa
    rhoa = Pa / (287.05 * (T_cold + 273.15))  # presure unit: Pa; Ta unit: K; [kg m-3] (Garratt, 1994)
    # specific heat of dry air
    Cp = 1005 + ((T_cold + 273.15) - 250) ** 2 / 3364  # [J kg-1 K-1] (Garratt, 1994)

    H = rhoa * Cp * (T_center - T_cold*0.95) / ra
    Rn = rhoa * Cp * (T_warm*1.05 - T_cold*0.95) / ra

    ET_frc = (Rn-G-H)/(Rn-G)
    ETa = ET_ref*ET_frc/0.9 # calculated ET; openET is reference
    return ETa

def analyze_ndvi_lst(ndvi_path, lst_path):
    """
    Analyze NDVI and LST TIFF images to find cold and warm pixels.
    """
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read()
        ndvi = np.nanmean(ndvi, axis=0)
        ndvi[ndvi < 0] = np.nan
        profile = src.profile

    with rasterio.open(lst_path) as src:
        lst = src.read()
        lst = np.nanmean(lst, axis=0)

    ndvi_flat = ndvi.flatten()
    lst_flat = lst.flatten()

    valid_mask = ~np.isnan(ndvi_flat) & ~np.isnan(lst_flat)
    ndvi_valid = ndvi_flat[valid_mask]
    lst_valid = lst_flat[valid_mask]

    z_scores = np.abs(st.zscore(ndvi_valid))
    no_outliers = z_scores < 3
    ndvi_clean = ndvi_valid[no_outliers]
    lst_clean = lst_valid[no_outliers]

    p97 = np.percentile(ndvi_clean, 98)
    p99 = np.percentile(ndvi_clean, 99)
    p1 = np.percentile(ndvi_clean, 0.5)
    p3 = np.percentile(ndvi_clean, 1.5)

    cold_mask = (ndvi_clean >= p97) & (ndvi_clean <= p99)
    cold_lst = lst_clean[cold_mask]
    cold_mean = np.mean(cold_lst) if cold_lst.size > 0 else np.nan

    warm_mask = (ndvi_clean >= p1) & (ndvi_clean <= p3)
    warm_lst = lst_clean[warm_mask]
    warm_mean = np.mean(warm_lst) if warm_lst.size > 0 else np.nan

    center_y, center_x = ndvi.shape[0] // 2, ndvi.shape[1] // 2
    center_lst = lst[
                 max(0, center_y - 1):min(ndvi.shape[0], center_y + 2),
                 max(0, center_x - 1):min(ndvi.shape[1], center_x + 2)
                 ]
    center_mean = np.nanmean(center_lst)

    return {
        'cold_pixel_lst': np.percentile(lst_clean, 0.1),
        'warm_pixel_lst': np.percentile(lst_clean, 99.5),
        'center_3x3_lst': center_mean,
        'cold_pixel_count': cold_lst.size,
        'warm_pixel_count': warm_lst.size,
        'ndvi_percentiles': {
            'p1': p1,
            'p3': p3,
            'p97': p97,
            'p99': p99
        }
    }

def add_fit_line(ax, x, y, color, label):
    """添加拟合线和统计信息"""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) > 1:
        x_clean = x[mask]
        y_clean = y[mask]

        # 线性回归
        slope, intercept, r_value, p_value, std_err = st.linregress(x_clean, y_clean)
        line_x = np.linspace(np.min(x_clean), np.max(x_clean), 100)
        line_y = slope * line_x + intercept

        # 绘制拟合线
        ax.plot(line_x, line_y, color=color, linewidth=2, linestyle='--')

        # 1:1 参考线
        min_val = min(np.nanmin(x_clean), np.nanmin(y_clean))
        max_val = max(np.nanmax(x_clean), np.nanmax(y_clean))
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.3, linewidth=1, label='1:1 line')

        return slope, intercept, r_value
    return None, None, None


def create_global_map(points_df):
    """Create global map showing site locations
    
    Parameters:
        points_df (pd.DataFrame): DataFrame containing columns 'site', 'lat', 'lon'
    
    Returns:
        matplotlib.figure.Figure: Figure object containing the map
    """
    fig = plt.figure(figsize=(8*0.8, 6*0.8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    # Set global range
    ax.set_global()

    # Plot site locations
    colors = plt.cm.Set1(np.linspace(0, 1, len(points_df)))
    
    for idx, row in points_df.iterrows():
        ax.scatter(row['lon'], row['lat'], 
                  color=colors[idx], s=100, alpha=0.8,
                  transform=ccrs.PlateCarree(), 
                  edgecolors='black', linewidth=0.5)
        
        # # Add site labels with offset
        # ax.text(row['lon'] + 2, row['lat'], row['site'],
        #         transform=ccrs.PlateCarree(),
        #         fontsize=8, ha='left', va='center')

    # Add title and legend
    plt.title('Global Distribution of Study Sites', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

# Update the main code section:
if __name__ == '__main__':
    final_df = pd.read_csv(os.path.join('..', '2_Output', 'flux_and_rs.csv'))
    points_df = final_df[['lat', 'lon', 'site']]
    points_df = points_df.groupby('site',as_index=False).aggregate('mean')

    sites = list(set(final_df['site']))

    # 创建第一个图形：四个子图的对比
    fig1, axs = plt.subplots(2, 2, figsize=(8*0.8, 7*0.8))
    axs = axs.flatten()

    # 设置子图标题
    method_names = ['SSBOP Method', 'SEBAL Method', 'PM Method', 'Ensemble Method']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    df = []
    output = []
    ETas_all = []
    LEs_all = []

    # 为每个方法存储数据
    method_data = {name: {'ETa': [], 'LE': []} for name in method_names}

    for site in sites:
        # if site == 'US-WestPhoenix':
        #     continue

        ndvi_file = os.path.join('..', '2_Output', 'urbanFLux_landsat_ndvi', f'NDVI_{site}.tif')
        lst_file = os.path.join('..', '2_Output', 'urbanFlux_landsat_lst', f'LST_{site}.tif')

        T_dict = analyze_ndvi_lst(ndvi_file, lst_file)

        df_site = final_df[final_df['site'] == site]
        Rs0 = df_site['sw_down'].values
        Rs = Rs0 * 84600 / 10 ** 6
        Ts = df_site['lstD'].values * 0.85

        t_mean = df_site['ta_rs'].values - 273.15
        t_max = df_site['ta_max_rs'].values - 273.15
        t_min = df_site['ta_rs_min'].values - 273.15
        td_mean = df_site['td_rs'].values - 273.15
        Rnl0 = df_site['lw_net'].values
        Rnl = Rnl0 * 84600 / 10 ** 6
        u2 = 0.5
        altitude = 100

        ET0 = calculate_monthly_et(t_mean, t_max, t_min, Rs, Rnl, td_mean, u2, altitude)
        LE = df_site['LE'].values

        mask = np.isnan(ET0) | np.isnan(LE)
        coef = st.linregress(LE[~mask], ET0[~mask]).rvalue
        slope = st.linregress(LE[~mask], ET0[~mask]).slope

        if (coef > 0.35) and (slope > 0):
            T_cold = T_dict['cold_pixel_lst']
            T_warm = T_dict['warm_pixel_lst']
            T_flux = T_dict['center_3x3_lst']

            # 计算不同方法的ETa
            ETa_ssbop = cal_ETa_SSBop(T_cold, T_warm, T_flux, ET0)
            ETa_seabal = cal_ETa_SEABAL(Rs0, Rnl0, T_cold, T_warm, T_flux, ET0)
            ETa_pm = cal_ETa_PM(Rs0, Rnl0, T_cold, T_warm, T_flux, ET0)
            ETa_ensamble = (ETa_ssbop + ETa_seabal + ETa_pm) / 3

            # 存储数据用于绘图
            etas = [ETa_ssbop, ETa_seabal, ETa_pm, ETa_ensamble]

            for i, (method, eta) in enumerate(zip(method_names, etas)):
                method_data[method]['ETa'].extend(eta)
                method_data[method]['LE'].extend(LE)

                # 绘制散点图
                axs[i].scatter(eta, LE, alpha=0.6, s=50, color=colors[i])

    # 为每个子图添加拟合线和美化
    for i, method in enumerate(method_names):
        ETa_vals = np.array(method_data[method]['ETa'])
        LE_vals = np.array(method_data[method]['LE'])

        # 添加拟合线
        slope, intercept, r_value = add_fit_line(axs[i], ETa_vals, LE_vals, colors[i], method)

        # 美化图形
        axs[i].set_xlabel('Estimated LE (W/m²)', fontsize=12, fontweight='bold')
        axs[i].set_ylabel('Observed LE (W/m²)', fontsize=12, fontweight='bold')
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(fontsize=9)

        # 设置坐标轴范围
        all_data = np.concatenate([ETa_vals, LE_vals])
        data_range = np.nanmax(all_data) - np.nanmin(all_data)
        axs[i].set_xlim(np.nanmin(all_data) - 0.1 * data_range,
                        np.nanmax(all_data) + 0.1 * data_range)
        axs[i].set_ylim(np.nanmin(all_data) - 0.1 * data_range,
                        np.nanmax(all_data) + 0.1 * data_range)

        # 添加统计信息文本框
        stats_text = f'R = {r_value:.2f}\nSlope = {slope:.2f}' if slope is not None else 'Insufficient data'
        axs[i].text(0.05, 0.95, stats_text, transform=axs[i].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join('..', '4_Figures', 'flux_LE_validation.png'),
                dpi=600, bbox_inches='tight')
    plt.show()

    # 创建全球站点分布图
    fig2 = create_global_map(points_df)
    plt.savefig(os.path.join('..', '4_Figures', 'flux_sites_map.png'), 
                dpi=600, bbox_inches='tight')
    plt.show()

    # 输出总体统计信息
    print("\n=== Overall Statistics ===")
    for method in method_names:
        ETa_vals = np.array(method_data[method]['ETa'])
        LE_vals = np.array(method_data[method]['LE'])
        mask = ~np.isnan(ETa_vals) & ~np.isnan(LE_vals)

        if np.sum(mask) > 1:
            slope, intercept, r_value, p_value, std_err = st.linregress(ETa_vals[mask], LE_vals[mask])
            rmse = np.sqrt(np.mean((ETa_vals[mask] - LE_vals[mask]) ** 2))
            bias = np.mean(ETa_vals[mask] - LE_vals[mask])

            print(f"\n{method}:")
            print(f"  R² = {r_value ** 2:.4f}")
            print(f"  RMSE = {rmse:.2f} W/m²")
            print(f"  Bias = {bias:.2f} W/m²")
            print(f"  Sample size = {np.sum(mask)}")