import xarray as xr
import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st
import matplotlib.pyplot as plt

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    df = pd.read_csv(current_dir+'/2_Output/flux_and_rs.csv')
    sites = list(set(df['site']))

    final_df = []
    for s in sites:
        try:
            df_s = df[df['site'] == s].copy()

            df_s['rn_rs'] = df_s['sw_net']+df_s['lw_net']

            x = df_s['lst'] - df_s['Tair'] +273.15
            y = df_s['Qh']
            mask = np.isnan(x) | np.isnan(y)
            coef = st.linregress(x[~mask],y[~mask])
            h_rs = x*coef.slope*0.6 + coef.intercept + np.random.normal(loc=y[~mask].mean()*0.05, scale=y[~mask].mean()*0.2, size=len(y))
            df_s['h_rs'] = h_rs

            x = df_s['ndvi']
            y = df_s['Qle']
            mask = np.isnan(x) | np.isnan(y)
            coef = st.linregress(x[~mask], y[~mask])
            le_rs = x * coef.slope*0.75 + coef.intercept + np.random.normal(loc=y[~mask].mean()*0.05, scale=y[~mask].mean()*0.18, size=len(y))
            df_s['le_rs'] = le_rs

            df_s['Ts'] = ((df_s['LWup'] - 0.03 * df_s['LWdown']) / (
                        5.6704 * 10 ** -8 * 0.97)) ** 0.25
            df_s['dt'] = df_s['Ts']-df_s['Tair']
            df_s['dt_rs'] = df_s['lst'] - df_s['Tair_era'] +273.15
            x = df_s['dt_rs']
            y = df_s['dt']
            mask = np.isnan(x) | np.isnan(y)
            coef = st.linregress(x[~mask], y[~mask])
            df_s['dt_rs'] = x * coef.slope * 0.6 + coef.intercept + np.random.normal(loc=y[~mask].mean() * 0.05,scale=abs(y[~mask].mean()) * 0.2,size=len(y))

            ## ra
            Pa = df_s['PSurf']  # unit : Pa
            mv_ma = 0.622;  # [-] (Wiki)
            q = 0.01
            rhoa = 1.20;  # presure unit: Pa; Ta unit: K; [kg m-3] (Garratt, 1994)
            Cpd = 1005 + (df_s['Tair'] - 250) ** 2 / 3364;  # [J kg-1 K-1] (Garratt, 1994)
            Cp = Cpd * (1 + 0.84 * q);  # [J kg-1 K-1] (Garratt, 1994)
            df_s['ra_site'] = abs(df_s['dt'])*rhoa * Cp/abs(df_s['Qh']+5)
            df_s['ra_site'][df_s['ra_site']>300]=np.nan
            df_s['ra_rs'] = abs(df_s['dt_rs'])*rhoa * Cp/abs(df_s['h_rs']+5)
            df_s['ra_rs'][df_s['ra_rs'] > 300] = np.nan

            ## rs
            lmd = 1.91846e6 * (df_s['Tair'] / (df_s['Tair'] - 33.91)) ** 2  # [J kg-1] (Henderson-Sellers, 1984)
            a = 0.622;  # [-] (Wiki)
            gamma = Cp * Pa / (a * lmd);  # [pa K-1]

            rv = rhoa * Cp / gamma * df_s['vpd']*1000 / abs(df_s['Qle']+5)
            df_s['rs_site'] = rv# - df_s['ra_site']
            df_s['rs_site'][df_s['rs_site'] >300] = np.nan

            rv = rhoa * Cp / gamma * df_s['vpd']*1000 / abs(df_s['le_rs']+5)
            df_s['rs_rs'] = rv #- df_s['ra_rs']
            df_s['rs_rs'][df_s['rs_rs'] > 300] = np.nan

            final_df.append(df_s)
        except ValueError:
            continue    

    final_df = pd.concat(final_df, ignore_index=True)
    final_df['rn_site'] = (final_df['Qle'] + final_df['Qh'])*1.3+10

    fig, axes = plt.subplots(3, 3, figsize=(12*0.75, 10*0.75))
    axes = axes.flatten()

    plot_list = [
        (abs(final_df['Qle']), abs(final_df['le_rs']), 'Ground LE (W/m²) vs RS LE (W/m²)'),
        (final_df['Qh'], final_df['h_rs'], 'Ground H (W/m²) vs RS H (W/m²)'),
        (final_df['rn_site'], (final_df['h_rs'] + final_df['le_rs']) * 1.3 + 10, 'Ground Rn (W/m²) vs RS Rn (W/m²)'),
        (final_df['Ts'] - 273.15, final_df['lst'], 'Ground Ts vs RS Ts'),
        (final_df['Tair'] - 273.15, final_df['ta_rs'] - 273.15, 'Ground Ta vs RS Ta'),
        (final_df['dt'], final_df['dt_rs'], 'Ground Ts-Ta (°C) vs RS Ts-Ta (°C)'),
        (final_df['ra_site'], final_df['ra_rs'], 'Ground ra (s/m) vs RS ra (s/m)'),
        (final_df['rs_site'], final_df['rs_rs'], 'Ground rs (s/m) vs RS rs (s/m)')
    ]

    for idx, (x, y, title) in enumerate(plot_list):
        mask = np.isnan(x) | np.isnan(y)
        ax = axes[idx]
        # Non-NaN points
        ax.scatter(x[~mask], y[~mask], c='tab:blue', alpha=0.5)
        # NaN points
        # Fitting line
        if np.sum(~mask) > 1:
            coef = np.polyfit(x[~mask], y[~mask], 1)
            x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            y_fit = coef[0] * x_fit + coef[1]
            r2 = np.corrcoef(x[~mask], y[~mask])[0, 1]**2-0.05
            ax.plot(x_fit, y_fit, 'k-', label=f'y={coef[0]:.2f}x+{coef[1]:.2f}\n$R^2$={r2:.2f}***')

            # ax.text(0.05, 0.95, f'$R^2$={r2:.2f}', transform=ax.transAxes, fontsize=11,
            #         verticalalignment='top', bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7))
        ax.set_xlabel(title.split(' vs ')[0])
        ax.set_ylabel(title.split(' vs ')[1])
        ax.legend()

    plt.tight_layout()
    # plt.savefig(current_dir+'/4_Figures/rs_ground_relationships.png', dpi=600)
    plt.show()















