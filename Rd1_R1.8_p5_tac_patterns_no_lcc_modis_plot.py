import os

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
plt.close()

AXIS_LABEL_SIZE = 14+1
TICK_LABEL_SIZE = 12+1
PANEL_LABEL_SIZE = 14+1
ANNOTATION_SIZE = 12+1


def load_tac_3zone_df(path):
    data = np.load(path)
    tac = data['array1']
    city_id = data['array2']
    df = pd.DataFrame(tac, columns=['urban_core', 'urban_edge', 'rural_bgr'])
    df['ID'] = city_id
    df['urban_rural_tac_diff'] = df['urban_core'] - df['rural_bgr']
    return df


def panel_lcc_histogram(ax, csv_path):
    df = pd.read_csv(csv_path)
    values = df.loc[df['status'].eq('ok'), 'landcover_change_fraction'].dropna().to_numpy()

    ax.hist(values, bins=np.linspace(0, 0.45, 19), color='0.70', edgecolor='0.25', linewidth=0.6)
    ax.axvline(np.nanmean(values), color='#b2182b', lw=1.2)
    ax.set_xlabel('LCC fraction', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('City count', fontsize=AXIS_LABEL_SIZE)
    ax.text(
        0.96, 0.94,
        f'mean = {np.nanmean(values):.2f}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=ANNOTATION_SIZE
    )


def panel_tac_bar(ax, tac_path):
    tac = np.load(tac_path)['array1']
    tac_mean = np.nanmean(tac, axis=0)
    tac_err = np.nanstd(tac, axis=0) * 0.25

    bars = ax.bar(
        np.arange(3),
        tac_mean,
        yerr=tac_err,
        width=0.36,
        color=['#2166ac', '#67a9cf', '#b2182b'],
        error_kw={'lw': 0.8, 'capsize': 2, 'capthick': 0.8}
    )

    ax.set_xticks([0, 1, 2], ['UC', 'UE', 'RA'])
    ax.set_ylabel('No-LCC TAC', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim([0.10, 0.22])


def panel_urban_rural_regression(ax, tac_path):
    tac = np.load(tac_path)['array1']
    mask = ~np.isnan(tac[:, 0]) & ~np.isnan(tac[:, 2])
    x = tac[mask, 0]
    y = tac[mask, 2]

    above = y > x
    below = ~above
    ax.plot(x[above], y[above], 'o', mfc='none', color='C0', ms=4, mew=0.8)
    ax.plot(x[below], y[below], 'o', mfc='none', color='C3', ms=4, mew=0.8)

    axis_min = min(np.nanmin(x), np.nanmin(y))
    axis_max = max(np.nanmax(x), np.nanmax(y))
    padding = (axis_max - axis_min) * 0.08
    axis_min -= padding
    axis_max += padding

    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7, lw=0.9)
    fit = np.poly1d(np.polyfit(x, y, 1))
    x_fit = np.linspace(axis_min, axis_max, 100)
    ax.plot(x_fit, fit(x_fit), 'k-', alpha=0.7, lw=0.9)

    reg = st.linregress(x, y)
    ax.text(
        0.05, 0.95,
        f'r = {reg.rvalue:.2f}',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=ANNOTATION_SIZE
    )
    ax.set_xlabel('No-LCC TAC$_{UC}$', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('No-LCC TAC$_{RA}$', fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal', adjustable='box')


def panel_tac_diff_scatter(ax, original_path, no_lcc_path):
    df_original = load_tac_3zone_df(original_path)
    df_no_lcc = load_tac_3zone_df(no_lcc_path)
    df_compare = df_original[['ID', 'urban_rural_tac_diff']].merge(
        df_no_lcc[['ID', 'urban_rural_tac_diff']],
        on='ID',
        suffixes=('_original', '_no_lcc')
    )
    df_compare = df_compare.dropna(subset=['urban_rural_tac_diff_original', 'urban_rural_tac_diff_no_lcc'])

    x = df_compare['urban_rural_tac_diff_original'].to_numpy()
    y = df_compare['urban_rural_tac_diff_no_lcc'].to_numpy()
    r, _ = st.pearsonr(x, y)
    rmse = np.sqrt(np.nanmean((y - x) ** 2))

    ax.scatter(x, y, s=14, color='#2166ac', alpha=0.65, edgecolors='none')

    axis_min = np.nanmin([np.nanmin(x), np.nanmin(y)])
    axis_max = np.nanmax([np.nanmax(x), np.nanmax(y)])
    padding = (axis_max - axis_min) * 0.08
    axis_min -= padding
    axis_max += padding
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color='0.25', lw=0.9, ls='--')

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Original ΔTAC', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('No-LCC ΔTAC', fontsize=AXIS_LABEL_SIZE)
    ax.text(
        0.05, 0.95,
        f'r = {r:.2f}\nRMSE = {rmse:.3f}',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=ANNOTATION_SIZE
    )


def format_panel(ax, label):
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, ha='left', va='top',
            fontsize=PANEL_LABEL_SIZE, fontweight='bold')
    ax.tick_params(width=0.8, labelsize=TICK_LABEL_SIZE)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    no_lcc_tac_path = current_dir + '/2_Output/tac_nadir_city_3zones_no_lcc_modis.npz'
    original_tac_path = current_dir + '/2_Output/tac_nadir_city_3zones_all.npz'
    lcc_fraction_path = current_dir + '/2_Output/landcover_change_fraction_by_city.csv'
    fig_to_path = current_dir + '/4_Figures/Rd1_R1_8_tac_lcc_four_panel_MODIS'

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4))
    panel_lcc_histogram(axes[0], lcc_fraction_path)
    panel_tac_bar(axes[1], no_lcc_tac_path)
    panel_urban_rural_regression(axes[2], no_lcc_tac_path)
    panel_tac_diff_scatter(axes[3], original_tac_path, no_lcc_tac_path)

    for ax, label in zip(axes, ['', '', '', '']):
        format_panel(ax, label)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(fig_to_path, dpi=900)
