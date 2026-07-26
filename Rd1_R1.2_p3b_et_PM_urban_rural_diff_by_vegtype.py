import os

import numpy as np
import pandas as pd
import scipy.stats as st


ET_COLUMNS = [f'et_m{i}' for i in range(1, 13)]


def pixel_mean_et(df):
    return df[ET_COLUMNS].astype(float).mean(axis=1, skipna=True)


def summarize_group(df, label):
    core = df[df['type'] == 'core']
    bg = df[df['type'] == 'bg']

    core_et = pixel_mean_et(core)
    bg_et = pixel_mean_et(bg)
    core_n = int(np.sum(~np.isnan(core_et)))
    bg_n = int(np.sum(~np.isnan(bg_et)))
    core_mean = np.nanmedian(core_et) if core_n > 0 else np.nan
    bg_mean = np.nanmedian(bg_et) if bg_n > 0 else np.nan

    return {
        f'{label}_core_et_mean': core_mean,
        f'{label}_bg_et_mean': bg_mean,
        f'{label}_urban_rural_et_diff': core_mean - bg_mean,
        f'{label}_core_n': core_n,
        f'{label}_bg_n': bg_n,
    }


def summarize_city(city_df):
    tree_dom = city_df['tree_frac'].astype(float) - city_df['grass_frac'].astype(float)>0.1
    grass_dom = city_df['tree_frac'].astype(float) - city_df['grass_frac'].astype(float)<-0.1

    result = {'id': city_df['id'].iloc[0]}
    result.update(summarize_group(city_df, 'all'))
    result.update(summarize_group(city_df[tree_dom], 'tree_dominant'))
    result.update(summarize_group(city_df[grass_dom], 'grass_dominant'))
    return result


def load_tac_diff(tac_path):
    tac_data = np.load(tac_path)
    tac = tac_data['array1']
    city_id = tac_data['array2']
    tac_df = pd.DataFrame(tac, columns=['tac_urban_core', 'tac_urban_edge', 'tac_rural_bgr'])
    tac_df['id'] = city_id
    tac_df['urban_rural_tac_diff'] = tac_df['tac_urban_core'] - tac_df['tac_rural_bgr']
    return tac_df[['id', 'urban_rural_tac_diff']]


def add_fit_line(ax, x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 3:
        return np.nan, np.nan

    slope, intercept, r, p, _ = st.linregress(x[mask], y[mask])
    x_fit = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
    ax.plot(x_fit, slope * x_fit + intercept, color='k', lw=1.0)
    return r, p


def plot_et_tac_diff_scatter(df, fig_path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [
        ('all_urban_rural_et_diff', 'All'),
        ('tree_dominant_urban_rural_et_diff', 'Tree-dominant'),
        ('grass_dominant_urban_rural_et_diff', 'Grass-dominant'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharex=True)
    for ax, (et_col, title) in zip(axes, panels):
        x = df['urban_rural_tac_diff'].astype(float).to_numpy()
        y = df[et_col].astype(float).to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y)

        ax.scatter(x[mask], y[mask], s=18, color='#2166ac', alpha=0.65, edgecolors='none')
        ax.axhline(0, color='0.65', lw=0.8, ls='--')
        ax.axvline(0, color='0.65', lw=0.8, ls='--')
        r, p = add_fit_line(ax, x, y)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Urban-rural TAC diff', fontsize=10)
        ax.tick_params(width=0.8, labelsize=9)
        ax.text(
            0.05, 0.95,
            f'n = {np.sum(mask)}\nr = {r:.2f}',
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=9
        )

    axes[0].set_ylabel('Urban-rural ET diff', fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=900)
    plt.close(fig)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    input_path = current_dir + '/2_Output/pure_veg/monthly_landsat_et_PM_global_100.csv'
    tac_path = current_dir + '/2_Output/tac_nadir_city_3zones_all.npz'
    output_path = current_dir + '/2_Output/pure_veg/monthly_landsat_et_PM_global_urban_rural_diff_by_vegtype_100.csv'
    fig_path = current_dir + '/4_Figures/Rd1_R1_2_ET_diff_vs_TAC_diff_by_vegtype'

    df = pd.read_csv(input_path)
    missing_columns = [col for col in ['id', 'type', 'fvc', 'tree_frac', 'grass_frac', *ET_COLUMNS] if col not in df.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')
    df = df[df['fvc'].astype(float) >= 0.98].copy()

    summaries = []
    for city_id, city_df in df.groupby('id', sort=True):
        summaries.append(summarize_city(city_df))

    out_df = pd.DataFrame(summaries)
    tac_df = load_tac_diff(tac_path)
    out_df = out_df.merge(tac_df, on='id', how='left')
    out_df.to_csv(output_path, index=False)
    print('Saved:', output_path)
    plot_et_tac_diff_scatter(out_df, fig_path)
    print('Saved:', fig_path)
