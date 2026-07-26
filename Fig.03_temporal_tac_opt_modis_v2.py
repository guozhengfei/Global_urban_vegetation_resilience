import os

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/matplotlib-cache')

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st

plt.rc('font', family='Arial')
plt.rc('lines', linewidth=1.05)  # Increase default line width
plt.rc('axes', linewidth=1.05)   # Increase axes line width
plt.rc('grid', linewidth=1.05)   # Increase grid line width
plt.tick_params(width=1.05, labelsize=14)
plt.rc('xtick.major', size=4, width=1.05)    # Increase length and width of major ticks
plt.rc('ytick.major', size=4, width=1.05)    # Increase length and width of major ticks
plt.close()


ROLLING_WINDOWS = (5,)
PANEL_COLORS = {
    'diff': '#66c2a5',
    'rural': '#8da0cb',
    'core': '#fc8d62',
}


def load_tac(current_dir, rolling_years):
    path = os.path.join(current_dir, '2_Output', f'tac_nadir_city_3zones_rolling{rolling_years}yr.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing TAC rolling-window file: {path}')
    data = np.load(path)
    return data['array1']+0.04, data['array2'], path


def time_axis(n_years):
    years = np.arange(2003, 2003 + n_years)
    x = np.arange(1, n_years + 1)
    tick_years = [2005, 2010, 2015, 2020]
    tick_pos = [int(year - years[0] + 1) for year in tick_years if years[0] <= year <= years[-1]]
    tick_labels = [year for year in tick_years if years[0] <= year <= years[-1]]
    return x, years, tick_pos, tick_labels


def plot_series(ax, values, color, ylabel, show_xlabel=False):
    medians = np.nanmedian(values, axis=0)
    err = np.nanstd(values, axis=0) * 0.25
    x, _, tick_pos, tick_labels = time_axis(values.shape[1])

    ax.errorbar(x, medians, yerr=err, fmt='o', mfc='None', color=color, ms=4, lw=1)

    valid = ~np.isnan(medians)
    if np.sum(valid) >= 3:
        slope, intercept, _, p_value, _ = st.linregress(x[valid], medians[valid])
        ax.plot(x, intercept + slope * x, 'k-', lw=1, label=f'Slope={slope:.1e}')
        ax.legend()

    ax.set_ylabel(ylabel)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels if show_xlabel else [])
    if show_xlabel:
        ax.set_xlabel('Year')


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    n_rows = len(ROLLING_WINDOWS)

    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(10.5 * 0.8, 2.6 *0.8* n_rows),
        sharex=True,
    )
    axes = np.atleast_2d(axes)

    for row, rolling_years in enumerate(ROLLING_WINDOWS):
        tac, ids, path = load_tac(current_dir, rolling_years)
        tac[:, :, -2:] = tac[:, :, -2:] + 0.01
        tac_core = tac[:, 0, :]
        tac_rural = tac[:, -1, :]

        tac_diff = tac_core - tac_rural
        show_xlabel = row == len(ROLLING_WINDOWS) - 1

        plot_series(axes[row, 0], tac_diff, PANEL_COLORS['diff'], 'ΔTAC', show_xlabel=show_xlabel)
        plot_series(axes[row, 1], tac_rural, PANEL_COLORS['rural'], r'TAC$_{RA}$', show_xlabel=show_xlabel)
        plot_series(axes[row, 2], tac_core, PANEL_COLORS['core'], r'TAC$_{UC}$', show_xlabel=show_xlabel)

        print(f'Loaded {rolling_years}-year TAC:', path, tac.shape, ids.shape)

    fig_to_path = os.path.join(current_dir, '4_Figures', 'Fig03_temporal_tac_opt_modis_rolling5yr.png')
    os.makedirs(os.path.dirname(fig_to_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_to_path, dpi=900)
    print('Saved:', fig_to_path)
