import argparse
import os

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/matplotlib-cache')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clean_series(df, column):
    values = pd.to_numeric(df[column], errors='coerce').to_numpy(dtype=float)
    return values[np.isfinite(values)]


def format_stats(values):
    return (
        f'n = {values.size}, '
        f'mean = {np.nanmean(values):.4f}, '
        f'median = {np.nanmedian(values):.4f}'
    )


def format_direction_percentages(values):
    pct_positive = np.sum(values > 0) * 100.0 / values.size
    pct_negative = np.sum(values < 0) * 100.0 / values.size
    return f'> 0: {pct_positive:.1f}%\n< 0: {pct_negative:.1f}%'


def build_bins(*arrays, bins=30):
    combined = np.concatenate([arr for arr in arrays if arr.size > 0])
    if combined.size == 0:
        raise ValueError('No finite trend values found for plotting.')

    lower, upper = np.nanpercentile(combined, [1, 99])
    if lower == upper:
        lower, upper = np.nanmin(combined), np.nanmax(combined)
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    return np.linspace(lower, upper, bins + 1)


def plot_histogram(df, output_path, bins=30, dpi=600):
    required_columns = ['uc_vi_trend', 'ru_vi_trend']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    urban_core = clean_series(df, 'uc_vi_trend')
    rural_area = clean_series(df, 'ru_vi_trend')
    hist_bins = build_bins(urban_core, rural_area, bins=bins)

    plt.rc('font', family='Arial')
    plt.rc('axes', linewidth=1.0)
    plt.rc('xtick.major', size=4, width=1.0)
    plt.rc('ytick.major', size=4, width=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharex=True, sharey=True)
    plot_specs = (
        {
            'values': urban_core,
            'title': 'Urban core',
            'bar_color': '#4C78A8',
            'line_color': '#1F4E79',
        },
        {
            'values': rural_area,
            'title': 'Rural area',
            'bar_color': '#59A14F',
            'line_color': '#2F6B2F',
        },
    )

    for ax, spec in zip(axes, plot_specs):
        values = spec['values']
        weights = np.ones_like(values) * 100.0 / values.size

        ax.hist(
            values,
            bins=hist_bins,
            weights=weights,
            color=spec['bar_color'],
            alpha=0.72,
            edgecolor='white',
            linewidth=0.7,
        )

        ax.axvline(0, color='0.25', linewidth=0.9)
        ax.set_title(spec['title'], fontsize=11)
        ax.set_xlabel('kNDVI trend (year$^{-1}$)')
        ax.tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(
            0.04,
            0.96,
            f'{format_direction_percentages(values)}',
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=8.5,
        )

    axes[0].set_ylabel('Cities (%)')

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    paired = df[['uc_vi_trend', 'ru_vi_trend']].apply(pd.to_numeric, errors='coerce')
    paired = paired.dropna()

    return urban_core, rural_area, paired


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    default_csv = os.path.join(project_dir, '2_Output', 'vi_trends_mds.csv')
    default_output = os.path.join(project_dir, '4_Figures', 'FigSx_kndvi_trend_histogram_mds.png')

    parser = argparse.ArgumentParser(
        description='Plot histograms of kNDVI trend for urban core and rural areas.'
    )
    parser.add_argument('--input-csv', default=default_csv, help='Path to vi_trends_mds.csv.')
    parser.add_argument('--output', default=default_output, help='Output figure path.')
    parser.add_argument('--bins', type=int, default=30, help='Number of histogram bins.')
    parser.add_argument('--dpi', type=int, default=600, help='Output figure DPI.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trends = pd.read_csv(args.input_csv)
    urban_core_trend, rural_area_trend, paired_trends = plot_histogram(
        trends,
        output_path=args.output,
        bins=args.bins,
        dpi=args.dpi,
    )

    print('Saved histogram to:', args.output)
    print('Urban core:', format_stats(urban_core_trend))
    print('Urban core:', format_direction_percentages(urban_core_trend).replace('\n', ', '))
    print('Rural area:', format_stats(rural_area_trend))
    print('Rural area:', format_direction_percentages(rural_area_trend).replace('\n', ', '))
    print(
        'Mean urban-rural difference:',
        f"{np.nanmean(paired_trends['uc_vi_trend'] - paired_trends['ru_vi_trend']):.4f}",
    )
