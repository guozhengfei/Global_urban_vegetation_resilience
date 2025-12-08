import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp
from statsmodels.formula.api import ols
import statsmodels.api as sm
import os

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
relative_path = '/2_Output/tac_nadir_city_3zones.npz'
relative_path2 = '/2_Output/tac_landsat_city_3zones.npz'

TACs = np.load(current_dir+relative_path)['array1'] # v5,v4.2
ID = np.load(current_dir+relative_path)['array2'] # v5,v4.2
ID2 = np.load(current_dir+relative_path2)['array2']
TACs_mean = np.nanmean(TACs, axis=2)
tac_global_mean = np.nanmean(TACs_mean, axis=0)

df_tac = pd.DataFrame(TACs_mean)
df_tac.columns=['urban_core','urban_edge','rural']
df_tac['ID'] = ID
df_tac = df_tac[df_tac['ID'].isin(ID2)]

# 2. 数据整合为DataFrame
df_tac.dropna(subset=['urban_core', 'urban_edge', 'rural'], inplace=True)
df_long = pd.melt(df_tac, id_vars=['ID'], value_vars=['urban_core', 'urban_edge', 'rural'],
                  var_name='area', value_name='tac')
df_long['area'] = df_long['area'].map({'urban_core': 'Urban Core', 'urban_edge': 'Urban Edge', 'rural': 'Rural'})


# 3. Friedman test (overall difference test for repeated measures)
friedman_stat, friedman_p = stats.friedmanchisquare(
    df_tac['urban_core'],
    df_tac['urban_edge'],
    df_tac['rural']
)

print(f"Friedman test result: statistic = {friedman_stat:.4f}, p = {friedman_p:.6f}")

# 4. If overall significant, perform Nemenyi post-hoc test
if friedman_p < 0.05:
    print("\nPerforming Nemenyi post-hoc test...")
    # Perform Nemenyi test using posthoc_nemenyi_friedman
    nemenyi_result = sp.posthoc_nemenyi_friedman(df_tac[['urban_core', 'urban_edge', 'rural']].to_numpy())
    nemenyi_result.columns = ['Urban Core', 'Urban Edge', 'Rural']
    nemenyi_result.index = ['Urban Core', 'Urban Edge', 'Rural']


    # Extract target comparisons
    comparisons = [
        ('Urban Core', 'Urban Edge'),
        ('Urban Core', 'Rural'),
        ('Urban Edge', 'Rural')
    ]

    # Print key results
    print("\nCorrected p-values for key group comparisons:")
    for group1, group2 in comparisons:
        p_adj = nemenyi_result.loc[group1, group2]
        sig_marker = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
        print(f"{group1} vs {group2}: p_adj = {p_adj:.5f} {sig_marker}")

    # 5. Visualization: Boxplot + significance annotation
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='area', y='tac', data=df_long, order=['Urban Core', 'Urban Edge', 'Rural'], palette='viridis')
    plt.title('Distribution of TAC Values by Area (Friedman Test Significant)', fontsize=14)
    plt.xlabel('Area Type', fontsize=12)
    plt.ylabel('TAC Value', fontsize=12)

    # Add significance markers (based on corrected p-values)
    y_max = df_long['tac'].max()
    y_range = y_max - df_long['tac'].min()
    y_max = y_max + y_range * 0.1

    sig_comparisons = []
    for comp in comparisons:
        if nemenyi_result.loc[comp[0], comp[1]] < 0.05:
            sig_comparisons.append(comp)

    for i, (group1, group2) in enumerate(sig_comparisons):
        p_adj = nemenyi_result.loc[group1, group2]
        x1 = ['Urban Core', 'Urban Edge', 'Rural'].index(group1)
        x2 = ['Urban Core', 'Urban Edge', 'Rural'].index(group2)
        y = y_max + i * y_range * 0.15
        plt.plot([x1, x1, x2, x2], [y, y + y_range*0.02, y + y_range*0.02, y], lw=1.5, c='k')
        star_count = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*"
        plt.text((x1 + x2) * 0.5, y + y_range*0.03, star_count, ha='center', va='bottom', color='k')

    if sig_comparisons:
        plt.ylim(top=y_max + (len(sig_comparisons)) * y_range * 0.15)

    plt.tight_layout()
    # plt.savefig('TAC_comparison_nemenyi.png', dpi=300)
    plt.show()

    # 6. Export complete Nemenyi test results (matrix format)
    print("\nComplete Nemenyi test results matrix (corrected p-values):")
    print(nemenyi_result.round(5))
else:
    print("Friedman test not significant (p >= 0.05), no post-hoc test needed.")