import pandas as pd
import os
from datetime import datetime
import numpy as np

sos_filepath = os.path.join('..','..','urban_env_data','sos_csv_urban_751.csv')
eos_filepath = os.path.join('..','..','urban_env_data','eos_csv_urban_751.csv')

df_sos = pd.read_csv(sos_filepath).iloc[:,1:-4]
df_eos = pd.read_csv(eos_filepath).iloc[:,1:-4]

dates = [col.split('_O')[0] for col in df_sos.columns]
start_date = datetime.strptime('2000-01-01', '%Y-%m-%d')

for i in range(len(dates)):
    end_date = datetime.strptime(dates[i], '%Y_%m_%d')
    offset = (end_date - start_date).days
    df_sos.iloc[:,i] = df_sos.iloc[:,i]-offset
    df_eos.iloc[:, i] = df_eos.iloc[:, i] - offset

sos_mean = df_sos.mean(axis=1)
eos_mean = df_eos.mean(axis=1)

# Number of regions
n_regions = len(sos_mean)
# Days in each month (non-leap year)
days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
# Cumulative DOY at the start of each month
month_start_doy = np.cumsum([0] + days_in_month[:-1]) + 1  # Jan=1, Feb=32, ...

# Create growing season label array: shape (n_regions, 12)
growing_season_label = np.zeros((n_regions, 12), dtype=int)

for i in range(n_regions):
    sos = sos_mean.iloc[i]
    eos = eos_mean.iloc[i]
    for m in range(12):
        # If any part of the month is within the growing season, label as 1
        month_start = month_start_doy[m]
        month_end = month_start + days_in_month[m] - 1
        # Handle cases where eos < sos (season crosses year end)
        if sos <= eos:
            if (month_end >= sos) and (month_start <= eos):
                growing_season_label[i, m] = 1
        else:
            # Growing season wraps around year end
            if (month_end >= sos) or (month_start <= eos):
                growing_season_label[i, m] = 1

# growing_season_label[i, m]: 1 if month m is in growing season for region i, else 0
