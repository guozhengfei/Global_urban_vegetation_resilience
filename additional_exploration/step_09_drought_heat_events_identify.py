import pandas as pd
import os
import numpy as np
import matplotlib;

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

df_spei = pd.read_csv(os.path.join('..', '..', 'urban_env_data', 'spei_csv_urban_751.csv')).iloc[:,1:-4]
df_ta = pd.read_csv(os.path.join('..', '..', 'urban_env_data', 'Ta_era5L_csv_urban_751.csv')).iloc[:,1:-16]

plt.figure(); plt.plot(df_spei.iloc[0,:].values)
# Thresholds for drought event identification
R0 = -0.5  # SPEI threshold for merging events
R1 = -1.0  # SPEI threshold for potential drought
R2 = -1.0  # SPEI threshold for filtering short events

def identify_drought_events(spei_series, R0, R1, R2):
    drought_events = []
    in_event = False
    event_start = None
    event_end = None

    for i, spei in enumerate(spei_series):
        if spei < R1:
            if not in_event:
                in_event = True
                event_start = i
            event_end = i
        else:
            if in_event:
                # Check if event is only 1 month and SPEI > R2
                if event_end - event_start == 0 and spei_series[event_start] > R2:
                    pass  # Discard minor event
                else:
                    drought_events.append((event_start, event_end))
                in_event = False

    # Handle event at end of series
    if in_event:
        if event_end - event_start == 0 and spei_series[event_start] > R2:
            pass
        else:
            drought_events.append((event_start, event_end))

    # Merge events with 1-month gap and gap SPEI < R0
    merged_events = []
    i = 0
    while i < len(drought_events):
        start, end = drought_events[i]
        while i + 1 < len(drought_events):
            next_start, next_end = drought_events[i+1]
            if next_start - end == 2 and spei_series[end+1] < R0:
                end = next_end
                i += 1
            else:
                break
        if (end-start+1)>=3:
            merged_events.append((start, end))
        i += 1

    return merged_events

# Example usage for all cities
drought_events_dict = {}
for idx, row in df_spei.iterrows():
    drought_events_dict[idx] = identify_drought_events(row.values, R0, R1, R2)

# Remove seasonality: calculate monthly climatology for each city
months = [col[4:6] for col in df_ta.columns]  # Assumes columns like '2001-01', '2001-02', ...
unique_months = sorted(set(months))

# Build monthly climatology for baseline period (2001-2010)
climatology_mean = pd.DataFrame(index=df_ta.index, columns=unique_months, dtype=float)
climatology_std = pd.DataFrame(index=df_ta.index, columns=unique_months, dtype=float)
climatology_p95 = pd.DataFrame(index=df_ta.index, columns=unique_months, dtype=float)

for m in unique_months:
    month_cols = [col for col in df_ta.columns if m == col[4:6]]
    climatology_mean[m] = df_ta[month_cols].mean(axis=1)
    climatology_std[m] = df_ta[month_cols].std(axis=1)
    climatology_p95[m]= np.percentile(df_ta[month_cols],95,axis=1)

# Calculate anomalies and detect heat events
heat_events_dict = {}
for idx, row in df_ta.iterrows():
    events = []
    for i, col in enumerate(df_ta.columns):
        month = col[4:6]
        anomaly = row[col] - climatology_mean.loc[idx, month]
        threshold = climatology_p95.loc[idx, month]#climatology_mean.loc[idx, month] + 1.5 * climatology_std.loc[idx, month]
        if row[col] > threshold:
            events.append(i)
    heat_events_dict[idx] = events

# Results:
# drought_events_dict: {city_index: [(start_idx, end_idx), ...]}
# heat_events_dict: {city_index: [month_indices_of_extreme_heat, ...]}

# Save drought events
drought_events_df = pd.DataFrame([
    {'city_index': idx, 'drought_events': drought_events_dict[idx]}
    for idx in drought_events_dict
])
drought_events_df['ID'] = pd.read_csv(os.path.join('..', '..', 'urban_env_data', 'spei_csv_urban_751.csv'))['ID']
drought_events_df.to_csv(current_dir+'/2_Output/drought_events_results.csv', index=False)

# Save heat events
heat_events_df = pd.DataFrame([
    {'city_index': idx, 'heat_events': heat_events_dict[idx]}
    for idx in heat_events_dict
])
heat_events_df['ID'] = pd.read_csv(os.path.join('..', '..', 'urban_env_data', 'Ta_era5L_csv_urban_751.csv'))['ID']
heat_events_df.to_csv(current_dir+'/2_Output/heat_events_results.csv', index=False)

