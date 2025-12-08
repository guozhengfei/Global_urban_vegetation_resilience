import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
from pathlib import Path

def cal_monthly_mean(flux_df,output_columns):
    time_label = flux_df['TIMESTAMP_START']

    # parse TIMESTAMP_START values like "202412312330" -> datetime and split into components
    time_str = time_label.astype(str)  # ensure string
    ts = pd.to_datetime(time_str, format='%Y%m%d%H%M', errors='coerce')  # NaT for invalid
    flux_df['year'] = ts.dt.year
    flux_df['month'] = ts.dt.month
    flux_df['day'] = ts.dt.day
    flux_df['hour'] = ts.dt.hour
    flux_df['minute'] = ts.dt.minute

    for col in output_columns:
        try:
            flux_df.loc[flux_df[col] ==-9999, col] = np.nan
        except KeyError:
            flux_df[col] = np.nan
    # aggregate monthly mean for LE and H (group by year and month)
    monthly_mean_LE_H = flux_df.groupby(['year', 'month'])[output_columns].mean().reset_index()
    return monthly_mean_LE_H


def find_subfolders_with_patterns(root_dir, patterns):
    """
    Find all subfolders whose names include any of the specified patterns,
    while ignoring hidden folders (those starting with '.').

    Args:
        root_dir (str): The root directory to search in
        patterns (list): List of strings to match in folder names

    Returns:
        list: Path objects of matching subfolders
    """
    root_path = Path(root_dir)
    matching_folders = []

    if not root_path.exists():
        print(f"Error: Directory '{root_dir}' does not exist.")
        return matching_folders

    if not root_path.is_dir():
        print(f"Error: '{root_dir}' is not a directory.")
        return matching_folders

    for folder in root_path.rglob('*'):
        if folder.is_dir() and not folder.name.startswith('.'):
            for pattern in patterns:
                if pattern in folder.name:
                    matching_folders.append(folder)
                    break  # No need to check other patterns once one matches

    return matching_folders

# Example usage
if __name__ == "__main__":
    urban_ICOS = ['CH-BaK', 'IT-OXm', 'FR-Tou', 'DE-BeR', 'IT-Sas', 'GR-HeK', 'GR-HeM', 'FI-Kmp']
    urban_AMF = ['US-EDN', 'US-ORv', 'US-Ro3']
    ICOS_site_info = pd.read_csv(os.path.join('..', '..', 'Fluxnet2025', 'ICOS', 'ICOS_stations.csv'))

    df_ICOS = []
    for site in urban_ICOS:
        subfolder = 'ICOSETC_'+site+'_ARCHIVE_L2'
        filename_flux = 'ICOSETC_'+site+'_FLUXES_L2.csv'
        filename_meteo = 'ICOSETC_'+site+'_METEO_L2.csv'
        flux_df = pd.read_csv(os.path.join('..', '..', 'Fluxnet2025', 'ICOS', subfolder,filename_flux))
        meteo_df = pd.read_csv(os.path.join('..', '..', 'Fluxnet2025', 'ICOS', subfolder,filename_meteo))

        monthly_LE_H = cal_monthly_mean(flux_df,['LE','H'])
        monthly_meteo = cal_monthly_mean(meteo_df,['LW_IN','LW_OUT','SW_IN','SW_OUT','PA','RH','TA','WS'])
        merge_df = pd.concat([monthly_LE_H,monthly_meteo.iloc[:,2:]],axis=1)
        merge_df['site']=site
        merge_df['lon'] = ICOS_site_info.loc[ICOS_site_info['site_ID'] == site, 'lon'].values[0]
        merge_df['lat'] = ICOS_site_info.loc[ICOS_site_info['site_ID'] == site, 'lat'].values[0]
        df_ICOS.append(merge_df)
        print(site)

    df_ICOS2 = pd.concat(df_ICOS,axis=0)

    filenames = find_subfolders_with_patterns(os.path.join('..', '..', 'Fluxnet2025', 'AMF'),urban_AMF)
    AMF_site_info = pd.read_csv(os.path.join('..', '..', 'Fluxnet2025', 'AMF', 'AmeriFlux-site-info.csv'))
    df_AMF = []
    for filename in filenames:
        subfolder = str(filename)
        filename_flux0 = subfolder.split('/')[-1].split('SUBSET')
        filename_flux = filename_flux0[0]+'SUBSET'+'_HH'+filename_flux0[1]+'.csv'
        flux_df = pd.read_csv(os.path.join(subfolder, filename_flux))

        monthly_meteo = cal_monthly_mean(flux_df, ['LE_F_MDS', 'H_F_MDS', 'LW_IN_F', 'LW_OUT', 'SW_IN_F', 'SW_OUT', 'PA_F', 'RH', 'TA_F', 'WS_F'])
        site =filename_flux0[0].split('_')[1]
        monthly_meteo['site'] = site
        monthly_meteo['lon'] = AMF_site_info.loc[AMF_site_info['Site ID'] == site, 'lon'].values[0]
        monthly_meteo['lat'] = AMF_site_info.loc[AMF_site_info['Site ID'] == site, 'lat'].values[0]
        df_AMF.append(monthly_meteo)
        print(site)

    df_AMF2 = pd.concat(df_AMF, axis=0)
    df_AMF2.columns = df_ICOS2.columns

    df_urban1 = pd.concat([df_ICOS2,df_AMF2],axis=0)
    df_urban2 = pd.read_csv('/Volumes/Zhengfei_01/project 1 urban vegetation thermoregulation/2_Output/urban_flux_data.csv')
    time_label = df_urban2['time']
    time_str = time_label.astype(str)  # ensure string
    ts = pd.to_datetime(time_str, format='%Y-%m-%d', errors='coerce')  # NaT for invalid
    df_urban2['year'] = ts.dt.year
    df_urban2['month'] = ts.dt.month
    df_urban2['RH'] = np.nan
    df_urban2['WS'] = (df_urban2['Wind_N']**2 +df_urban2['Wind_E']**2)**0.5
    df_urban2['PA'] = df_urban2['PSurf']/1000

    df_urban2_match = df_urban2[['year','month','Qle', 'Qh', 'LWdown', 'LWup', 'SWdown', 'SWup', 'PA', 'RH', 'Tair', 'WS','site','longitude','latitude']]
    df_urban2_match.columns = df_urban1.columns

    df_urban = pd.concat([df_urban1,df_urban2_match],axis=0)
    df_urban.to_csv(os.path.join('..', '..', 'Fluxnet2025', 'urban_flux_30_sites.csv'))


