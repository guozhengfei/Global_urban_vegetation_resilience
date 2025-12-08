import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;
matplotlib.use('Qt5Agg')
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore")
import cv2
from scipy.interpolate import griddata

def bilinear_interpolation_fast(array):
    """
    Fill NaN values in a 2D array using bilinear interpolation with optimized performance.
    """
    # Create a copy of the array
    result = array.copy()
    rows, cols = array.shape

    # Create coordinate grid
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Find non-NaN points
    mask = ~np.isnan(array)
    points = np.array([y[mask].ravel(), x[mask].ravel()]).T
    values = array[mask].ravel()

    # Find NaN locations
    nan_mask = np.isnan(array)
    nan_points = np.array([y[nan_mask].ravel(), x[nan_mask].ravel()]).T

    if len(nan_points) == 0:
        return result

    # Perform bilinear interpolation using griddata
    interpolated_values = griddata(
        points,
        values,
        nan_points,
        method='linear',  # 'linear' gives bilinear interpolation
        fill_value=np.nan  # In case some points can't be interpolated
    )

    # Fill the interpolated values back into the array
    result[nan_mask] = interpolated_values

    # Handle any remaining NaN values (near edges where interpolation might fail)
    if np.any(np.isnan(result)):
        # Simple fallback: fill with nearest neighbor
        from scipy.ndimage import distance_transform_edt
        mask = ~np.isnan(result)
        distances, indices = distance_transform_edt(
            ~mask,
            return_indices=True
        )
        result[~mask] = result[
            indices[0][~mask],
            indices[1][~mask]
        ]

    return result

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/nadir_ndvi_15d_500/'
    urbanExp_folder = current_dir + '/1_Input/urban_expansion_frac_500m/'
    Data_folder = os.path.dirname(current_dir) + 'urban_env_data/solarRad_ear5Land_urban/'
    filenames = os.listdir(Data_folder)

    IDs = []
    for name in filenames:
        id = float(name.split('_')[-1].split('.tif')[0])
        IDs.append(id)

    IDs = np.sort(IDs)
    yr_num = 23
    bands_year = 12

    for id in IDs:
        EVI0 = tf.imread(EVI_folder + 'nadir_ndvi_15d_' + str(id) + '.tif')[:,:,0]
        data_i = tf.imread(Data_folder + 'Rad_monthly_' + str(id) + '.tif')#[:, :, :264 * 2]
        if np.isnan(data_i).sum() / data_i.size >= 0.5:
            nanmean_values = np.nanmean(data_i, axis=(0, 1))
            data_i = np.where(np.isnan(data_i), nanmean_values, data_i)
        data_rsp_i = []
        for i in range(data_i.shape[2]):
            orgD = data_i[:, :, i] * 1
            fillD = bilinear_interpolation_fast(orgD)
            rsp_i = cv2.resize(fillD, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)
            data_rsp_i.append(rsp_i)
        data_rsp_i = np.stack(data_rsp_i, axis=2)

        # remove seasonality and trend
        # calculate monthly anomaly
        EVI = data_rsp_i.reshape((data_rsp_i.shape[0]*data_rsp_i.shape[1],data_rsp_i.shape[2]))
        EVI_yr = np.zeros_like(EVI)
        for year in range(yr_num):
            st = year * bands_year
            ed = st + bands_year
            evi_year = np.nanmean(EVI[:, st:ed], axis=1)
            evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_year, axis=1)
            EVI_yr[:, st:ed] = evi_year_rep
        rm_offline = EVI - EVI_yr

        Evi_sea_rep = np.zeros_like(rm_offline)
        for yr in range(yr_num):
            start_index = (yr - 3) * bands_year
            end_index = (yr + 3) * bands_year
            if yr < 3: start_index = 0; end_index = 6 * bands_year
            if yr > (yr_num - 3): start_index = (yr_num - 6) * bands_year; end_index = yr_num * bands_year
            data_yr = rm_offline[:, start_index:end_index]
            Evi_sea = np.mean(np.reshape(data_yr, (data_yr.shape[0], int(data_yr.shape[1] / bands_year), bands_year)), axis=1)
            Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea
        res = rm_offline - Evi_sea_rep
        Ta_anom = res.reshape(data_rsp_i.shape)

        np.save(current_dir + '/1_Input/rad_anom/rad_month_anomaly_'+str(id)+'.npy', Ta_anom)
        print(id)
