# Global Urban Vegetation Resilience

This repository contains Python scripts used to analyze global urban vegetation resilience from satellite and climate datasets. The workflow compares vegetation temporal autocorrelation, disturbance response, recovery, and related environmental drivers across urban cores, urban edges, and rural reference areas.

## Repository contents

- `S.*.py`: data extraction and preprocessing scripts, including Google Earth Engine exports, climate anomalies, land-cover fractions, canopy height, emissivity, and related inputs.
- `step_*.py`: main analysis scripts for calculating temporal autocorrelation, urban-rural differences, sensitivity tests, disturbance metrics, and robustness checks.
- `Fig.*.py`: scripts used to generate main and supplementary figures.
- `Rd1_*.py`: additional analyses prepared for reviewer-response checks and sensitivity analyses.
- `geeCodes.py`: helper functions used by Google Earth Engine extraction scripts.

## Data layout

Most scripts assume this directory is located inside a project folder with sibling data/output folders such as:

```text
1_Input/
2_Output/
4_Figures/
3_Code/
```

Large input rasters, intermediate arrays, and figure outputs are not included in this repository. Update local paths in individual scripts as needed before running them.

## Environment

The scripts are written in Python and commonly use:

```text
numpy, pandas, scipy, matplotlib, seaborn, geopandas, rasterio,
tifffile, OpenCV, scikit-learn, xgboost, shap, earthengine-api
```

Some scripts require Google Earth Engine authentication:

```bash
earthengine authenticate
```

## Usage

Run scripts individually from the `3_Code` directory according to the analysis stage needed. A typical workflow is:

1. Extract or prepare input datasets with `S.*.py`.
2. Calculate vegetation resilience metrics with `step_*.py`.
3. Generate figures with `Fig.*.py`.
4. Run additional sensitivity or reviewer-response analyses with `Rd1_*.py`.

Because paths, datasets, and processing choices are script-specific, inspect each script before execution.
