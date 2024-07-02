"""A script to download a selection of EUMETSAT satellite imagery from the google public dataset

Example usage from command line:

```
python download_uk_satellite.py \
  "2020-06-01 00:00" \
  "2020-06-30 23:55" \
  "path/to/new/satellite/directory"
```

Note that the output directory must already exist. This script will create a zarr directory within
the supplied output directory.
"""

import os
import logging
import typer
import warnings

import xarray as xr
import pandas as pd

from dask.diagnostics import ProgressBar
from dask.distributed import LocalCluster

from ocf_datapipes.utils.geospatial import lon_lat_to_geostationary_area_coords
import ocf_blosc2


xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Our default spatial bounds for satellite images "over" the uk. See notebook for example image
# These are in degrees
default_lon_min = -16
default_lon_max = 10
default_lat_min = 45
default_lat_max = 70


def rechunk(ds):
    """Rechunk the satellite data"""
    for v in ds.variables:
        del ds[v].encoding['chunks']
    
    target_chunks_dict=dict(time=1, x_geostationary=100, y_geostationary=100, variable=-1)
    ds = ds.chunk(target_chunks_dict)
    return ds


def get_sat_public_dataset_path(year, is_hrv=False):
    """Get the path to the Google Public Dataset of EUMETSAT satellite data"""
    file_end = "hrv.zarr" if is_hrv else "nonhrv.zarr"
    
    path = (
        "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/"
        f"{year}_{file_end}"
    )
    
    return path


def download_satellite_data(
    start_date: str, 
    end_date: str, 
    output_directory: str,
    lon_min: float=default_lon_min,
    lon_max: float=default_lon_max,
    lat_min: float=default_lat_min,
    lat_max: float=default_lat_max,
    get_hrv: bool=False,
    override_date_bounds: bool=False, 
    use_localcluster: bool=False,
):
    """Download a selection of the available EUMETSAT data.
    
    Each calendar year of data within the supplied date range will be saved to a separate file in
    the output directory
    
    Args:
        start_date: First datetime (inclusive) to download
        end_date: Last datetime (inclusive) to download
        output_directory: Directory to which the satellite data should be saved
        lon_min: The west-most longitude (in degrees) of the bounding box to download
        lon_min: The east-most longitude (in degrees) of the bounding box to download
        lat_min: The south-most latitude (in degrees) of the bounding box to download
        lat_max: The south-most latitude (in degrees) of the bounding box to download
        get_hrv: Whether to download the HRV data, else non-HRV is downloaded
        override_date_bounds: Whether to override the date range limits
        use_localcluster: Whether to use a dask local cluster for the download
    """
    
    # Check output directory exists
    if not os.path.isdir(output_directory):
        raise FileNotFoundError(
            f"Outut directory {output_directory} does not exist. "
            "Please create it before attempting to download satellite data."
        )
    
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
       
    # Check date range for known errors
    if not override_date_bounds:
        if start_date < pd.Timestamp("2018"):
            raise ValueError(
                "There are currently some issues with the EUMETSAT data before 2019/01/01."
                "We recommend only using data from this date forward."
                "To override this error set `override_date_bounds=True`"
            )
    
    years = range(start_date.year, end_date.year+1)
    
    # Check that none of the filenames we will save to already exist. If they exist this will cause
    # a zarr error
    file_end = "hrv.zarr" if get_hrv else "nonhrv.zarr"
    
    for year in years:
        output_zarr_file = f"{output_directory}/{year}_{file_end}"
        if os.path.exists(output_zarr_file):
            raise ValueError(
                f"The zarr file {output_zarr_file} already exists. "
                "This function will not overwrite data"
            )
    
    # Create a dask local cluster to run the compute on - this is appropriate for running on your 
    # local machine but may not be appropriate for larger shared resources since it uses a lot of
    # the available resources
    if use_localcluster:
        cluster = LocalCluster()
        client = cluster.get_client()
    
    for year in years:
        logger.info(f"Downloading data from {year}")
        
        path = get_sat_public_dataset_path(year, is_hrv=get_hrv)
        
        # Slice the data from this year which are between the start and end dates
        ds = xr.open_zarr(path).sortby("time").sel(time=slice(start_date, end_date))
        
        # Convert lon-lat bounds to geostationary-coords        
        ((x_min, x_max), (y_min, y_max)) = lon_lat_to_geostationary_area_coords(
            [lon_min, lon_max],
            [lat_min, lat_max], 
            ds.data,
        )

        # Define the spatial area to slice from
        ds = ds.sel(
            x_geostationary=slice(x_max, x_min), # x-axis is in decreasing order
            y_geostationary=slice(y_min, y_max),
        )
        
        # Rechunk the satellite data
        ds = rechunk(ds)
        
        # Save data
        output_zarr_file = f"{output_directory}/{year}_{file_end}"
        with ProgressBar(dt=5):
            ds.to_zarr(output_zarr_file)
            
    client.close()

    
if __name__=="__main__":
    typer.run(download_satellite_data)