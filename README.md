# A starting library for future satellite prediction


### Getting started

Create and activate a new python environment, e.g.

```
conda create -n sat_pred python=3.10
conda activate sat_pred
```

Clone this repo

```
git clone https://github.com/openclimatefix/sat_pred.git

```

Install the package and its dependencies

```
cd sat_pred
pip install -e .
```

Create a empty directory to save the satellite data to

```
mkdir path/to/new/satellite/directory
```

Run the command line utility to download download satellite data

```
python scripts/download_uk_satellite.py \
  "2020-06-01 00:00" \
  "2020-06-30 23:55" \
  "path/to/new/satellite/directory"
```

The above script downloads all the satellite imagery from June 2020. The input arguments are:
 - start_date: First datetime (inclusive) to download
 - end_date: Last datetime (inclusive) to download
 - output_directory: Directory to which the satellite data should be saved

Note that the above script creates a satellite dataset which is 21GB. On my machine it used about
12GB of RAM at its peak and took around 30 minutes to run.


See the notebook `01-plot_satellite_image_example.ipynb` for loading and plotting example.


See the notebook `02-data_loader_demo.ipynb` for getting started with the dataloader.