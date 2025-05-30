# smart-kages-movement

Analysing data from [Smart-Kages](https://cambridgephenotyping.com/products)
using [movement](https://movement.neuroinformatics.com).

## Installation

First, create a conda environment with some required dependencies:

```bash
conda create -n smart-kages -c conda-forge python=3.13 pytables
conda activate smart-kages
```

Then, clone this repository, navigate to its directory, and install the package:

```bash
git clone https://github.com/neuroinformatics-unit/smart-kages-movement.git
cd smart-kages-movement
pip install -e .
```

To contribute, make sure to include the `dev` dependencies as well:

```bash
pip install -e '.[dev]'
```

## Usage

The main way to use the code is by working through the Jupyter notebooks in the `notebooks` directory, in the order they are listed.

The `smart_kages_movement` package, which you installed earlier, provides some helper functions that are imported and used within the notebooks.

The notebooks are as follows:

* `01_parse_data_into_df.ipynb`: Parses data paths from the Smart-Kages folder structure and stores them in pandas DataFrames. Also loads time adjustments to help estimate start and end times for each 1-hour segment, and identifies potential issues with the data, such as overlapping segments.
* `02_load_kages_as_movement_ds.ipynb`: Loads all DeepLabCut `.h5` pose files for each kage and concatenates them into a single `movement` dataset per kage. Also assigns a datetime index across the `time` dimension for easy access, and saves the resulting datasets to NetCDF files.
* `03_diagnostic_plots.ipynb`: Still a work in progress—stay tuned!

## Input/Output Data Structure

We expect all data to be stored under a single folder, hereafter referred to as `DATA_DIR`, with subfolders for each kage, named `kage1`, `kage2`, etc.

Each kage folder should contain at least the `videos/` and `analysis/dlc_output/` subfolders, which are themselves hierarchically subdivided by date, i.e. `YYYY/MM/DD/`. Each day's folder contains videos and DeepLabCut predictions saved as `.h5` files, split into 1-hour segments.

The `videos/YYYY/MM/DD/` subfolder is also expected to contain an `adjustments.txt` file, which contains time adjustments for each 1-hour segment. This file is used to calculate the start datetime for each segment. Segment
end datetimes are estimated by counting the number of video frames and assuming a constant frame rate of 2 fps.

A `DATA_DIR/movement_analysis/` subfolder is created to store the outputs, but the path can be customised in the notebooks.


## License
This code is licensed under the [3-Clause BSD License](https://opensource.org/license/bsd-3-clause), see the [LICENSE](LICENSE) file for details.
