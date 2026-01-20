# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project analyzes mouse home cage monitoring data from Smart-Kages (Cambridge Phenotyping) tracked with DeepLabCut, using the [movement](https://movement.neuroinformatics.com) package. The workflow processes pose estimation data into xarray datasets with datetime indices for time-series analysis.

## Common Commands

```bash
# Install (requires conda environment with pytables and ffmpeg)
conda create -n smart-kages -c conda-forge python=3.13 pytables ffmpeg
pip install -e .           # basic install
pip install -e '.[dev]'    # with dev dependencies

# Run tests
pytest                     # runs with coverage by default
pytest tests/test_unit/    # unit tests only
pytest -v tests/test_unit/test_placeholder.py  # single test file

# Linting and formatting
ruff check .               # lint
ruff format .              # format
pre-commit run --all-files # run all pre-commit hooks (ruff, mypy, codespell, etc.)

# Type checking
mypy smart_kages_movement/
```

## Architecture

### Data Flow

The analysis pipeline is notebook-driven, with helper functions in `smart_kages_movement/`:

1. **Parse** (`01_parse_data_into_df.ipynb`, `io.py`): Discover kage directories, construct DataFrame with pose/video paths, multi-indexed by (kage, date, hour)
2. **Load** (`02_load_kages_as_movement_ds.ipynb`, `datetime.py`): Load DLC `.h5` files, assign datetime index from `adjustments.txt` and `corrected_timestamps.pkl`, save as NetCDF
3. **QC** (`03_QC_and_cleaning.ipynb`, `reports.py`, `plots.py`): Filter low-confidence points, smooth, interpolate gaps, generate QC plots
4. **Select** (`04_select_data_for_analysis.ipynb`): Choose best week and keypoints based on QC
5. **Analyze** (`05_best_week_overview.ipynb`): Generate comparison plots across cohort

### Package Modules

- `io.py`: Data discovery and parsing (expects `kageN/videos/YYYY/MM/DD/` and `kageN/analysis/dlc_output/YYYY/MM/DD/` structure)
- `datetime.py`: Timestamp extraction from `adjustments.txt` and video frames via ffprobe, segment overlap detection
- `reports.py`: Daily missing keypoint and empty frame counting using xarray groupby
- `plots.py`: QC visualizations (heatmaps, histograms, trajectories, speed plots)

### Key Dependencies

- `movement[napari]>=0.10.0`: Core analysis library for pose data
- `xarray`: Primary data structure (datasets with `time`, `space`, `keypoints`, `individuals` dimensions)
- `sleap_io`: Video loading
- `ffprobe` (system): Frame timestamp extraction

### Data Conventions

- Datasets have attributes: `kage` (name), `fps` (framerate)
- Position DataArrays have dimensions: `time`, `space` (x, y), `keypoints`, `individuals`
- Time dimension uses pandas datetime index for groupby operations (`time.date`, etc.)
