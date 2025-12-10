import pandas as pd
import xarray as xr
from movement.validators.arrays import validate_dims_coords


def _squeeze_individuals_dim(da: xr.DataArray) -> xr.DataArray:
    """Squeeze out individuals dimension if it has size 1."""
    if "individuals" in da.dims and da.sizes["individuals"] == 1:
        da = da.squeeze().drop_vars("individuals")
    return da


def count_missing_keypoints_daily(
    da: xr.DataArray,
) -> xr.DataArray:
    """Generate a daily report of missing keypoints.

    Per day, we compute the % of frames during which a given keypoint
    was missing (NaN) in the provided DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        The input data with ``time`` and ``space`` dimensions.

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the daily % of
        missing values per keypoint.

    """
    validate_dims_coords(da, {"time": [], "space": []})
    da = _squeeze_individuals_dim(da)

    # Boolean DataArray indicating when EACH keypoint is missing (NaN)
    kp_missing = da.isnull().any("space")
    # Return daily % of missing values per keypoint
    daily_missing_pct = 100 * kp_missing.groupby("time.date").mean("time")
    daily_missing_pct.name = "Missing values (%)"
    return daily_missing_pct


def count_empty_frames_daily(
    da: xr.DataArray,
) -> pd.DataFrame:
    """Generate a daily report of empty frames.

    An empty frame is defined as a frame where all keypoints are missing (NaN).

    Parameters
    ----------
    da : xarray.DataArray
        The input data with ``time`` and ``space`` dimensions.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the daily counts of empty frames.

        The DataFrame has the following columns:
        - 'n_empty_frames': Number of empty frames per day.
        - 'n_total_frames': Total number of frames per day.
        - 'pct_empty_frames': Percentage of empty frames per day.

    """
    validate_dims_coords(da, {"time": [], "space": []})
    da = _squeeze_individuals_dim(da)

    # Boolean DataArray indicating when ALL keypoints are missing (NaN)
    all_kp_missing = da.isnull().any("space").all("keypoints")
    # Daily sum of frames with all keypoints missing, aka "empty frames"
    daily_empty_frames = all_kp_missing.groupby("time.date").sum("time")
    daily_total_frames = all_kp_missing.groupby("time.date").count("time")
    daily_pct_empty_frames = all_kp_missing.groupby("time.date").mean("time") * 100

    df = pd.DataFrame(
        {
            "n_empty_frames": daily_empty_frames.astype(int),
            "n_total_frames": daily_total_frames.astype(int),
            "pct_empty_frames": daily_pct_empty_frames.round(3),
        }
    )
    df.index = list(all_kp_missing.groupby("time.date").groups.keys())
    return df
