"""Functions for loading Smart-Kages data into movement datasets."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from movement.io import load_poses

from smart_kages_movement.io import (
    fix_dlc_h5_key,
    load_background_frame,
    load_segment_timestamps,
)


def _finalize_kage_dataset(
    ds_segments: list[xr.Dataset],
    kage: str,
    kage_start_datetime: pd.Timestamp,
) -> xr.Dataset:
    """Concatenate per-segment datasets and add kage-level metadata.

    Parameters
    ----------
    ds_segments : list of xr.Dataset
        Per-segment datasets to concatenate, in chronological order.
    kage : str
        The name of the kage, stored as a dataset attribute.
    kage_start_datetime : pd.Timestamp
        Datetime of the first frame across all segments, used to compute
        the ``seconds_elapsed`` secondary time coordinate.

    Returns
    -------
    xr.Dataset
        Concatenated dataset with ``kage`` and ``kage_start_datetime`` attrs
        and a ``seconds_elapsed`` coordinate alongside ``time``.

    Raises
    ------
    AssertionError
        If the concatenated time coordinate is not monotonically increasing.
    """
    ds_kage = xr.concat(ds_segments, dim="time")
    ds_kage.attrs["kage"] = kage
    ds_kage.attrs["kage_start_datetime"] = kage_start_datetime.isoformat()

    assert (ds_kage.time.values[1:] >= ds_kage.time.values[:-1]).all(), (
        f"Combined timestamps for {kage} are not monotonically increasing!"
    )

    seconds_since_kage_start = (
        ds_kage.time.data - np.datetime64(kage_start_datetime)
    ) / pd.Timedelta("1s")
    return ds_kage.assign_coords(seconds_elapsed=("time", seconds_since_kage_start))


def kage_to_movement_ds(
    df: pd.DataFrame,
    kage: str,
    fps: float,
) -> tuple[xr.Dataset, np.ndarray, pd.DataFrame | None]:
    """Load all poses for a given kage and return an xarray Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the paths to pose files as well as metadata
        for each 1-hour segment.
    kage : str
        The name of the kage to process, e.g., "kage1", "kage2", etc.
    fps : float
        Frames per second, used when loading DLC pose files.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the poses for the specified kage,
        with time coordinates assigned based on the corrected timestamps.
    np.ndarray
        A background image (numpy array) loaded from the middle segment
        of the kage, used for visualization purposes.
    df_problematic : pd.DataFrame or None
        A DataFrame indexed by (kage, date, hour) listing segments with a
        frame count mismatch, with columns ``n_frames`` (video),
        ``n_tracked_frames`` (DLC), and ``action`` ("truncated" or "skipped").
        None if all segments matched.

    Notes
    -----
    The returned Dataset will have two time coordinates:

    - ``time``: the primary time coordinate based on datetime timestamps.
    - ``seconds_elapsed``: seconds elapsed since the start of the kage.
    """
    print(f"Processing kage: {kage}")
    df_kage = df.loc[kage].sort_index()
    n_days = df_kage.index.get_level_values("date").nunique()
    print(f"Number of days: {n_days}")
    n_segments = df_kage.shape[0]
    print(f"Number of 1-hour segments: {n_segments}")

    kage_start_datetime = pd.Timestamp(df_kage["start_datetime"].iloc[0])

    ds_segments = []
    problematic_rows = {}
    previous_segment_end = kage_start_datetime
    gap_tolerance = pd.Timedelta("1s")

    for date, hour in df_kage.index:
        pose_file_path = Path(df_kage.loc[(date, hour), "pose_file_path"])

        # Fix DLC files that have 'data' key instead of 'df_with_missing'
        if fix_dlc_h5_key(pose_file_path):
            print(f"Fixed HDF5 key for {pose_file_path.name}")

        # Load the pose data for the current 1-hour segment
        poses = load_poses.from_file(
            pose_file_path,
            source_software="DeepLabCut",
            fps=fps,
        )

        n_frames = df_kage.loc[(date, hour), "n_frames"]
        n_tracked = poses.sizes["time"]

        if n_tracked != n_frames:
            mismatch_msg = (
                f"{kage} {date} {hour}: DLC has {n_tracked} tracked frames "
                f"but video has {n_frames} frames. "
            )
            if n_tracked > n_frames:
                warnings.warn(
                    mismatch_msg + f"Truncating to {n_frames}.",
                    UserWarning,
                    stacklevel=2,
                )
                poses = poses.isel(time=slice(None, n_frames))
                action = "truncated"
            else:
                warnings.warn(
                    mismatch_msg + "Skipping segment.",
                    UserWarning,
                    stacklevel=2,
                )
                problematic_rows[(kage, date, hour)] = {
                    "n_frames": n_frames,
                    "n_tracked_frames": n_tracked,
                    "action": "skipped",
                }
                continue

            problematic_rows[(kage, date, hour)] = {
                "n_frames": n_frames,
                "n_tracked_frames": n_tracked,
                "action": action,
            }

        # Load datetime timestamps and assign to poses
        timestamps = load_segment_timestamps(
            Path(df_kage.loc[(date, hour), "timestamps_file_path"])
        )
        poses = poses.assign_coords(time=timestamps)

        # Mark the first timepoint as NaN if there is a gap since the
        # previous segment (i.e. the recording was interrupted).
        if pd.Timestamp(timestamps[0]) - previous_segment_end > gap_tolerance:
            poses = poses.copy(deep=True)
            poses.loc[{"time": timestamps[0]}] = np.nan

        previous_segment_end = timestamps[-1]
        ds_segments.append(poses)

    n_skipped = sum(1 for r in problematic_rows.values() if r["action"] == "skipped")
    n_truncated = sum(
        1 for r in problematic_rows.values() if r["action"] == "truncated"
    )
    if n_skipped:
        print(f"Skipped {n_skipped} segments (too few tracked frames).")
    if n_truncated:
        print(f"Truncated {n_truncated} segments (too many tracked frames).")

    ds_kage = _finalize_kage_dataset(ds_segments, kage, kage_start_datetime)

    # Load a background image from the middle segment for visualisation
    video_path = df_kage.iloc[n_segments // 2]["video_file_path"]
    background_img = load_background_frame(video_path=video_path, i=0, n_average=100)
    print(f"Loaded background image for {kage} from {video_path} \n")

    if problematic_rows:
        df_problematic = pd.DataFrame.from_dict(problematic_rows, orient="index")
        df_problematic.index = pd.MultiIndex.from_tuples(
            df_problematic.index, names=["kage", "date", "hour"]
        )
    else:
        df_problematic = None

    return ds_kage, background_img, df_problematic
