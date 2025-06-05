"""Utilities for working with datetime data"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def adjust_start_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive correct start datetimes for each video.

    The timestamps are stored per day, in a file named "adjustments.txt".
    This file contains a mapping between video filenames and their
    corrected start datetimes in the format `video_filename:H,M,S`.
    We will read this file and adjust the start datetimes
    in the input dataframe accordingly.

    We have an additional source of corrected timestamps in the form of
    a file named "corrected_timestamps.pkl" in the same directory as the
    pose files. This file contains a dictionary mapping each pose `.h5`
    file to an array of corrected timestamps, in units of seconds since
    the start of the hour. We will use this to derive an alternative
    start datetime and an end datetime for each pose file (based on
    the first and last timestamps in the array, respectively).

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe summarising all pose files
        and their corresponding video files.

    Returns
    -------
    df : pd.DataFrame
        The input dataframe with the 'start_datetime' column updated
        to reflect the correct start times of each video.

    """
    kage_date_pairs = df.index.droplevel("hour").unique()
    for kage, date in kage_date_pairs:
        sub_df = df.loc[kage, date]
        video_dir = sub_df["video_file_path"].iloc[0].parent
        dlc_dir = sub_df["pose_file_path"].iloc[0].parent
        adjustments_file = video_dir / "adjustments.txt"
        timestamps_file = dlc_dir / "corrected_timestamps.pkl"

        if adjustments_file.exists():
            adjustments = _load_adjustments_file(adjustments_file)
        else:
            raise FileNotFoundError(
                f"Adjustments file {adjustments_file} does not exist."
            )

        if timestamps_file.exists():
            timestamps_pkl = _load_corrected_timestamps(timestamps_file)
        else:
            raise FileNotFoundError(
                f"Timestamps file {timestamps_file} does not exist."
            )

        midnight = pd.Timestamp(f"{date} 00:00:00")

        for hour in sub_df.index:
            # Extract start_datetime based on adjustments.txt
            video_filename = sub_df.loc[hour, "video_file_path"].name
            if video_filename in adjustments:
                hours, minutes, seconds = adjustments[video_filename]
                # Convert to seconds (to also handle negative values)
                seconds_since_midnight = (
                    3600 * hours + 60 * minutes + seconds
                )
                # Calculate the adjusted start datetime
                df.loc[(kage, date, hour), "start_datetime"] = (
                    midnight
                    + pd.Timedelta(seconds=seconds_since_midnight)
                )
            else:
                raise KeyError(
                    f"Video {video_filename} not found in adjustments file."
                )

            # Extract start and end datetimes based on corrected_timestamps.pkl
            pose_filename = sub_df.loc[hour, "pose_file_path"].name
            if pose_filename in timestamps_pkl:
                seconds_since_hour = timestamps_pkl[pose_filename]
                # This can also be negative, which means before the hour
                seconds_since_midnight = np.array(seconds_since_hour) + (
                    3600 * int(hour)  # Convert hour to seconds
                )
                # Add an alternative start datetime based on first timestamp
                df.loc[(kage, date, hour), "start_datetime_pkl"] = (
                    midnight
                    + pd.Timedelta(seconds=seconds_since_midnight[0])
                )
                # Extract end datetime based on last timestamp
                df.loc[(kage, date, hour), "end_datetime_pkl"] = (
                    midnight
                    + pd.Timedelta(seconds=seconds_since_midnight[-1])
                )
            else:
                raise KeyError(
                    f"Pose file {pose_filename} not found in timestamps file."
                )

    return df


def find_datetime_diffs(df, threshold=0.5, kind="start", plot_hist=True):
    """
    Find rows where the difference between the two datetime sources
    exceeds a certain threshold (in seconds).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns "{kind}_datetime" and "{kind}_datetime_pkl".
    threshold : float, optional
        Threshold in seconds for the difference to be considered significant.
        Default is 0.5 seconds.
    kind : {"start", "end"}
        Which datetime to compare.
    plot_hist : bool, optional
        If True, plot a histogram of the differences.
        The y-axis will be on a log scale to help visualise
        any outliers.

    Returns
    -------
    pd.DataFrame
        Subset of df with columns
        [<kind>_datetime, <kind>_datetime_pkl, <kind>_diff_seconds].
    """
    if kind not in {"start", "end"}:
        raise ValueError("kind must be 'start' or 'end'")
    dt_col = f"{kind}_datetime"
    pkl_col = f"{kind}_datetime_pkl"
    diff_col = f"{kind}_diff_seconds"

    # compute diff (pkl - original) in seconds
    df[diff_col] = (df[pkl_col] - df[dt_col]).dt.total_seconds()

    # filter where abs(diff) > threshold
    mask = df[diff_col].abs() > threshold
    result = df.loc[mask, [dt_col, pkl_col, diff_col]]

    print(
        f"Found {len(result)} segments with "
        f"{kind}-datetime differences > {threshold:.3f} sec."
    )

    if plot_hist:
        df.plot.hist(
            y=f"{kind}_diff_seconds",
            bins=50,
            title=f"Histogram of {kind} datetime differences (seconds)",
            xlabel="corrected_timestamps.pkl - adjustments.txt (seconds)",
            ylabel="N segments",
            grid=True,
        )
        plt.yscale("log")
    return result


def find_segment_overlaps(
        df: pd.DataFrame, use_pkl: bool = False
) -> pd.DataFrame | None:
    """
    Find 1-hour segments that overlap in time.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe summarising all pose files
        and their corresponding video files.
        We assume a multi-index with levels 'kage', 'date', and 'hour',
        and the existence of 'start_datetime' and 'end_datetime' columns.
    use_pkl : bool, optional
        If True, use the 'start_datetime_pkl' and 'end_datetime_pkl'
        columns instead of 'start_datetime' and 'end_datetime'.
        Default is False.

    Returns
    -------
    pd.DataFrame | None
        A dataframe containing pairs of segments that overlap,
        with columns: 'segment_A', 'segment_B', 'end_A', 'start_B',
        and 'overlap_duration'. If no overlaps are found, returns None.

    """

    start = "start_datetime_pkl" if use_pkl else "start_datetime"
    end = "end_datetime_pkl" if use_pkl else "end_datetime"

    # Build an IntervalIndex per group
    results = []  # will collect tuples (iloc_i, iloc_j)
    for _, group in df.groupby(["kage", "date"]):
        iv = pd.IntervalIndex.from_arrays(
            group[start], group[end], closed="both"
        )

        # for each row in the group, find which intervals overlap
        for iloc_i, interval in zip(group.index, iv, strict=True):
            hits = group.index[iv.overlaps(interval)]
            # drop the row itself
            hits = hits[hits != iloc_i]
            for iloc_j in hits:
                # to avoid duplicates you could enforce iloc_i < iloc_j
                if iloc_i < iloc_j:
                    results.append(
                        {
                            "segment_A": iloc_i,
                            "segment_B": iloc_j,
                            # grab end of A, start of B from the group
                            "end_A": group.at[iloc_i, end],
                            "start_B": group.at[iloc_j, start],
                        }
                    )

    if not results:
        print("No overlapping segments found.")
        return None
    else:
        print(f"Found {len(results)} overlapping segments.")
        overlaps = pd.DataFrame(
            results,
            columns=[
                "segment_A",
                "segment_B",
                "end_A",
                "start_B",
            ],
        )
        overlaps["overlap_duration_seconds"] = (
            (overlaps["end_A"] - overlaps["start_B"])
            .dt.total_seconds()
        )
        return overlaps


def _load_adjustments_file(
    adjustments_file: Path,
) -> dict[str, tuple[int, int, int]]:
    """
    Load adjustments from a file into a dictionary.

    Parameters
    ----------
    adjustments_file : Path
        Path to the adjustments file.

    Returns
    -------
    dict
        A dictionary mapping video filenames to their time adjustments
        in the format (hours, minutes, seconds).
    """
    adjustments = {}
    with adjustments_file.open("r") as f:
        for line in f:
            video_filename, time_str = line.strip().split(":")
            # if video_filename in a path, extract the filename only
            video_filename = Path(video_filename).name
            hours, minutes, seconds = map(int, time_str.split(","))
            adjustments[video_filename] = (hours, minutes, seconds)
    return adjustments


def _load_corrected_timestamps(file_path: Path) -> dict[Path, np.ndarray]:
    """
    Load timestamps from a single "corrected_timestamps.pkl" file.

    This file contains a dictionary mapping
    each pose .h5 file to an array of corrected timestamps
    in seconds since the start of the hour.

    Parameters
    ----------
    file_path : Path
        The path to the "corrected_timestamps.pkl" file.

    Returns
    -------
    dict
        A dictionary pose file paths to arrays of
        corrected timestamps for the corresponding pose file.

    """
    with file_path.open("rb") as f:
        timestamps = pd.read_pickle(f)
    # In case the dict key is a path, take only the name of the file
    timestamps = {Path(k).name: v for k, v in timestamps.items()}
    return timestamps
