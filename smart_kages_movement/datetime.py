"""Utilities for working with datetime data"""

import json
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def extract_datetimes(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    """
    Extract the start datetimes and the frame timestamps for each video.

    The start times are stored per day, in a file named "adjustments.txt".
    This file contains a mapping between video filenames and their
    corrected start datetimes in the format "video_filename:H,M,S".
    We will read this file and adjust the start datetimes
    in the input dataframe accordingly. Invalid H,M,S values
    (e.g. negative numbers ) will raise a
    warning and be marked as NaN in the output.

    Additionally, we also have timestamps for each frame in the form of
    a file named "corrected_timestamps.pkl", stored in the same directory as
    the pose files. This file contains a dictionary mapping each pose .h5
    file to an array of corrected timestamps, in units of seconds since
    the start of the hour. The first element should be derived from the
    offset in the "adjustments.txt" file, and the rest should be derived
    by adding the frame timestamps (extracted from the .mp4 file) to this
    offset. If the first timestamp does not match the adjustment,
    a warning will be raised and the start_datetime will be set to NaT.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe summarising all pose files
        and their corresponding video files.

    Returns
    -------
    df : pd.DataFrame
        The input dataframe with the 'start_datetime' adjusted or set to NaT.
    frame_timestamps: dict[tuple[str, str, str], np.ndarray]
        A dictionary mapping (kage, date, hour) tuples to arrays of
        timestamps in seconds elapsed since the start of the video
        (difference between timestamp of current frame and first frame).
    """
    kage_date_pairs = df.index.droplevel("hour").unique()
    frame_timestamps = {}

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

        try:
            timestamps = _load_corrected_timestamps(timestamps_file)
        except (FileNotFoundError, EOFError) as e:
            timestamps = {}
            warnings.warn(
                f"Error loading timestamps file {timestamps_file}: {e}. "
                "Will attempt to extract frame timestamps from video file.",
                stacklevel=2,
            )

        midnight = pd.Timestamp(f"{date} 00:00:00")

        for hour in sub_df.index:
            # Extract start_datetime based on adjustments.txt
            video_filename = sub_df.loc[hour, "video_file_path"].name
            adjustment = adjustments.get(video_filename, np.nan)
            # adjustment is expressed in seconds since midnight
            df.loc[(kage, date, hour), "start_datetime"] = midnight + pd.to_timedelta(
                adjustment, unit="s"
            )
            pose_filename = sub_df.loc[hour, "pose_file_path"].name

            if pose_filename in timestamps:
                seconds_since_hour = timestamps[pose_filename]
            else:
                seconds_since_hour = extract_frame_timestamps(
                    sub_df.loc[hour, "video_file_path"]
                )
                print(
                    "Extracted frame timestamps from video file "
                    f"{sub_df.loc[hour, 'video_file_path']}"
                )

            first_timestamp = 3600 * int(hour) + seconds_since_hour[0]
            # If the first timestamp is not equal to the adjustment,
            # raise a warning and re-calculate the timestamps
            if first_timestamp != adjustment:  # seconds since midnight
                warnings.warn(
                    f"First timestamp for {pose_filename} does not match "
                    f"the adjustment for {video_filename}. Setting "
                    f"start_datetime to NaT.",
                    stacklevel=2,
                )
                # Set the start_datetime to NaT to flag it as problematic
                df.loc[(kage, date, hour), "start_datetime"] = pd.NaT

            # Express the timestamps as seconds since start of the video
            frame_timestamps[(kage, date, hour)] = (
                seconds_since_hour - seconds_since_hour[0]
            )

    return df, frame_timestamps


def find_segment_overlaps(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Find 1-hour segments that overlap in time.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe summarising all pose files
        and their corresponding video files.
        We assume a multi-index with levels 'kage', 'date', and 'hour',
        and the existence of 'start_datetime' and 'end_datetime' columns.

    Returns
    -------
    pd.DataFrame | None
        A dataframe containing pairs of segments that overlap,
        with columns: 'segment_A', 'segment_B', 'end_A', 'start_B',
        and 'overlap_duration'. If no overlaps are found, returns None.

    """

    start = "start_datetime"
    end = "end_datetime"

    # Build an IntervalIndex per group
    results = []  # will collect tuples (iloc_i, iloc_j)
    for _, group in df.groupby(["kage", "date"]):
        iv = pd.IntervalIndex.from_arrays(group[start], group[end], closed="both")

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
                            "start_A": group.at[iloc_i, start],
                            "end_A": group.at[iloc_i, end],
                            "start_B": group.at[iloc_j, start],
                            "end_B": group.at[iloc_j, end],
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
                "start_A",
                "end_A",
                "start_B",
                "end_B",
            ],
        )
        overlaps["overlap_duration_seconds"] = (
            overlaps["end_A"] - overlaps["start_B"]
        ).dt.total_seconds()
        return overlaps


def extract_frame_timestamps(
    video_path: Path,
    expected_n_frames: int | None = None,
) -> np.ndarray:
    """Extract timestamps of video frames using ffprobe.

    If there are frames without timestamps, they will be filled with
    linear interpolation.

    Parameters
    ----------
    video_path : Path
        Path to the video file.
    expected_n_frames : int | None, optional
        If provided, this is the expected number of frames in the video, which
        we may know from another source (e.g. sleap_io). If None (default),
        the function will count the total number of frames in the video using
        ffprobe.

    Returns
    -------
    np.ndarray
        An array of timestamps in seconds for each frame in the video.
        The timestamps are expressed as seconds since the start of the video.

    Notes
    -----
    This function relies on the "best_effort_timestamp_time" field
    from ffprobe's output.
    """

    if shutil.which("ffprobe") is None:
        raise OSError("ffprobe not found. Please install FFmpeg (includes ffprobe).")

    cmd = [
        "ffprobe",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "json",
        video_path.as_posix(),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(  # noqa: B904
            f"ffprobe failed to extract timestamps from '{video_path}'.\n"
            f"Error message:\n{e.stderr}"
        )

    data = json.loads(result.stdout)

    frames = data.get("frames", [])
    n_frames = expected_n_frames or count_total_frames(video_path)

    timestamps = np.full(n_frames, np.nan, dtype=np.float32)  # Init with NaNs

    for i, frame in enumerate(frames):
        ts = frame.get("best_effort_timestamp_time")
        if ts is not None:
            timestamps[i] = float(ts)

    n_frames_missing_ts = np.count_nonzero(np.isnan(timestamps))

    # Interpolate missing timestamps
    if n_frames_missing_ts > 0:
        warnings.warn(
            f"Video {video_path} has {n_frames_missing_ts} missing timestamps."
            " The following frames will be filled with linear interpolation: "
            f"{np.flatnonzero(np.isnan(timestamps)).tolist()}",
            stacklevel=2,
        )
        timestamps = _interpolate_timestamps(timestamps)

    return timestamps


def count_total_frames(video_path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        video_path.as_posix(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


def _interpolate_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate missing frame timestamps.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of frame timestamps with NaNs for missing values.

    Returns
    -------
    np.ndarray
        Array with missing timestamps filled by linear interpolation.

    """
    n_frames = len(timestamps)
    valid = ~np.isnan(timestamps)
    x_valid = np.flatnonzero(valid)
    y_valid = timestamps[valid]

    interpolator = interp1d(
        x_valid,
        y_valid,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interpolator(np.arange(n_frames))


def _load_adjustments_file(
    adjustments_file: Path,
) -> dict[str, float]:
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
        in seconds since midnight. If an adjustment is invalid,
        it will be set to NaN and a warning will be emitted.
    """
    adjustments = {}
    with adjustments_file.open("r") as f:
        for line in f:
            video_filename, time_str = line.strip().split(":")
            # if video_filename in a path, extract the filename only
            video_filename = Path(video_filename).name
            hours, minutes, seconds = map(int, time_str.split(","))
            adjustment = (hours, minutes, seconds)
            if not _adjustment_is_valid(adjustment):
                warnings.warn(
                    f"Invalid adjustment {adjustment} for video "
                    f"{video_filename}. It should be in the format H,M,S "
                    "where H is 0-23, M is 0-59, and S is 0-59. "
                    "Setting to NaN.",
                    stacklevel=2,
                )
                adjustments[video_filename] = np.nan
            else:
                seconds_since_midnight = 3600 * hours + 60 * minutes + seconds
                adjustments[video_filename] = seconds_since_midnight
    return adjustments


def _adjustment_is_valid(adjustment: tuple[int, int, int]) -> bool:
    """Returns True if the adjustment is valid.

    Verifies that hours, minutes, and seconds are within valid ranges.
    """
    in_valid_range = (
        0 <= adjustment[0] < 24 and 0 <= adjustment[1] < 60 and 0 <= adjustment[2] < 60
    )
    return in_valid_range


def _load_corrected_timestamps(file_path: Path) -> dict[str, np.ndarray]:
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
        A dictionary mapping pose file names to arrays of corrected timestamps.

    """
    with file_path.open("rb") as f:
        timestamps = pd.read_pickle(f)
    # In case the dict key is a path, take only the name of the file
    timestamps = {Path(k).name: np.array(v) for k, v in timestamps.items()}
    return timestamps
