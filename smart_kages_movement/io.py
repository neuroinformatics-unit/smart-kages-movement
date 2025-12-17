"""Input/Output module for Smart Kages data handling.

We assume the following directory structure for the Smart-Kages data:

The data is stored per Smart-Kage, in folders names as `kageN`,
e.g. `kage1`, `kage2`, etc.

Each Smart-Kage folder contains:
- daily videos are stored in `videos/YYYY/MM/DD/`, split into 1-hour segments.
  Each 1-hour segment is an `.mp4` file named `kageN_YYYYMMDD_HHMMSS.mp4`
- corresponding DeepLabCut (DLC) predictions are stored in
  `analysis/dlc_output/YYYY/MM/DD/`. Each 1-hour `.h5` file therein is
  prefixed with `kageN_YYYYMMDD_HHMMSS`
- The `analysis/dlc_output/YYYY/MM/DD/` directory also contains a file
  named `corrected_timestamps.pkl`, storing a dictionary mapping
  each pose `.h5` file to an array of corrected timestamps, in units
  of seconds since the start of the hour.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sleap_io as sio


def parse_data_into_df(data_dir: Path) -> pd.DataFrame:
    """Parse the data directory into a pandas DataFrame.

    - First, discover all kage directories and their .h5 pose files
      output by DeepLabCut.
    - Then, construct the corresponding video file paths based on the
      datetime information in the pose file names.
    - Finally, create a DataFrame with the kage, date, hour, pose file path,
      video file path, and a boolean indicating if the video exists.

    The DataFrame will have a multi-index with
    levels `kage`, `date`, and `hour`.

    Parameters
    ----------
    data_dir : Path
        The path to the directory containing kage data. It must
        contain subdirectories named like "kageN" where N is
        an integer indicating the kage number.

    Returns
    -------
    pd.DataFrame
        A dataframe summarising all pose files
        and their corresponding video files.
    """

    kage_dirs = sorted(data_dir.glob("kage*"))
    print(
        f"Found {len(kage_dirs)} kage directories: ",
        *[d.name for d in kage_dirs],
    )

    pose_files = list(data_dir.glob("kage*/analysis/dlc_output/*/*/*/*.h5"))
    print(
        f"Found a total of {len(pose_files)} .h5 pose files output by DLC.",
    )

    list_of_dicts = []
    for pose_file in pose_files:  # Construct a dictionary for each pose file
        pose_dict: dict[str, Any] = {}
        parts = pose_file.stem.split("_")
        pose_dict["kage"] = parts[0]
        pose_dict["date"] = parts[1]
        start_time = parts[2].strip("DLC")
        pose_dict["hour"] = start_time[:2]
        pose_dict["start_datetime"] = pd.to_datetime(
            f"{pose_dict['date']} {start_time}",
            format="%Y%m%d %H%M%S",
        )
        pose_dict["pose_file_path"] = pose_file

        video_path = _get_video_path(
            data_dir / pose_dict["kage"], pose_dict["start_datetime"]
        )
        pose_dict["video_file_path"] = video_path

        list_of_dicts.append(pose_dict)

    # Convert the list of dictionaries to a DataFrame
    # Create DataFrame first
    df = pd.DataFrame(list_of_dicts)

    # Enforce str type for certain columns
    df = df.astype({"kage": str, "date": str, "hour": str})

    # Check that all video file paths are unique
    assert df["video_file_path"].is_unique, "Video file paths are not unique."

    # Sort the DataFrame by kage, date, and hour
    df = df.sort_values(by=["kage", "date", "hour"])
    # Set multi-index for easier access
    df.set_index(["kage", "date", "hour"], inplace=True)
    return df


def _get_video_path(kage_dir, datetime: pd.Timestamp) -> Path:
    """Get the video file path given a kage directory and a datetime."""
    kage = kage_dir.name
    year = datetime.strftime("%Y")
    month = datetime.strftime("%m")
    day = datetime.strftime("%d")
    date = datetime.strftime("%Y%m%d")
    time = datetime.strftime("%H%M%S")
    video_dir = kage_dir / "videos" / year / month / day
    assert video_dir.exists(), f"Directory {video_dir} does not exist."
    video_path = video_dir / f"{kage}_{date}_{time}.mp4"
    assert video_path.exists(), f"Video file {video_path} does not exist."

    return video_path


def get_video_shape(path: Path) -> tuple[int, int, int, int]:
    """Get the shape of a video without loading the entire file into memory.

    Parameters
    ----------
    path : Pat
        The path to the video file.

    Returns
    -------
    tuple[int, int, int, int]
        A tuple containing (n_frames, height, width, n_channels)
    """
    video = sio.load_video(path)
    try:
        shape = video.shape  # (frames, H, W[, C])
        if len(shape) == 3:
            return shape[0], shape[1], shape[2], 1
        else:
            return shape[0], shape[1], shape[2], shape[3]
    finally:
        video.close()


def load_background_frame(
    video_path: Path, i: int = 0, n_average: int = 100
) -> np.ndarray:
    """
    Load a specific frame or an average of several frames from a video.

    Parameters
    ----------
    video_path : Path
        The path to the video file from which to load the frame.
    i : int, optional
        The index of the frame to load. Default is 0 (meaning the first frame).
    n_average : int, optional
        The number of frames to average. The frames being averaged span the
        period from the ``i``-th frame to the ``i + n_average``-th frame.
        Default is 100. Set to 1 to load a single frame without averaging.

    Returns
    -------
    np.ndarray
        The loaded frame as a numpy array.
    """
    video = sio.load_video(video_path)

    # Ensure n_average does not exceed the number of frames
    n_frames = video.shape[0] - i
    n_average = min(n_average, n_frames)
    # Average the frames from i to i + n_average
    background_image = video[i : i + n_average].mean(axis=0).astype(np.uint8)
    return background_image


def save_segment_timestamps(
    segment: tuple[str, str, str], timestamps: pd.Series, timestamps_dir: Path
) -> tuple[tuple[str, str, str], Path]:
    """Save timestamps for a single (kage, date, hour) segment.

    Parameters
    ----------
    segment : tuple[str, str, str]
        A tuple containing (kage, date, hour) identifying the segment,
        e.g., ('kage1', '20230101', '01').
    timestamps : pd.Series
        A pandas Series containing the datetime timestamps for the segment.
    timestamps_dir : Path
        The directory where the timestamps file will be saved.
    """
    timestamps_dir = Path(timestamps_dir)
    (kage, date, hour) = segment
    iso_stamps = pd.to_datetime(timestamps).strftime("%Y-%m-%dT%H:%M:%S.%f")
    path = timestamps_dir / f"{kage}_{date}_{hour}_timestamps.txt"
    np.savetxt(path, iso_stamps, fmt="%s")
    return (kage, date, hour), path
