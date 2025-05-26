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
"""

from pathlib import Path
from typing import Any

import pandas as pd


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
        pose_dict["video_exists"] = video_path.exists()
        pose_dict["video_file_path"] = video_path

        list_of_dicts.append(pose_dict)

    # Convert the list of dictionaries to a DataFrame
    # Create DataFrame first
    df = pd.DataFrame(list_of_dicts)

    # Enforce str type for certain columns
    df = df.astype({"kage": str, "date": str, "hour": str})

    # Check that all video file paths are unique
    assert df["video_file_path"].is_unique, "Video file paths are not unique."

    # Check how many pose files lack corresponding videos
    missing_videos = df["video_exists"].value_counts().get(False, 0)
    total_pose_files = len(df)
    if missing_videos > 0:
        print(
            f"Warning: {missing_videos}/{total_pose_files} "
            "pose files are missing corresponding videos."
        )

    # Sort the DataFrame by kage, date, and hour
    df = df.sort_values(by=["kage", "date", "hour"])
    # Set multi-index for easier access
    df.set_index(["kage", "date", "hour"], inplace=True)
    return df


def load_corrected_timestamps(data_dir):
    """
    Load corrected timestamps from all kage directories.

    The timestamps are stored per day, in a file named
    "corrected_timestamps.pkl". This file contains a dictionary mapping
    each pose .h5 file to an array of corrected timestamps.

    Parameters
    ----------
    data_dir : Path
        The path to the directory containing kage data.
        It must contain subdirectories named like "kageN" where N is
        an integer indicating the kage number.

    Returns
    -------
    dict
        A dictionary with corrected timestamps for each pose file.
        The keys are the pose file paths, and the values are arrays of
        corrected timestamps.

    """
    TS_FILENAME = "corrected_timestamps.pkl"
    timestamps = {}

    kage_date_dirs = sorted(data_dir.glob("kage*/analysis/dlc_output/*/*/*"))
    # exclude dirs whose last component is not a day of the form DD
    kage_date_dirs = [
        d for d in kage_date_dirs if d.name.isdigit() and len(d.name) == 2
    ]

    for kage_date_dir in kage_date_dirs:
        ts_file = kage_date_dir / TS_FILENAME
        if ts_file.exists():
            with ts_file.open("rb") as f:
                ts_data = pd.read_pickle(f)
                timestamps.update(ts_data)
        else:
            print(f"Warning: could not find {TS_FILENAME} in {kage_date_dir}")
    return timestamps


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

    return (
        kage_dir / "videos" / year / month / day / f"{kage}_{date}_{time}.mp4"
    )
