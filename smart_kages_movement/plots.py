from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


def plot_confidence_hist_by_keypoint(
    ds: xr.Dataset,
    save_path: Path | None = None,
) -> None:
    """
    Plot histograms of confidence values for each keypoint in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing keypoint positions and confidence values.
    save_path : Path | None
        Optional path to save the plot. If None, the plot will not be saved.
    """
    n_keypoints = ds.sizes["keypoints"]

    # Create subplots for each keypoint
    fig, axes = plt.subplots(
        nrows=2,
        ncols=(n_keypoints + 1) // 2,
        figsize=(n_keypoints * 1.5, n_keypoints * 0.75),
        sharey=True,
        sharex=True,
    )

    # Loop through each keypoint and plot its confidence histogram
    for i, kpt in enumerate(ds.keypoints.values):
        ax = axes[i % 2, i // 2]
        ds.confidence.sel(keypoints=kpt).plot.hist(
            bins=20,
            ax=ax,
            label=kpt,
            histtype="stepfilled",
            density=True,
        )
        ax.set_ylabel("Density")
        ax.set_xlabel("")
        ax.set_xlabel("Confidence")
        ax.set_title(kpt)

    plt.suptitle("Confidence Histograms by Keypoint")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=128)


def plot_speed(
    speed: xr.DataArray,
    save_path: Path | None = None,
) -> None:
    """
    Plot speed over time and a histogram of the speed values.

    Parameters
    ----------
    ds: xr.DataArray
        The data array containing body speed data.
    save_path: Path | None
        Optional path to save the plot. If None, the plot will not be saved.

    """
    fig, (ax, ax_hist) = plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [4, 1]}
    )

    speed.plot.line(x="time", lw=0.5, ax=ax)
    ax.set_title("Speed over time")
    ax.set_ylabel("Speed (cm/sec)")
    ax.set_xlabel("Datetime")
    ax.set_xlim(
        speed.time.min().values,
        speed.time.max().values,
    )
    # set x-ticks to every day
    plt.xticks(
        pd.date_range(
            start=speed.time.min().values,
            end=speed.time.max().values,
            freq="1D",
        ),
        rotation=90,
    )

    # Plot histogram of speed
    speed.plot.hist(bins=25, orientation="horizontal", ax=ax_hist)
    ax_hist.set_title("Histogram")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("log count")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=128)
