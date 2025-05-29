from pathlib import Path

import matplotlib.pyplot as plt
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
