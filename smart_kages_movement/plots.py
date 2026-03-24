from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from movement.plots import plot_occupancy
from sleap_io import load_video

from smart_kages_movement.datetime import hhmm_to_minutes
from smart_kages_movement.reports import (
    count_empty_frames_daily,
    count_missing_keypoints_daily,
)

CMAP = "Set2"

DEFAULT_SCATTER_ARGS = {
    "s": 15,
    "marker": "o",
    "alpha": 1.0,
}

_HOUR_TICKS = np.arange(0, 24 * 60 + 1, 4 * 60)
_HOUR_LABELS = [f"{int(m // 60):02d}:00" for m in _HOUR_TICKS]


def show_first_frame_corner(video_path: Path, crop_height=20, crop_width=300):
    """Display the first frame of the video.

    Only the top-left corner of the frame is shown, cropped to
    `crop_height` x `crop_width` pixels.
    """
    video = load_video(video_path)
    try:
        frame = video[0]
        plt.imshow(frame[:crop_height, :crop_width])
        plt.axis("off")
        plt.show()
    finally:
        video.close()


def plot_missing_keypoints_heatmap(
    da: xr.DataArray,
    kage_name: str,
    title_str: str = "daily % missing keypoints",
    save_path: Path | None = None,
    cmap: str = CMAP,
) -> None:
    """Plot heatmap of daily % missing keypoints.

    Parameters
    ----------
    da : xr.DataArray
        The data array containing keypoint positions.
    kage_name : str
        The name of the kage, to be used in the plot title.
    title_str : str
        The title string to be used in the plot title, after the kage name.
        Default is "daily % missing keypoints".
    save_path : Path | None
        Optional path to save the plot. If None, the plot will not be saved.
        If provided, a CSV file with the underlying data will also be saved
        alongside the plot.
    cmap : str
        The colormap to use for the heatmap. Any of the qualitative matplotlib
        colormaps can be used.
    """
    missing_kpt = count_missing_keypoints_daily(da)

    # Assign strings as "date" coords
    days = [day.strftime("%Y-%m-%d") for day in list(missing_kpt.date.values)]
    missing_kpt = missing_kpt.assign_coords(date=days)

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    plot_params = {
        "vmin": 0,
        "vmax": 100,
        "cmap": cmap,
    }
    missing_kpt.T.plot.imshow(ax=ax, **plot_params)

    ax.set_title(f"{kage_name}: {title_str}")
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=90)

    if save_path:
        plt.savefig(save_path, dpi=128)
        missing_kpt.to_pandas().to_csv(save_path.with_suffix(".csv"))


def plot_daily_qc(
    ds: xr.Dataset,
    keypoint: str,
    save_path: Path | None = None,
    xlim: tuple[str, str] | None = None,
) -> None:
    """Generate plots with some daily QC metrics.

    The purpose of these plots is to help us identify suitable days for
    analysis within a kage.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing keypoint positions and confidence values.
    keypoint : str
        The name of the keypoint, e.g., 'bodycenter'.
    save_path : Path | None
        Optional path to save the plot. If None, the plot will not be saved.
        If provided, a CSV file with the underlying data will also be saved
        alongside the plot.
    xlim : tuple[str, str] | None
        Optional x-axis limits as (start_date, end_date) in 'YYYY-MM-DD'
        format. If None, the x-axis limits will be determined automatically.

    """
    kage_name = ds.attrs.get("kage", "kageX")

    df = count_empty_frames_daily(ds.position)

    fig, axes = plt.subplots(
        nrows=2, figsize=(12, 6), sharex=True, layout="constrained"
    )
    plt.suptitle(f"{kage_name}: daily QC metrics")

    # Plot number of expected, total and empty frames per day
    axes[0].axhline(
        int(24 * 60 * 60 * ds.attrs["fps"]),
        color="gray",
        ls="--",
        label="Expected",
    )
    df["n_total_frames"].plot.line(marker="o", label="Total", ax=axes[0])
    df["n_empty_frames"].plot.line(marker="o", label="Empty", ax=axes[0])
    axes[0].set_title("Number of frames")
    axes[0].set_xlabel("Date")
    axes[0].set_xticks(df.index)
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].legend()
    axes[0].grid()

    # Plot median confidence per day for a given keypoint
    kpt_conf = ds.confidence.squeeze().sel(keypoints=keypoint)
    daily_median_conf = kpt_conf.groupby("time.date").median().to_pandas()
    daily_median_conf.plot.line(marker="o", color="k", ax=axes[1])

    axes[1].set_title(f"Median {keypoint} confidence")
    axes[1].set_xlabel("Date")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_xticks(daily_median_conf.index)
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].grid()

    if xlim:
        plt.xlim(*xlim)

    if save_path:
        plt.savefig(save_path, dpi=128)
        df[f"median_{keypoint}_confidence"] = daily_median_conf
        df.to_csv(save_path.with_suffix(".csv"))


def plot_confidence_quartiles_per_keypoint(
    ds: xr.Dataset,
    save_path: Path | None = None,
) -> None:
    """
    Plot confidence quartiles per keypoint.
    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing keypoint positions and confidence values.
    save_path : Path | None
        Optional path to save the plot. If None, the plot will not be saved.
        If provided, a CSV file with the underlying data will also be saved
        alongside the plot.
    """
    confidence = ds.confidence.squeeze()
    kage = ds.attrs.get("kage", "kageX")

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    Q1 = confidence.quantile(q=0.25, dim="time")
    median = confidence.median(dim="time")
    Q3 = confidence.quantile(q=0.75, dim="time")
    keypoints = confidence.keypoints.values

    Q1.plot.line("--", color="gray", ax=ax)
    Q3.plot.line("--", color="gray", ax=ax)
    median.plot.line("o-", color="black", ax=ax, label="Median (50%)")
    ax.fill_between(
        keypoints, Q1, Q3, color="lightgray", alpha=0.5, label="IQR (25-75%)"
    )

    ax.legend()
    ax.set_title(f"{kage}: confidence quartiles per keypoint")

    if save_path:
        plt.savefig(save_path, dpi=128)

        # Put the quartile data into a DataFrame and save as CSV
        df_quartiles = pd.DataFrame(
            {
                "Q1": Q1.values,
                "Median": median.values,
                "Q3": Q3.values,
            },
            index=keypoints,
        )
        df_quartiles.to_csv(save_path.with_suffix(".csv"))


def plot_confidence_hist_per_keypoint(
    ds: xr.Dataset,
    save_path: Path | None = None,
    cmap: str = CMAP,
    layout: Literal["subplots", "overlay"] = "subplots",
) -> None:
    """
    Plot histograms of confidence values for each keypoint in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing keypoint positions and confidence values.
    save_path : Path | None
        Optional path to save the plot. If None, the plot will not be saved.
    cmap : str
        The colormap to use for the histograms.
        Any of the qualitative matplotlib colormaps can be used.
    layout : str
        The plot layout to use. Options are 'subplots' (each keypoint
        in its own subplot) or 'overlay' (all keypoints in a single plot).
        Default is 'subplots'.
    """
    n_keypoints = ds.sizes["keypoints"]

    if "kage" in ds.attrs and ds.attrs["kage"]:
        title = f"{ds.attrs['kage']}: confidence histograms per keypoint"
    else:
        title = "Confidence histograms per keypoint"

    colors = plt.colormaps[cmap].colors

    if layout == "subplots":
        # Create subplots for each keypoint
        fig, axes = plt.subplots(
            nrows=2,
            ncols=(n_keypoints + 1) // 2,
            figsize=(n_keypoints * 1.5, n_keypoints * 0.75),
            sharey=True,
            sharex=True,
            layout="constrained",
        )

        # Loop through each keypoint and plot its confidence histogram
        for i, kpt in enumerate(ds.keypoints.values):
            color_i = colors[i % len(colors)]
            ax = axes[i % 2, i // 2]
            ds.confidence.sel(keypoints=kpt).plot.hist(
                bins=20,
                ax=ax,
                label=kpt,
                histtype="stepfilled",
                density=True,
                color=color_i,
            )
            ax.set_title(kpt, color=color_i)
            # Set ylabel only for the left plots (first column)
            if i // 2 == 0:
                ax.set_ylabel("Density")
            else:
                ax.set_ylabel("")
            # Set xlabel only for the bottom plots (last row)
            if i % 2 == 1:
                ax.set_xlabel("Confidence")
            else:
                ax.set_xlabel("")

        plt.suptitle(title)

    elif layout == "overlay":
        fig, ax = plt.subplots(layout="constrained")
        for i, kpt in enumerate(ds.keypoints.values):
            ds.confidence.sel(keypoints=kpt).plot.hist(
                bins=20,
                histtype="step",
                density=True,
                ax=ax,
                color=colors[i % len(colors)],
                label=kpt,
            )
        ax.set_ylabel("Density")
        ax.set_xlabel("Confidence")
        ax.set_title(title)
        plt.legend()

    else:
        raise ValueError("Invalid layout option. Choose 'subplots' or 'overlay'.")

    if save_path:
        plt.savefig(save_path, dpi=128)


def plot_speed(
    speed: xr.DataArray,
    dark_period: tuple[str, str] | None = None,
    save_path: Path | None = None,
) -> Figure:
    """
    Plot speed over time and a histogram of the speed values.

    Parameters
    ----------
    ds: xr.DataArray
        The data array containing body speed data.
    dark_period: tuple[str, str] | None
        Optional tuple specifying the start and end of the dark period, which
        will be shaded on plot. The tuple has to be of the format
        (start_time, end_time), e.g., ('09:30', '20:30').
        Default is None, meaning no shading is applied.
    save_path: Path | None
        Optional path to save the plot. If None, the plot will not be saved.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The figure containing the two subplots (speed over time and histogram).

    """
    fig, (ax, ax_hist) = plt.subplots(
        1,
        2,
        figsize=(12, 4),
        gridspec_kw={"width_ratios": [4, 1]},
        layout="constrained",
    )

    # Apply shading for dark periods if specified
    if dark_period:
        start_time, end_time = dark_period
        # Shade dark periods on the speed over time plot
        current_date = pd.to_datetime(str(speed.time.dt.date.min().values))
        end_date = pd.to_datetime(str(speed.time.dt.date.max().values))
        while current_date <= end_date:
            on = pd.to_datetime(f"{current_date.date()} {start_time}")
            off = pd.to_datetime(f"{current_date.date()} {end_time}")
            ax.axvspan(on, off, facecolor="gray", edgecolor=None, alpha=0.3)
            current_date += pd.Timedelta(days=1)

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
    )

    # Plot histogram of speed
    speed.plot.hist(bins=25, orientation="horizontal", ax=ax_hist)
    ax_hist.set_title("Histogram")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("log count")

    if save_path:
        plt.savefig(save_path, dpi=128)

    return fig


def plot_trajectory(
    position: xr.DataArray,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot trajectory of a single keypoint.

    This function plots the trajectory of a single point's ``position``,
    by default colored by time (using the default colormap). Pass a different
    variable through ``c`` in ``kwargs`` if desired.

    Parameters
    ----------
    position : xr.DataArray
        A data array containing position information, with `time` and `space`
        as the only dimensions.
    ax : matplotlib.axes.Axes or None, optional
        Axes object on which to draw the trajectory. If None, a new
        figure and axes are created.
    **kwargs : dict
        Additional keyword arguments passed to
        :meth:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    (figure, axes) : tuple of (matplotlib.pyplot.Figure, matplotlib.axes.Axes)
        The figure and axes containing the trajectory plot.

    """
    # Squeeze out any lingering singleton dimensions
    position = position.squeeze()

    fig, ax = (
        plt.subplots(figsize=(6, 6), layout="constrained")
        if ax is None
        else (ax.figure, ax)
    )

    # Merge default plotting args with user-provided kwargs
    for key, value in DEFAULT_SCATTER_ARGS.items():
        kwargs.setdefault(key, value)

    if "c" not in kwargs:
        kwargs["c"] = position.time

    # Plot the scatter, colouring by time or user-provided colour
    sc = ax.scatter(
        position.sel(space="x"),
        position.sel(space="y"),
        **kwargs,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add 'colorbar' for time dimension if no colour was provided by user
    cbar_label = getattr(kwargs["c"], "name", "time")
    if "units" in kwargs["c"].attrs:
        cbar_label += f" [{kwargs['c'].units}]"
    fig.colorbar(sc, ax=ax, label=cbar_label).solids.set(alpha=1.0)

    return fig, ax


def _make_grouped_subplots(
    da: xr.DataArray,
    group_by: str,
    figsize: tuple[float, float],
    rows_per_individual: int = 1,
) -> tuple[Figure, np.ndarray, np.ndarray]:
    """Create a figure with one subplot per unique value of ``group_by``.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with an ``individuals`` dimension and ``group_by`` as a
        non-dimension coordinate.
    group_by : str
        Name of the coordinate to group by.
    figsize : tuple[float, float]
        Figure size as ``(width, height)``.
    rows_per_individual : int
        Height weight each individual contributes to its subplot. Use
        ``n_days`` for actogram plots where each individual occupies
        multiple rows. Default is ``1``.

    Returns
    -------
    fig : Figure
    axes : np.ndarray of Axes, shape ``(n_groups,)``
    unique_groups : np.ndarray of unique group values
    """
    if group_by not in da.coords:
        raise ValueError(f"'{group_by}' is not a coordinate of the DataArray.")
    unique_groups = np.unique(da.coords[group_by].values)
    height_ratios = [
        int((da.coords[group_by] == g).sum()) * rows_per_individual
        for g in unique_groups
    ]
    fig, axes = plt.subplots(
        nrows=len(unique_groups),
        figsize=figsize,
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": height_ratios},
    )
    return fig, np.atleast_1d(axes), unique_groups


def _cbar_label(da: xr.DataArray) -> str:
    """Return a colorbar label from a DataArray's attributes or name."""
    return da.attrs.get("long_name", da.name or "")


def _finalize_grouped_xaxis(
    axes: np.ndarray,
    ticks,
    labels: Sequence[str | int],
    xlabel: str,
) -> None:
    """Hide x-tick labels on all but the bottom axes and set ticks/label."""
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel(xlabel)


def plot_activity_heatmap(
    activity: xr.DataArray,
    save_path: Path | None = None,
    cmap: str = CMAP,
    group_by: str | None = None,
) -> None:
    """Plot heatmap of activity over a week.

    Parameters
    ----------
    activity : xr.DataArray
        The data array containing activity levels, with dimensions
        'time' and `individuals`.
    save_path : Path | None
        Optional path to save the plot. If None, the plot will not be saved.
        If provided, a CSV file with the underlying data will also be saved
        alongside the plot.
    cmap : str
        The colormap to use for the heatmap. Any of the qualitative matplotlib
        colormaps can be used.
    group_by : str | None
        Name of a non-dimension coordinate of the 'individuals' dimension.
        If provided, one subplot per unique group value is created, each
        showing only the individuals belonging to that group. Subplot heights
        are proportional to group size. Default is None.
    """
    # If there is a singleton keypoint dimension, squeeze it out
    if "keypoints" in activity.dims and activity.sizes["keypoints"] == 1:
        activity = activity.squeeze("keypoints")

    activity_vmax = float(activity.quantile(0.99))  # use 99th percentile

    week_start = activity.time.min().item()
    week_end = activity.time.max().item()
    daily_ticks = pd.date_range(start=week_start, end=week_end, freq="D")
    daily_tick_labels = [
        int((pd.Timestamp(tick) - pd.Timestamp(week_start)).days)
        for tick in daily_ticks
    ]

    pcolormesh_kwargs = dict(
        x="time", y="individuals", cmap=cmap, vmin=0, vmax=activity_vmax
    )

    if group_by is not None:
        fig, axes, unique_groups = _make_grouped_subplots(
            activity, group_by, figsize=(12, 8)
        )
        pm = None
        for ax, group in zip(axes, unique_groups, strict=True):
            mask = activity.coords[group_by].values == group
            group_da = activity.isel(individuals=mask)
            pm = group_da.plot.pcolormesh(
                **pcolormesh_kwargs, ax=ax, add_colorbar=False
            )
            ax.set_title(f"{group_by} = {group}")
            ax.set_ylim(-0.5, len(group_da.individuals) - 0.5)

        _finalize_grouped_xaxis(axes, daily_ticks, daily_tick_labels, "Days")
        fig.colorbar(pm, ax=axes.tolist(), label=_cbar_label(activity))

    else:
        fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
        activity.plot.pcolormesh(**pcolormesh_kwargs, ax=ax)
        ax.set_xticks(daily_ticks)
        ax.set_xticklabels(daily_tick_labels)
        ax.set_xlabel("Days")
        ax.set_ylim(-0.5, len(activity.individuals) - 0.5)

    plt.suptitle("Activity heatmap")

    if save_path:
        plt.savefig(save_path, dpi=128)
        activity.to_pandas().to_csv(save_path.with_suffix(".csv"))


def _add_dark_period_bar(
    ax: Axes, dark_period: tuple[str, str], bar_height: float = 0.05
) -> None:
    """Add a dark period indicator bar as an inset axis above ax.

    Parameters
    ----------
    ax : Axes
        The axes above which the bar will be drawn.
    dark_period : tuple[str, str]
        Start and end of the dark period as ``('HH:MM', 'HH:MM')``.
    bar_height : float
        Height of the bar as a fraction of the axes height. Default is 0.05.
    """
    dark_start = hhmm_to_minutes(dark_period[0])
    dark_end = hhmm_to_minutes(dark_period[1])
    bar_ax = ax.inset_axes(
        [0, 1 + bar_height, 1, bar_height],
        sharex=ax,
        sharey=None,
        transform=ax.transAxes,
    )
    bar_ax.set_ylim(0, 1)
    bar_ax.axis("off")
    bar_ax.axvspan(dark_start, dark_end, facecolor="0.1", edgecolor="none")


def _stack_actogram(actogram: xr.DataArray) -> xr.DataArray:
    """Stack a 3D actogram into a tall 2D array for cohort plotting.

    Parameters
    ----------
    actogram : xr.DataArray
        3D DataArray with dimensions ``(individuals, day, minutes)``.

    Returns
    -------
    xr.DataArray
        2D DataArray with dimensions ``(minutes, individuals_days)`` and a
        ``kage_day_id`` string coordinate for y-axis labelling.
    """
    actogram_tall = actogram.stack(individuals_days=("individuals", "day"))
    kage_day_labels = [
        f"{ind}_day{day}" for ind, day in actogram_tall.individuals_days.values
    ]
    return actogram_tall.assign_coords(
        kage_day_id=("individuals_days", kage_day_labels)
    )


def _set_cohort_yticks(ax: Axes, individuals: np.ndarray, n_days: int) -> None:
    """Set y-ticks at individual start rows for a stacked actogram axes."""
    tick_positions = [i * n_days for i in range(len(individuals))]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(individuals)
    ax.set_ylabel("")


def plot_actogram(
    actogram: xr.DataArray,
    dark_period: tuple[str, str],
    title: str = "",
    save_path: Path | None = None,
    cmap: str = "viridis",
    vmax: float | None = None,
) -> Figure:
    """Plot an actogram for a single individual.

    Parameters
    ----------
    actogram : xr.DataArray
        2D DataArray with dimensions ``(day, minutes)``.
    dark_period : tuple[str, str]
        Start and end of the dark period as ``('HH:MM', 'HH:MM')``.
    title : str
        Plot title. Default is ``""``.
    save_path : Path | None
        Optional path to save the figure.
    cmap : str
        Colormap. Default is ``"viridis"``.
    vmax : float | None
        Maximum value for the color scale. If ``None``, the 99th percentile
        of the data is used.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if vmax is None:
        vmax = float(actogram.quantile(0.99))

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

    actogram.plot.pcolormesh(
        x="minutes",
        y="day",
        ax=ax,
        yincrease=False,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        cbar_kwargs={"fraction": 0.03, "pad": 0.02, "label": _cbar_label(actogram)},
    )

    _add_dark_period_bar(ax, dark_period)

    ax.set_xticks(_HOUR_TICKS)
    ax.set_xticklabels(_HOUR_LABELS)
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Day")
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=128)

    return fig


def plot_actogram_cohort(
    actogram: xr.DataArray,
    dark_period: tuple[str, str],
    save_path: Path | None = None,
    cmap: str = "viridis",
    vmax: float | None = None,
    group_by: str | None = None,
) -> Figure:
    """Plot a stacked actogram for the full cohort.

    Internally stacks the 3D actogram into a tall 2D array with one row
    per individual * day combination.

    Parameters
    ----------
    actogram : xr.DataArray
        3D DataArray with dimensions ``(individuals, day, minutes)``.
    dark_period : tuple[str, str]
        Start and end of the dark period as ``('HH:MM', 'HH:MM')``.
    save_path : Path | None
        Optional path to save the figure.
    cmap : str
        Colormap. Default is ``"viridis"``.
    vmax : float | None
        Maximum value for the color scale. If ``None``, the 99th percentile
        of the data is used.
    group_by : str | None
        Name of a non-dimension coordinate of the 'individuals' dimension.
        If provided, one subplot per unique group value is created, each
        showing the stacked actogram for individuals in that group. Subplot
        heights are proportional to group size × days. Default is None.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if vmax is None:
        vmax = float(actogram.quantile(0.99))

    n_days = actogram.sizes["day"]
    pcolormesh_kwargs = dict(
        x="minutes",
        y="kage_day_id",
        yincrease=False,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
    )

    if group_by is not None:
        fig, axes, unique_groups = _make_grouped_subplots(
            actogram,
            group_by,
            figsize=(12, 12),
            rows_per_individual=n_days,
        )
        pm = None
        for ax, group in zip(axes, unique_groups, strict=True):
            mask = actogram.coords[group_by].values == group
            group_actogram = actogram.isel(individuals=mask)
            group_tall = _stack_actogram(group_actogram)
            group_inds = group_actogram.individuals.values

            pm = group_tall.plot.pcolormesh(
                **pcolormesh_kwargs, ax=ax, add_colorbar=False
            )
            ax.set_title(f"{group_by} = {group}")
            _set_cohort_yticks(ax, group_inds, n_days)

        for ax in axes:
            _add_dark_period_bar(ax, dark_period, bar_height=0.02)
        _finalize_grouped_xaxis(axes, _HOUR_TICKS, _HOUR_LABELS, "Time of day")
        fig.colorbar(
            pm, ax=axes.tolist(), fraction=0.03, pad=0.02, label=_cbar_label(actogram)
        )

    else:
        actogram_tall = _stack_actogram(actogram)
        individuals = actogram.individuals.values

        fig, ax = plt.subplots(figsize=(12, 12), layout="constrained")
        actogram_tall.plot.pcolormesh(
            **pcolormesh_kwargs,
            ax=ax,
            cbar_kwargs={"fraction": 0.03, "pad": 0.02},
        )
        ax.set_title("Full cohort actogram")
        _set_cohort_yticks(ax, individuals, n_days)
        ax.set_xticks(_HOUR_TICKS)
        ax.set_xticklabels(_HOUR_LABELS)
        ax.set_xlabel("Time of day")
        _add_dark_period_bar(ax, dark_period, bar_height=0.02)

    if save_path:
        plt.savefig(save_path, dpi=128)

    return fig


def plot_mean_daily_activity_profile(
    actogram: xr.DataArray,
    dark_period: tuple[str, str],
    save_path: Path | None = None,
    group_by: str | None = None,
    cmap: str = CMAP,
) -> Figure:
    """Plot the average daily activity profile across individuals.

    Averages the actogram over days to produce a single representative profile
    per individual, then plots each individual as a thin line with the cohort
    mean overlaid. The dark period is shaded.

    Parameters
    ----------
    actogram:
        3D DataArray with dimensions ``(individuals, day, minutes)``.
    dark_period:
        Start and end of the dark period as ``('HH:MM', 'HH:MM')``.
    save_path:
        Optional path to save the figure.
    group_by : str | None
        Name of a non-dimension coordinate of the 'individuals' dimension.
        If provided, traces are coloured by group with a per-group mean
        overlaid. If None, all individuals are treated as one group labelled
        ``"Cohort"``.
    cmap : str
        Qualitative colormap used to assign colours to groups. Default is
        ``CMAP``.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if group_by is not None and group_by not in actogram.coords:
        raise ValueError(f"'{group_by}' is not a coordinate of the DataArray.")

    mean_daily_profile = actogram.mean(dim="day")
    colors = plt.colormaps[cmap].colors

    if group_by is not None:
        unique_groups = np.unique(actogram.coords[group_by].values)
        group_labels = actogram.coords[group_by].values
    else:
        unique_groups = np.array(["Cohort"])
        group_labels = np.full(len(actogram.individuals), "Cohort")

    group_color = {g: colors[i % len(colors)] for i, g in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

    for g in unique_groups:
        mask = group_labels == g
        group_profile = mean_daily_profile.isel(individuals=mask)
        group_mean = group_profile.mean(dim="individuals")
        group_std = group_profile.std(dim="individuals")
        group_mean.plot.line(
            x="minutes",
            ax=ax,
            color=group_color[g],
            lw=2,
            label=f"{g} (n={int(mask.sum())})",
        )
        ax.fill_between(
            group_mean.minutes.values,
            group_mean - group_std,
            group_mean + group_std,
            color=group_color[g],
            alpha=0.2,
        )

    for time_str in dark_period:
        t = hhmm_to_minutes(time_str)
        ax.axvline(t, color="0.5", ls="--", lw=1)
        ax.text(
            t,
            0.98,
            time_str,
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize="small",
            rotation=90,
            color="0.5",
        )

    ax.set_xticks(_HOUR_TICKS)
    ax.set_xticklabels(_HOUR_LABELS)
    ax.set_xlabel("Time of day")
    ax.set_ylabel(f"Activity ({mean_daily_profile.attrs.get('units', 'cm/bin')})")
    ax.set_title("Average daily activity profile")
    ax.set_xlim(0, 24 * 60)
    ax.legend(title="mean ± SD")

    if save_path:
        plt.savefig(save_path, dpi=128)

    return fig


def plot_trajectory_and_occupancy(
    position: xr.DataArray,
    bg_image: np.ndarray | None = None,
    title: str = "",
    save_path: Path | None = None,
    trajectory_kwargs: dict | None = None,
    occupancy_kwargs: dict | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot trajectory and occupancy heatmap side by side.

    Parameters
    ----------
    position : xr.DataArray
        Position data with ``time`` and ``space`` dimensions.
    bg_image : np.ndarray | None
        Optional background image to display behind both panels.
    title : str
        Figure suptitle. Default is "".
    save_path : Path | None
        Optional path to save the plot. If None, the plot will not be saved.
    trajectory_kwargs : dict | None
        Additional keyword arguments forwarded to
        :func:`plot_trajectory` (e.g. ``c``, ``cmap``, ``vmin``, ``vmax``).
    occupancy_kwargs : dict | None
        Additional keyword arguments forwarded to
        :func:`movement.plots.plot_occupancy`.

    Returns
    -------
    (fig, (ax_traj, ax_occ)) : tuple
        The figure and the trajectory / occupancy axes.
    """
    trajectory_kwargs = trajectory_kwargs or {}
    occupancy_kwargs = occupancy_kwargs or {}

    fig, (ax_traj, ax_occ) = plt.subplots(
        ncols=2, figsize=(12, 4), layout="constrained"
    )

    if bg_image is not None:
        height, width = bg_image.shape[:2]
        extent = (0, width, height, 0)
        for ax in (ax_traj, ax_occ):
            ax.imshow(bg_image, extent=extent)
            ax.axis("off")

    plot_trajectory(position, ax=ax_traj, **trajectory_kwargs)
    plot_occupancy(position, ax=ax_occ, **occupancy_kwargs)

    # Invert y-axis on the occupancy panel to match the video frame
    if bg_image is not None:
        ax_occ.set_ylim([height - 1, 0])

    ax_traj.set_title("Trajectory")
    ax_occ.set_title("Occupancy")

    if title:
        fig.suptitle(title)

    if save_path:
        plt.savefig(save_path, dpi=128)

    return fig, (ax_traj, ax_occ)
