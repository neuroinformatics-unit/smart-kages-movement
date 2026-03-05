"""Circadian metrics computed from binned activity data."""

import pandas as pd
import xarray as xr

from smart_kages_movement.datetime import hhmm_to_minutes


def diurnality_index(actogram: xr.DataArray, dark_period: tuple[str, str]) -> pd.Series:
    """Compute the Diurnality Index (DI) per individual.

    ``DI = (A_light - A_dark) / (A_light + A_dark)``

    where ``A_light`` and ``A_dark`` are the total activity summed over light
    and dark bins respectively, for a given day. The per-day values are
    averaged across all days.

    Range is from -1 (fully nocturnal) to +1 (fully diurnal),
    with 0 indicating no preference.

    Parameters
    ----------
    actogram:
        DataArray with dimensions (individuals, day, minutes).
    dark_period:
        Start and end of the dark period as ``("HH:MM", "HH:MM")``.

    Returns
    -------
    pd.Series
        DI per individual, indexed by individuals.
    """
    dark_start = hhmm_to_minutes(dark_period[0])
    dark_end = hhmm_to_minutes(dark_period[1])

    is_dark = (actogram.minutes >= dark_start) & (actogram.minutes <= dark_end)
    dark_act = actogram.where(is_dark).sum(dim="minutes")
    light_act = actogram.where(~is_dark).sum(dim="minutes")

    di = ((light_act - dark_act) / (light_act + dark_act)).mean(dim="day")
    di.name = "DI"
    return di.to_pandas()


def intra_daily_variability(activity: xr.DataArray) -> pd.Series:
    """Compute Intra-daily Variability (IV) per individual.

    ``IV = n * sum((A[i+1] - A[i])^2) / ((n-1) * sum((A[i] - A_mean)^2))``

    where ``A[i]`` is the activity in bin ``i``, ``A_mean`` is the overall
    mean activity, and ``n`` is the number of valid (non-NaN) observations.
    Differences that span a NaN gap are also excluded, so gaps do not
    artificially inflate IV.

    Higher values indicate more frequent transitions between rest and activity
    (more fragmentation).

    Parameters
    ----------
    activity:
        DataArray with dimensions (time, individuals).

    Returns
    -------
    pd.Series
        IV per individual, indexed by individuals.
    """
    n_obs = activity.notnull().sum(dim="time")
    diffs_sq = activity.diff(dim="time") ** 2
    n_diffs = diffs_sq.notnull().sum(dim="time")
    total_ss = ((activity - activity.mean(dim="time", skipna=True)) ** 2).sum(
        dim="time", skipna=True
    )

    iv = (n_obs * diffs_sq.sum(dim="time", skipna=True)) / (n_diffs * total_ss)
    iv.name = "IV"
    return iv.to_pandas()


def inter_daily_stability(activity: xr.DataArray, actogram: xr.DataArray) -> pd.Series:
    """Compute Inter-daily Stability (IS) per individual.

    ``IS = n_t * sum((A_h_mean - A_mean)^2) / (n_h * sum((A[i] - A_mean)^2))``

    where ``A[i]`` is the activity in observation ``i``, ``A_mean`` is the
    overall mean activity, ``A_h_mean`` is the mean activity at time-of-day
    bin ``h`` averaged across all days (the average daily profile), ``n_h`` is
    the number of bins per day, and ``n_t`` is the total number of valid
    observations.

    Higher values indicate that each day looks more similar to the others
    (more stable rhythm).

    Parameters
    ----------
    activity:
        DataArray with dimensions (time, individuals).
    actogram:
        DataArray with dimensions (individuals, day, minutes).

    Returns
    -------
    pd.Series
        IS per individual, indexed by individuals.
    """
    mean_daily_profile = actogram.mean(dim="day")
    overall_mean = activity.mean(dim="time", skipna=True)
    n_t = activity.notnull().sum(dim="time")
    n_h = actogram.sizes["minutes"]

    between_ss = ((mean_daily_profile - overall_mean) ** 2).sum(
        dim="minutes", skipna=True
    )
    total_ss = ((activity - overall_mean) ** 2).sum(dim="time", skipna=True)

    is_metric = (n_t * between_ss) / (n_h * total_ss)
    is_metric.name = "IS"
    return is_metric.to_pandas()


def compute_circadian_metrics(
    ds_activity: xr.Dataset, dark_period: tuple[str, str]
) -> pd.DataFrame:
    """Compute all circadian metrics and return them as a combined DataFrame.

    Parameters
    ----------
    ds_activity:
        Dataset containing ``activity`` (time, individuals) and ``actogram``
        (individuals, day, minutes) data variables.
    dark_period:
        Start and end of the dark period as ``("HH:MM", "HH:MM")``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``DI``, ``IV``, ``IS``, indexed by individuals.
    """
    activity = ds_activity.activity
    actogram = ds_activity.actogram

    return pd.DataFrame(
        {
            "DI": diurnality_index(actogram, dark_period),
            "IV": intra_daily_variability(activity),
            "IS": inter_daily_stability(activity, actogram),
        }
    )
