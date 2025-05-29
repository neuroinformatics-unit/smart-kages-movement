"""Utilities for working with datetime data"""

import pandas as pd


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

    # Build an IntervalIndex per group
    results = []  # will collect tuples (iloc_i, iloc_j)
    for _, group in df.groupby(["kage", "date"]):
        iv = pd.IntervalIndex.from_arrays(
            group["start_datetime"], group["end_datetime"], closed="both"
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
                            "end_A": group.at[iloc_i, "end_datetime"],
                            "start_B": group.at[iloc_j, "start_datetime"],
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
        overlaps["overlap_duration"] = (
            (overlaps["end_A"] - overlaps["start_B"])
            .dt.floor("ms")
            .apply(lambda x: str(x).split()[-1])
        )
        return overlaps
