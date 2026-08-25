"""Final runner with non-mutating event masks for every metric and XID code."""

import warnings

import numpy as np
import pandas as pd

import eda_event_windows as eda


def curve_summary(events, curves, short):
    start, end = (eda.SHORT_START, eda.SHORT_END) if short else (eda.LONG_START, eda.LONG_END)
    flag = "isolated_2h" if short else "clean_pre_72h"
    baseline_slice = slice(-60 - start, -30 - start) if short else slice(-72 - start, -48 - start)
    minimum_baseline_bins = 15 if short else 12
    rows = []
    for metric, arrays in curves.items():
        values = arrays[0] if short else arrays[2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            baseline = np.nanmedian(values[:, baseline_slice], axis=1)
        baseline_count = np.isfinite(values[:, baseline_slice]).sum(axis=1)
        normalized = values - baseline[:, None]
        normalized[baseline_count < minimum_baseline_bins] = np.nan
        for code_label, code in [("all", None), ("31", 31), ("43", 43)]:
            mask = events[flag].to_numpy(copy=True)
            if code is not None:
                mask &= events["xid_code"].eq(code).to_numpy()
            selected = pd.DataFrame(normalized[mask])
            selected.insert(0, "cluster_id", events.loc[mask, "cluster_id"].to_numpy())
            clusters = selected.groupby("cluster_id").median(numeric_only=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(clusters, axis=0)
                q25 = np.nanpercentile(clusters, 25, axis=0)
                q75 = np.nanpercentile(clusters, 75, axis=0)
            valid = np.isfinite(clusters).sum(axis=0).to_numpy()
            for index, offset in enumerate(range(start, end)):
                rows.append(
                    {
                        "metric": metric,
                        "xid_code": code_label,
                        "offset": offset,
                        "node_event_clusters": int(valid[index]),
                        "median_delta_from_baseline": median[index],
                        "q25": q25[index],
                        "q75": q75[index],
                    }
                )
    return pd.DataFrame(rows)


eda.curve_summary = curve_summary
eda.main()
