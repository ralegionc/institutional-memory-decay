"""Kaplan-Meier estimation and the log-rank test, implemented directly.

Survival analysis is the right frame for "which events resist decay", because
the data are censored in a way that is easy to get wrong: an article about a
2019 event that is *still* actively edited has not "survived 5 years and then
gone dormant" -- we simply have not observed its dormancy yet. Treating that as
either a long life or a missing value both bias the comparison. Right-censoring
handles it correctly.

Written from scratch rather than pulled from ``lifelines`` so the repository
has no compiled dependencies beyond the scientific stack, and so the estimator
is auditable in one screen.

Event definition used throughout: an article becomes **dormant** at the first
month where its trailing 12-month edit rate drops below 5% of its peak rate,
and stays below for the rest of the observed window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class KMResult:
    time: np.ndarray
    survival: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    lower: np.ndarray = field(default_factory=lambda: np.array([]))
    upper: np.ndarray = field(default_factory=lambda: np.array([]))

    def median_survival(self) -> float:
        below = np.flatnonzero(self.survival <= 0.5)
        return float(self.time[below[0]]) if below.size else np.inf

    def survival_at(self, t: float) -> float:
        idx = np.searchsorted(self.time, t, side="right") - 1
        return float(self.survival[idx]) if idx >= 0 else 1.0

    def to_frame(self) -> pd.DataFrame:
        data = {
            "time": self.time,
            "survival": self.survival,
            "at_risk": self.at_risk,
            "events": self.events,
        }
        if self.lower.size:
            data["lower"] = self.lower
            data["upper"] = self.upper
        return pd.DataFrame(data)


def kaplan_meier(durations, observed, alpha: float = 0.05) -> KMResult:
    """Kaplan-Meier survival curve with Greenwood log-log confidence bands.

    ``observed`` is True when the dormancy event was actually seen and False
    when the article was still active at the end of our observation window.
    """
    durations = np.asarray(durations, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    if durations.size == 0:
        return KMResult(np.array([0.0]), np.array([1.0]), np.array([0]), np.array([0]))

    order = np.argsort(durations)
    durations, observed = durations[order], observed[order]
    unique_times = np.unique(durations[observed]) if observed.any() else np.array([])

    times = [0.0]
    surv = [1.0]
    at_risk_l = [len(durations)]
    events_l = [0]
    variance_sum = 0.0
    var_l = [0.0]

    s = 1.0
    for t in unique_times:
        n_at_risk = int(np.sum(durations >= t))
        d = int(np.sum((durations == t) & observed))
        if n_at_risk == 0:
            continue
        s *= 1 - d / n_at_risk
        if n_at_risk > d:
            variance_sum += d / (n_at_risk * (n_at_risk - d))
        times.append(float(t))
        surv.append(s)
        at_risk_l.append(n_at_risk)
        events_l.append(d)
        var_l.append(variance_sum)

    time = np.asarray(times)
    survival = np.asarray(surv)
    variance = np.asarray(var_l)

    # Log-log transformed bands: they respect the [0, 1] bounds, unlike the
    # plain Greenwood interval which can escape them in small samples.
    z = stats.norm.ppf(1 - alpha / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_s = np.log(np.clip(survival, 1e-12, 1.0))
        se = np.sqrt(variance) / np.abs(log_s)
        theta = np.exp(z * se)
        lower = np.power(survival, theta)
        upper = np.power(survival, 1 / theta)
    lower = np.nan_to_num(lower, nan=survival)
    upper = np.nan_to_num(upper, nan=survival)

    return KMResult(
        time, survival, np.asarray(at_risk_l), np.asarray(events_l), lower, upper
    )


def logrank_test(groups: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """K-sample log-rank test.

    Compares observed versus expected event counts in each group at every
    distinct event time, under the null that all groups share one hazard.
    """
    names = list(groups)
    k = len(names)
    if k < 2:
        return {"statistic": np.nan, "p_value": np.nan, "df": 0}

    all_times = np.unique(
        np.concatenate([d[obs.astype(bool)] for d, obs in groups.values() if len(d)])
        if any(obs.any() for _, obs in groups.values())
        else [np.nan]
    )
    all_times = all_times[np.isfinite(all_times)]
    if all_times.size == 0:
        return {"statistic": np.nan, "p_value": np.nan, "df": k - 1}

    observed = np.zeros(k)
    expected = np.zeros(k)
    var = np.zeros((k, k))

    for t in all_times:
        n_j = np.array([np.sum(groups[n][0] >= t) for n in names], dtype=float)
        d_j = np.array(
            [
                np.sum((groups[n][0] == t) & groups[n][1].astype(bool))
                for n in names
            ],
            dtype=float,
        )
        n, d = n_j.sum(), d_j.sum()
        if n <= 1 or d == 0:
            continue
        e_j = n_j * d / n
        observed += d_j
        expected += e_j
        factor = d * (n - d) / (n - 1)
        p = n_j / n
        var += factor * (np.diag(p) - np.outer(p, p))

    diff = (observed - expected)[:-1]
    v = var[:-1, :-1]
    try:
        stat = float(diff @ np.linalg.pinv(v) @ diff)
    except np.linalg.LinAlgError:
        return {"statistic": np.nan, "p_value": np.nan, "df": k - 1}
    df = k - 1
    return {
        "statistic": stat,
        "p_value": float(stats.chi2.sf(stat, df)),
        "df": df,
        "observed": dict(zip(names, observed)),
        "expected": dict(zip(names, expected)),
    }


def dormancy_events(
    panel: pd.DataFrame,
    threshold: float = 0.05,
    window: int = 12,
    peak_window: int = 3,
    value_col: str = "edits",
) -> pd.DataFrame:
    """Derive (duration, observed) for each article from the monthly panel."""
    rows = []
    for title, group in panel.groupby("title", sort=False):
        group = group.sort_values("month_index")
        y = group[value_col].to_numpy(dtype=float)
        if len(y) < window + 6 or y.sum() <= 0:
            continue
        smooth = pd.Series(y).rolling(peak_window, min_periods=1).mean()
        peak_idx = int(smooth.idxmax())
        peak = float(smooth.max())
        if peak <= 0:
            continue
        trailing = pd.Series(y).rolling(window, min_periods=window).mean()
        after = trailing.iloc[peak_idx:]
        low = after < threshold * peak

        duration = float(len(y) - peak_idx)
        observed = False
        if low.any():
            # Require the drop to persist: a single quiet year inside an
            # otherwise active history is not dormancy.
            for idx in np.flatnonzero(low.to_numpy()):
                if low.to_numpy()[idx:].all():
                    duration = float(idx)
                    observed = True
                    break
        row = {
            "title": title,
            "duration_months": duration,
            "observed": observed,
            "peak_rate": peak,
            "peak_month": peak_idx,
        }
        for col in ("event_class", "event_date"):
            if col in group.columns:
                row[col] = group[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)
