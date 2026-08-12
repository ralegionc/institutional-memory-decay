"""Forgetting-curve models and their estimation.

The observable is a count -- how many human edits an article received in month
*t* after its event -- so the models are fitted by **Poisson maximum
likelihood** rather than least squares. This matters more than it sounds. Edit
counts span four orders of magnitude within a single article's history (a
thousand edits in month 0, two in month 90), and ordinary least squares on
those raw counts is dominated entirely by the peak, while OLS on log counts is
undefined for the many months with zero edits. Poisson MLE handles both ends
natively and makes AIC comparisons across the four functional forms legitimate.

Four candidate forms are fitted to every article:

- **Exponential** ``A·exp(-t/τ)`` -- constant hazard of being forgotten. This
  is the classic Ebbinghaus shape and, as it turns out, a poor description of
  collective memory.
- **Power law** ``A·(t+1)^-b`` -- scale-free decay; attention drops fast at
  first and then keeps trickling.
- **Stretched exponential** ``A·exp(-(t/τ)^β)`` -- interpolates between the
  two and is the standard form in the human-dynamics literature.
- **Power law with floor** ``A·(t+1)^-b + c`` -- decay onto a non-zero
  asymptote. The floor ``c`` is the interesting parameter: it is the permanent
  maintenance rate an article settles into once the news cycle is over, and it
  separates events that are *quiet* from events that are *finished*.

Alongside the parametric fits we compute model-free summaries (attention mass
in year one, empirical half-life, temporal Gini). Those never fail to converge
and are used as the robustness check on every parametric claim.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special

log = logging.getLogger(__name__)

EPS = 1e-9


# --------------------------------------------------------------------------
# Model definitions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DecayModel:
    name: str
    n_params: int
    param_names: tuple[str, ...]

    def predict(self, t: np.ndarray, params: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def initial_guess(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def bounds(self, t: np.ndarray, y: np.ndarray):  # pragma: no cover
        raise NotImplementedError


class Exponential(DecayModel):
    def __init__(self) -> None:
        super().__init__("exponential", 2, ("A", "tau"))

    def predict(self, t, p):
        A, tau = p
        return A * np.exp(-t / max(tau, EPS))

    def initial_guess(self, t, y):
        return np.array([max(y[0], 1.0), max(np.ptp(t) / 4, 1.0)])

    def bounds(self, t, y):
        return ([EPS, 1e-3], [max(y.max() * 20, 10), 1e5])


class PowerLaw(DecayModel):
    def __init__(self) -> None:
        super().__init__("power_law", 2, ("A", "b"))

    def predict(self, t, p):
        A, b = p
        return A * np.power(t + 1.0, -b)

    def initial_guess(self, t, y):
        return np.array([max(y[0], 1.0), 1.0])

    def bounds(self, t, y):
        return ([EPS, 0.0], [max(y.max() * 20, 10), 6.0])


class StretchedExponential(DecayModel):
    def __init__(self) -> None:
        super().__init__("stretched_exponential", 3, ("A", "tau", "beta"))

    def predict(self, t, p):
        A, tau, beta = p
        return A * np.exp(-np.power(np.maximum(t, 0) / max(tau, EPS), beta))

    def initial_guess(self, t, y):
        return np.array([max(y[0], 1.0), max(np.ptp(t) / 4, 1.0), 0.5])

    def bounds(self, t, y):
        return ([EPS, 1e-3, 0.05], [max(y.max() * 20, 10), 1e5, 3.0])


class PowerLawFloor(DecayModel):
    """Power-law decay onto a persistent maintenance floor."""

    def __init__(self) -> None:
        super().__init__("power_law_floor", 3, ("A", "b", "c"))

    def predict(self, t, p):
        A, b, c = p
        return A * np.power(t + 1.0, -b) + c

    def initial_guess(self, t, y):
        tail = y[max(len(y) // 2, 1) :]
        floor = max(float(np.median(tail)), 1e-3)
        return np.array([max(y[0] - floor, 1.0), 1.0, floor])

    def bounds(self, t, y):
        return ([EPS, 0.0, 0.0], [max(y.max() * 20, 10), 6.0, max(y.max(), 1.0)])


MODELS: tuple[DecayModel, ...] = (
    Exponential(),
    PowerLaw(),
    StretchedExponential(),
    PowerLawFloor(),
)
MODELS_BY_NAME = {m.name: m for m in MODELS}


# --------------------------------------------------------------------------
# Poisson likelihood machinery
# --------------------------------------------------------------------------


def poisson_nll(params: np.ndarray, model: DecayModel, t: np.ndarray, y: np.ndarray) -> float:
    """Negative Poisson log-likelihood, dropping the constant log(y!) term
    only where it is safe to do so -- we keep it, so that log-likelihoods are
    comparable to any externally computed value."""
    mu = np.clip(model.predict(t, params), EPS, 1e12)
    if not np.all(np.isfinite(mu)):
        return np.inf
    return float(np.sum(mu - y * np.log(mu) + special.gammaln(y + 1.0)))


def aicc(nll: float, n_params: int, n_obs: int) -> float:
    """Small-sample-corrected AIC. Falls back to AIC when n is too small."""
    aic = 2 * n_params + 2 * nll  # nll is already the *negative* log-likelihood
    denom = n_obs - n_params - 1
    if denom <= 0:
        return aic
    return aic + (2 * n_params * (n_params + 1)) / denom


def fit_model(
    model: DecayModel,
    t: np.ndarray,
    y: np.ndarray,
    n_restarts: int = 4,
    seed: int = 0,
    trim_resurgences: bool = True,
    trim_z: float = 3.0,
) -> dict:
    """Fit one model by bounded Poisson MLE with multi-start.

    ``trim_resurgences`` runs the fit twice, masking months whose Pearson
    residual exceeds ``trim_z`` *on the high side only* before the second pass.
    This is not generic outlier rejection -- it encodes a claim about the
    process. Anniversary coverage, a later related event, a court verdict:
    these add a burst of editing on top of the ongoing decay rather than
    changing its rate. Left in, a single large bump drags the fitted exponent
    toward zero and makes a fast-fading event look persistent; on simulated
    data with known parameters this one change moves exponent recovery from
    a Spearman of 0.27 to 0.60. Downward deviations are never trimmed, because
    a genuinely quiet month is signal.
    """
    lower, upper = model.bounds(t, y)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rng = np.random.default_rng(seed)

    starts = [np.clip(model.initial_guess(t, y), lower, upper)]
    for _ in range(n_restarts - 1):
        # Log-uniform jitter: the parameters span many orders of magnitude.
        span = np.log(np.maximum(upper, EPS)) - np.log(np.maximum(lower, EPS))
        draw = np.exp(np.log(np.maximum(lower, EPS)) + rng.random(len(lower)) * span)
        starts.append(np.clip(draw, lower, upper))

    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x0 in starts:
            try:
                res = optimize.minimize(
                    poisson_nll,
                    x0,
                    args=(model, t, y),
                    method="L-BFGS-B",
                    bounds=list(zip(lower, upper)),
                    options={"maxiter": 2000},
                )
            except Exception:
                continue
            if not np.isfinite(res.fun):
                continue
            if best is None or res.fun < best.fun:
                best = res

    if best is None:
        return {"model": model.name, "converged": False, "nll": np.inf, "aicc": np.inf}

    n_trimmed = 0
    if trim_resurgences:
        mu = np.clip(model.predict(t, np.asarray(best.x, dtype=float)), EPS, 1e12)
        pearson = (y - mu) / np.sqrt(mu)
        keep = pearson <= trim_z
        n_trimmed = int((~keep).sum())
        # Refit on the survivors, but only if trimming left a usable series.
        if 0 < n_trimmed and keep.sum() >= max(12, 0.6 * len(t)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    refit = optimize.minimize(
                        poisson_nll,
                        np.asarray(best.x, dtype=float),
                        args=(model, t[keep], y[keep]),
                        method="L-BFGS-B",
                        bounds=list(zip(lower, upper)),
                        options={"maxiter": 2000},
                    )
                    if np.isfinite(refit.fun):
                        # Score the refitted parameters on the FULL series, so
                        # AICc stays comparable across models and across
                        # articles regardless of how much was trimmed.
                        best = refit
                        best.fun = poisson_nll(
                            np.asarray(refit.x, dtype=float), model, t, y
                        )
                except Exception:
                    pass
        else:
            n_trimmed = 0

    params = np.asarray(best.x, dtype=float)
    n = len(t)
    out = {
        "model": model.name,
        "converged": bool(best.success),
        "nll": float(best.fun),
        "aicc": aicc(float(best.fun), model.n_params, n),
        "n_obs": n,
        "n_trimmed": n_trimmed,
    }
    for name, value in zip(model.param_names, params):
        out[f"p_{name}"] = float(value)
    out["_params"] = params

    # Poisson deviance R^2: 1 - D(model) / D(intercept-only).
    mu = np.clip(model.predict(t, params), EPS, 1e12)
    y_safe = np.where(y > 0, y, 1.0)  # avoid log(0); the term is masked out anyway
    dev = 2 * np.sum(np.where(y > 0, y * np.log(y_safe / mu), 0.0) - (y - mu))
    mu0 = max(float(y.mean()), EPS)
    dev0 = 2 * np.sum(np.where(y > 0, y * np.log(y_safe / mu0), 0.0) - (y - mu0))
    out["pseudo_r2"] = float(1 - dev / dev0) if dev0 > 0 else np.nan
    return out


# --------------------------------------------------------------------------
# Model-free summaries
# --------------------------------------------------------------------------


def temporal_gini(y: np.ndarray) -> float:
    """Gini coefficient of attention across months.

    1.0 = every edit happened in a single month (pure flash-in-the-pan);
    0.0 = attention spread perfectly evenly over the article's whole life.
    """
    total = y.sum()
    if total <= 0:
        return np.nan
    v = np.sort(y.astype(float))
    n = len(v)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v) / (n * total)) - (n + 1) / n)


def empirical_half_life(y: np.ndarray, peak_window: int = 3) -> float:
    """Months from the attention peak until the smoothed rate halves.

    Two details do real work here. First, the *same* centred smoother defines
    both the peak and the crossing, so the two are measured on one timescale --
    an earlier version compared a 3-month peak against a trailing 12-month mean
    and inherited a systematic multi-month lag from the mismatch, which showed
    up immediately as a +3 month bias against the simulator's ground truth.
    Second, the crossing is linearly interpolated between the bracketing
    months: event articles frequently halve in well under a month, and an
    integer-month answer collapses most of the corpus onto two or three values.
    """
    if len(y) < 18 or y.sum() <= 0:
        return np.nan
    smooth = (
        pd.Series(y, dtype=float)
        .rolling(peak_window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    peak_idx = find_peak(y, peak_window)
    peak = float(smooth[peak_idx])
    if peak <= 0:
        return np.nan

    target = 0.5 * peak
    after = smooth[peak_idx:]
    below = np.flatnonzero(after <= target)
    if below.size == 0:
        return np.nan
    i = int(below[0])
    if i == 0:
        return 0.0
    prev, curr = after[i - 1], after[i]
    if prev <= curr:
        return float(i)
    return float(i - 1 + (prev - target) / (prev - curr))


PEAK_SEARCH_MONTHS = 12


def find_peak(y: np.ndarray, peak_window: int = 3,
              search_months: int = PEAK_SEARCH_MONTHS) -> int:
    """Index of the attention peak, searched only within the first year.

    The unconstrained argmax is wrong for exactly the articles we care most
    about. A trial verdict three years on, or a tenth-anniversary retrospective,
    can produce more editing in that month than the event itself did -- and then
    "months since peak" starts counting from the resurgence, discarding the
    entire original decay. The peak of an *event's* attention is near the event
    by construction, so we look for it there.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0
    smooth = pd.Series(y).rolling(peak_window, center=True, min_periods=1).mean().to_numpy()
    window = smooth[: max(1, min(search_months, len(smooth)))]
    return int(np.argmax(window))


def summarise_series(y: np.ndarray, peak_window: int = 3) -> dict:
    y = np.asarray(y, dtype=float)
    total = float(y.sum())
    n = len(y)
    smooth = pd.Series(y).rolling(peak_window, center=True, min_periods=1).mean()
    peak_idx = find_peak(y, peak_window)
    return {
        "n_months": n,
        "total_edits": total,
        "peak_month": peak_idx,
        "peak_rate": float(smooth.iloc[peak_idx]) if n else np.nan,
        "share_year1": float(y[:12].sum() / total) if total > 0 else np.nan,
        "share_year1_of_5": float(y[:12].sum() / y[:60].sum())
        if n >= 60 and y[:60].sum() > 0
        else np.nan,
        "tail_rate_5y": float(y[60:].mean()) if n > 60 else np.nan,
        "gini_time": temporal_gini(y),
        "empirical_half_life": empirical_half_life(y, peak_window),
        "months_zero_share": float((y == 0).mean()) if n else np.nan,
    }


# --------------------------------------------------------------------------
# Derived quantities from a fitted curve
# --------------------------------------------------------------------------


def curve_decay_time(
    model: DecayModel,
    params: np.ndarray,
    frac: float = 0.5,
    horizon: int = 600,
    step: float = 0.02,
) -> float:
    """Time for the fitted curve to fall to ``frac`` of its value at t=0.

    Evaluated on a fine sub-month grid: a curve with exponent 1.3 halves in
    about 0.7 months, so an integer-month grid would report 1.0 for most of
    the corpus and destroy the between-article variation we are trying to
    measure.

    Returns ``inf`` when the curve never gets there within the horizon -- which
    is exactly what happens when the maintenance floor exceeds half the peak,
    and is informative rather than a failure.
    """
    grid = np.arange(0, horizon, step, dtype=float)
    mu = model.predict(grid, params)
    if not np.isfinite(mu[0]) or mu[0] <= 0:
        return np.nan
    below = np.flatnonzero(mu <= frac * mu[0])
    return float(grid[below[0]]) if below.size else np.inf


def curve_half_life(model: DecayModel, params: np.ndarray, horizon: int = 600) -> float:
    return curve_decay_time(model, params, 0.5, horizon)


# --------------------------------------------------------------------------
# Article-level driver
# --------------------------------------------------------------------------


def fit_article(
    months: np.ndarray,
    edits: np.ndarray,
    peak_window: int = 3,
    min_months: int = 36,
    seed: int = 0,
) -> dict | None:
    """Fit every model to one article's post-peak attention series."""
    order = np.argsort(months)
    months = np.asarray(months, dtype=float)[order]
    edits = np.asarray(edits, dtype=float)[order]

    if len(months) < min_months or edits.sum() < 20:
        return None

    summary = summarise_series(edits, peak_window)
    peak_idx = summary["peak_month"]

    # Fit from the peak onward: the rise into the peak is the event happening,
    # not the forgetting of it, and including it biases every decay parameter.
    t = months[peak_idx:] - months[peak_idx]
    y = edits[peak_idx:]
    if len(t) < min_months // 2 or y.sum() < 10:
        return None

    fits = [fit_model(m, t, y, seed=seed) for m in MODELS]
    valid = [f for f in fits if np.isfinite(f["aicc"])]
    if not valid:
        return None

    best = min(valid, key=lambda f: f["aicc"])
    best_model = MODELS_BY_NAME[best["model"]]
    min_aicc = best["aicc"]

    row: dict = {
        **summary,
        "fit_n_months": len(t),
        "best_model": best["model"],
        "best_aicc": best["aicc"],
        "best_pseudo_r2": best["pseudo_r2"],
        "half_life_fitted": curve_half_life(best_model, best["_params"]),
        "t90_fitted": curve_decay_time(best_model, best["_params"], 0.1),
    }

    # Akaike weights: the probability each form is the best of the candidate
    # set. Reporting these instead of a bare argmin keeps us honest about how
    # often the winner wins by a nose.
    for f in fits:
        w = np.exp(-0.5 * (f["aicc"] - min_aicc)) if np.isfinite(f["aicc"]) else 0.0
        row[f"aicc_{f['model']}"] = f["aicc"]
        row[f"weight_{f['model']}"] = float(w)
        for key, value in f.items():
            if key.startswith("p_"):
                row[f"{f['model']}_{key[2:]}"] = value
    total_w = sum(row[f"weight_{m.name}"] for m in MODELS)
    if total_w > 0:
        for m in MODELS:
            row[f"weight_{m.name}"] /= total_w

    # Floor height as a share of peak: the permanent-maintenance fraction.
    floor_fit = next((f for f in fits if f["model"] == "power_law_floor"), None)
    if floor_fit and "_params" in floor_fit:
        A, b, c = floor_fit["_params"]
        row["floor_rate"] = float(c)
        row["floor_share_of_peak"] = float(c / (A + c)) if (A + c) > 0 else np.nan
        row["power_exponent"] = float(b)
    return row


def fit_panel(
    panel: pd.DataFrame,
    value_col: str = "edits",
    peak_window: int = 3,
    min_months: int = 36,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit decay models for every article in the monthly panel."""
    rows = []
    for title, group in panel.groupby("title", sort=False):
        group = group.sort_values("month_index")
        result = fit_article(
            group["month_index"].to_numpy(),
            group[value_col].to_numpy(),
            peak_window=peak_window,
            min_months=min_months,
            seed=seed,
        )
        if result is None:
            continue
        result["title"] = title
        for col in ("event_class", "event_date", "date_source"):
            if col in group.columns:
                result[col] = group[col].iloc[0]
        rows.append(result)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    front = ["title", "event_class", "best_model", "half_life_fitted", "empirical_half_life"]
    cols = [c for c in front if c in out.columns] + [
        c for c in out.columns if c not in front
    ]
    return out[cols]
