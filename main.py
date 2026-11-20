"""gta6-trophy-rarity-forecast-01 entry: ensemble trophy-rarity forecaster.

Forecasts, for every trophy of Grand Theft Auto VI on PS5, five quantiles of
its Exophase earn rate at the case's target, from the house's reads before the
case's checkpoint. Pure standard library, deterministic, offline.

Model (per trophy, on the logit scale, x = ln(1 + hours since release)):
  A. logit-linear: least squares of logit(rate) on x over the recent window.
  B. logit-linear on the last 24 h only (the current growth regime).
  C. earners / dilution: log(earners + 0.5) on x and log(players_tracked) on
     x, forecast rate = 100 * earners* / players* -- separates real trophy
     progress from the tracked-player pool growing and diluting rates.
  D. saturating logistic: rate = L * sigmoid(a + b*x) with L on a grid,
     the best L by fit error -- early story trophies flatten as the active
     population finishes them.
The median is the backtest-weighted mean of the models' medians in logit
space. Backtesting (fit before a cutoff inside the history, score the
forecast at a later read with the same pinball loss the contest uses)
chooses the weights and measures each trophy's residual spread; the
quantile spread is that residual spread plus an extrapolation penalty that
grows with the log-time distance to the target, so a 4.5-day final gets a
wider band than a 90-minute practice case. Consecutive reads with an
unchanged page (Exophase refreshes discretely) are collapsed to one point
so a frozen stretch cannot flatten the fitted slope. Upward logit growth
is damped (saturation: the active population finishes early content and
the marginal player is slower), while downward moves -- dilution as the
tracked pool grows -- pass through undamped. A trophy the last read puts
at ~100% keeps a wide lower tail: new players can dilute it below 100%.

Run: python3 main.py  (MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR override
/input and /output, as the starter documents.)
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Tier priors, used only when a case has no usable history at all.
TIER_PRIOR = {"platinum": 0.5, "gold": 2.0, "silver": 4.0, "bronze": 8.0}

# Upward logit growth is damped by 1 / (1 + ASYM_DAMP * extra_log_time):
# early-content earn rates saturate as the active population finishes it.
ASYM_DAMP = 0.3

# A trophy the last read puts at ~100% can only fall (new tracked players
# dilute it); keep a wide lower tail instead of a razor-thin band at 100.
NEAR100_Q = [97.5, 99.3, 99.9, 99.99, 99.99]


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 1.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def hours_since_release(iso):
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return (dt - RELEASE).total_seconds() / 3600.0


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_std)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return (ys[0] if ys else 0.0, 0.0, 0.0)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        b = 0.0
    else:
        b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    return (a, b, math.sqrt(max(var, 0.0)))


def distinct_reads(history):
    """Collapse consecutive reads with an identical rates vector (and the
    same players_tracked) into one point at the latest timestamp: the page
    was frozen between Exophase refreshes, so the repeats carry no signal
    and would flatten any fitted slope."""
    out = []
    for read in history:
        rates = tuple(read.get("rates") or [])
        if out and out[-1]["rates"] == rates and \
                out[-1]["players_tracked"] == read.get("players_tracked"):
            out[-1] = read
        else:
            out.append(read)
    return out


class TrophySeries:
    """One trophy's usable time series, after collapsing frozen reads."""

    def __init__(self, idx, case, reads):
        self.idx = idx
        self.points = []   # (t_hours, rate_pct)
        self.earners = []  # (t_hours, earners, players_tracked)
        for read in reads:
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            t = hours_since_release(read["at"])
            r = float(rates[idx])
            self.points.append((t, r))
            n = read.get("players_tracked")
            if n:
                self.earners.append((t, r * float(n) / 100.0, float(n)))

    def window(self, t_end, window_hours):
        pts = [(t, r) for (t, r) in self.points if t <= t_end + 1e-9]
        if window_hours is not None:
            pts = [(t, r) for (t, r) in pts if t >= t_end - window_hours]
        return pts


def fit_logit_linear(pts, t_star):
    """Model A/B: logit(rate) ~ a + b*ln(1+t). Returns (median_pct, resid)."""
    if not pts:
        return None
    xs = [math.log1p(max(t, 0.0)) for (t, _) in pts]
    ys = [logit(r) for (_, r) in pts]
    ws = [1.0] * len(pts)
    a, b, resid = wls(xs, ys, ws)
    lg = a + b * math.log1p(max(t_star, 0.0))
    return (100.0 * sigmoid(lg), resid)


def fit_earners(series, t_end, window_hours, t_star):
    """Model C: log(earners+0.5) and log(players) each ~ a + b*ln(1+t)."""
    rows = [(t, e, n) for (t, e, n) in series.earners if t <= t_end + 1e-9]
    if window_hours is not None:
        rows = [(t, e, n) for (t, e, n) in rows if t >= t_end - window_hours]
    if len(rows) < 2:
        return None
    xs = [math.log1p(max(t, 0.0)) for (t, _, _) in rows]
    ye = [math.log(e + 0.5) for (_, e, _) in rows]
    yn = [math.log(n) for (_, _, n) in rows]
    ae, be, re_ = wls(xs, ye, [1.0] * len(rows))
    an, bn, _rn = wls(xs, yn, [1.0] * len(rows))
    x_star = math.log1p(max(t_star, 0.0))
    e_star = math.exp(ae + be * x_star) - 0.5
    n_star = math.exp(an + bn * x_star)
    if n_star <= 0:
        return None
    return (100.0 * max(e_star, 0.0) / n_star, re_)


def fit_saturating(pts, t_star):
    """Model D: rate = L*sigmoid(a+b*x), L on a grid, best L by SSE."""
    if len(pts) < 3:
        return None
    best = None
    for L in (0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        xs, ys = [], []
        ok = True
        for (t, r) in pts:
            if r >= L * 100.0:
                ok = False
                break
            xs.append(math.log1p(max(t, 0.0)))
            ys.append(logit(r / L))
        if not ok or len(xs) < 3:
            continue
        a, b, resid = wls(xs, ys, [1.0] * len(xs))
        sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        if best is None or sse < best[0]:
            lg = a + b * math.log1p(max(t_star, 0.0))
            best = (sse, 100.0 * L * sigmoid(lg), resid)
    return None if best is None else (best[1], best[2])


def model_medians(series, t_end, t_star):
    """Median forecast (percent) of each model at t_star, or None."""
    out = {}
    pts_all = series.window(t_end, None)
    pts_48 = series.window(t_end, 48.0)
    pts_24 = series.window(t_end, 24.0)
    a = fit_logit_linear(pts_48 if len(pts_48) >= 3 else pts_all, t_star)
    if a:
        out["A"] = a
    b = fit_logit_linear(pts_24, t_star)
    if b and len(pts_24) >= 3:
        out["B"] = b
    c = fit_earners(series, t_end, 48.0, t_star)
    if c:
        out["C"] = c
    d = fit_saturating(pts_48 if len(pts_48) >= 3 else pts_all, t_star)
    if d:
        out["D"] = d
    return out


def pinball_loss(y_pct, quantiles):
    """Mean pinball loss on the logit scale for one trophy (practice
    scoring helper, also used by the backtest)."""
    ly = logit(y_pct)
    total = 0.0
    for tau, q in zip(LEVELS, quantiles):
        d = ly - logit(q)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(LEVELS)


def backtest(case, series_list, reads):
    """Fit each model before cutoffs inside the history and score its
    forecast at a later read. Returns (weights, resid_std per trophy)."""
    times = sorted({hours_since_release(r["at"]) for r in reads})
    if len(times) < 6:
        return None, None
    t_min, t_max = times[0], times[-1]
    span = t_max - t_min
    cutoffs = []
    for frac in (0.5, 0.75):
        c = t_min + frac * span
        h = min(24.0, t_max - c)
        if h >= 6.0:
            cutoffs.append((c, h))
    if not cutoffs:
        return None, None
    model_loss = {}
    model_n = {}
    resid = {i: [] for i in range(len(series_list))}
    for (c, h) in cutoffs:
        t_eval = c + h
        # actual rates: first distinct read at/after t_eval
        actual = None
        for read in reads:
            if hours_since_release(read["at"]) >= t_eval - 1e-9:
                actual = read
                break
        if actual is None:
            continue
        x_star = math.log1p(max(t_eval, 0.0))
        base = 0.30 + 0.25 * (x_star - math.log1p(max(c, 0.0)))
        sigma = min(max(base, 0.35), 2.0)
        for i, series in enumerate(series_list):
            rates = actual.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            meds = model_medians(series, c, t_eval)
            if not meds:
                continue
            y = float(rates[i])
            for name, (m, _r) in meds.items():
                lg = logit(m)
                qs = [100.0 * sigmoid(lg + z * sigma) for z in Z]
                loss = pinball_loss(y, qs)
                model_loss[name] = model_loss.get(name, 0.0) + loss
                model_n[name] = model_n.get(name, 0) + 1
            # ensemble residual for this trophy
            if meds:
                lg_e = sum(logit(m) for (m, _r) in meds.values()) / len(meds)
                resid[i].append(logit(y) - lg_e)
    if not model_loss:
        return None, None
    avg = {k: model_loss[k] / model_n[k] for k in model_loss}
    best = min(avg.values())
    weights = {k: math.exp(-(v - best) / 0.25) for k, v in avg.items()}
    tot = sum(weights.values())
    weights = {k: v / tot for k, v in weights.items()}
    resid_std = {}
    for i, rs in resid.items():
        if len(rs) >= 2:
            mean = sum(rs) / len(rs)
            var = sum((r - mean) ** 2 for r in rs) / len(rs)
            resid_std[i] = math.sqrt(var)
    return weights, resid_std


def forecast_case(case):
    trophies = case["trophies"]
    reads = distinct_reads(case.get("history") or [])
    t_star = hours_since_release(case["target"])
    t_last = hours_since_release(case["at"])
    if not reads:
        rows = []
        for tr in trophies:
            med = TIER_PRIOR.get(tr.get("tier"), 5.0)
            lg = logit(med)
            rows.append({"id": tr["id"],
                         "q": [round4(100.0 * sigmoid(lg + z * 1.5))
                               for z in Z]})
        return {"trophies": rows}

    series_list = [TrophySeries(i, case, reads) for i in range(len(trophies))]
    weights, resid_std = backtest(case, series_list, reads)
    if not weights:
        weights = {"A": 0.4, "B": 0.2, "C": 0.25, "D": 0.15}
    x_star = math.log1p(max(t_star, 0.0))
    x_last = math.log1p(max(t_last, 0.0))
    extra = max(x_star - x_last, 0.0)

    rows = []
    for i, tr in enumerate(trophies):
        series = series_list[i]
        last_rate = series.points[-1][1] if series.points else None
        if last_rate is not None and last_rate >= 99.5:
            rows.append({"id": tr["id"], "q": list(NEAR100_Q)})
            continue
        meds = model_medians(series, t_last, t_star)
        if meds:
            lg_last = logit(last_rate) if last_rate is not None \
                else logit(TIER_PRIOR.get(tr.get("tier"), 5.0))
            num = 0.0
            den = 0.0
            for name, (m, _r) in meds.items():
                w = weights.get(name, 0.0)
                lg = logit(m)
                g = lg - lg_last
                if g > 0:
                    lg = lg_last + g / (1.0 + ASYM_DAMP * extra)
                num += w * lg
                den += w
            lg_med = num / den if den > 0 else lg_last
        else:
            lg_med = logit(TIER_PRIOR.get(tr.get("tier"), 5.0))
        rstd = resid_std.get(i, 0.0) if resid_std else 0.0
        base = 0.30 + 0.25 * extra
        sigma = math.sqrt(rstd * rstd + base * base)
        sigma = min(max(sigma, 0.35), 2.0)
        qs = [round4(100.0 * sigmoid(lg_med + z * sigma)) for z in Z]
        qs = [min(max(q, 0.01), 99.99) for q in qs]
        # keep non-decreasing after rounding/clamping
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            # Never emit an invalid case: fall back to last-read persistence.
            last = case["history"][-1] if case.get("history") else None
            rows = []
            for i, tr in enumerate(case["trophies"]):
                r = 0.0
                if last and last.get("rates") and i < len(last["rates"]) \
                        and last["rates"][i] is not None:
                    r = float(last["rates"][i])
                lg = logit(r)
                rows.append({"id": tr["id"],
                             "q": [round4(100.0 * sigmoid(lg + z * 0.5))
                                   for z in Z]})
            forecast = {"trophies": rows}
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
