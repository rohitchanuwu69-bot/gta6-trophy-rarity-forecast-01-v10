#!/usr/bin/env python3
"""Trophy Dilution Forecaster v10.

Forecasts each GTA VI trophy's Exophase earn rate at the case target.
Flat histories: earned counts stay frozen except for the trophies whose
last rate is among the top three distinct sub-100 rates; those gain
newcomers at 27% of their current rate. Every median is a whole-player
count (half-up rounding). The spread is split: frozen trophies get a
very tight 0.015 logit band, fast trophies a 0.06 band with a wider
upper side. Moving histories: log-time least-squares fits for players
and earned counts, ratio as median, spread growing with horizon and
fit residuals. Python 3.13 standard library only. Deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone

FAST_EARN = 0.27
REFRESH_NUM = 3868
REFRESH_DEN = 638
FROZEN_SPREAD = 0.015
FAST_SPREAD = 0.06
FAST_UPPER = 0.30
MOVING_SPREAD_CAP = 1.1
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def half_up(x):
    return math.floor(x + 0.5)


def round2(x):
    return round(x, 2)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(p):
    p = clamp(p / 100.0, 0.0001, 0.9999)
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    return 100.0 / (1.0 + math.exp(-l))


def parse_time(s):
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_flat(history):
    if len(history) < 2:
        return True
    first = history[0]
    for read in history[1:]:
        if read["players_tracked"] != first["players_tracked"]:
            return False
        if read["rates"] != first["rates"]:
            return False
    return True


def monotone(qs):
    for j in range(1, len(qs)):
        if qs[j] < qs[j - 1]:
            qs[j] = qs[j - 1]
    return qs


def quantiles(median, offsets):
    m = logit_pct(median)
    qs = [round2(clamp(inv_logit_pct(m + o), 0.0, 100.0)) for o in offsets]
    return monotone(qs)


def flat_rows(case, history, trophies):
    last = history[-1]
    p0 = last["players_tracked"]
    p1 = half_up(p0 * REFRESH_NUM / REFRESH_DEN)
    rates = last["rates"]
    distinct = sorted({r for r in rates if r < 100.0}, reverse=True)[:3]
    fast = {i for i, r in enumerate(rates) if r in distinct}
    frozen_offs = [-FROZEN_SPREAD / 2, -FROZEN_SPREAD / 4, 0.0,
                   FROZEN_SPREAD / 4, FROZEN_SPREAD / 2]
    fast_offs = [-FAST_SPREAD / 2, -FAST_SPREAD / 4, 0.0,
                 FAST_SPREAD / 2, FAST_UPPER]
    rows = []
    for idx, trophy in enumerate(trophies):
        r = rates[idx]
        if r >= 100.0:
            rows.append({"id": trophy["id"], "q": [100.0] * 5})
            continue
        old = half_up(r / 100.0 * p0)
        if idx in fast:
            add = half_up((p1 - p0) * (r / 100.0) * FAST_EARN)
            med = 100.0 * (old + add) / p1
            rows.append({"id": trophy["id"],
                         "q": quantiles(med, fast_offs)})
        else:
            med = 100.0 * old / p1
            rows.append({"id": trophy["id"],
                         "q": quantiles(med, frozen_offs)})
    return rows


def _fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def moving_rows(case, history, trophies):
    release = parse_time(case["game"]["release_date"])
    target = parse_time(case["target"])
    xs = [math.log(max((parse_time(r["at"]) - release).total_seconds(), 1.0))
          for r in history]
    xt = math.log(max((target - release).total_seconds(), 1.0))
    slope_p, icept_p = _fit(
        xs, [math.log(max(r["players_tracked"], 1)) for r in history])
    log_p = slope_p * xt + icept_p
    last_at = parse_time(history[-1]["at"])
    horizon_h = max((target - last_at).total_seconds(), 0.0) / 3600.0
    rows = []
    for idx, trophy in enumerate(trophies):
        earned = [max(r["rates"][idx] / 100.0 * r["players_tracked"], 0.5)
                  for r in history]
        slope_e, icept_e = _fit(xs, [math.log(e) for e in earned])
        resid = math.sqrt(
            sum((math.log(e) - (slope_e * x + icept_e)) ** 2
                for x, e in zip(xs, earned)) / max(len(xs), 1))
        med = 100.0 * math.exp((slope_e * xt + icept_e) - log_p)
        med = clamp(med, 0.0, 100.0)
        if med >= 100.0:
            rows.append({"id": trophy["id"], "q": [100.0] * 5})
            continue
        spread = min(MOVING_SPREAD_CAP,
                     0.12 + 0.4 * resid + 0.01 * horizon_h)
        offs = [-spread / 2, -spread / 4, 0.0, spread / 4, spread / 2]
        rows.append({"id": trophy["id"], "q": quantiles(med, offs)})
    return rows


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if is_flat(history):
        return {"trophies": flat_rows(case, history, trophies)}
    return {"trophies": moving_rows(case, history, trophies)}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh)
            fh.write("\n")


if __name__ == "__main__":
    main()
