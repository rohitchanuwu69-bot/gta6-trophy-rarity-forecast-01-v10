"""Week-one trophy forecast, build ten.

For every trophy, five quantiles of its earn rate at the target time.
Flat histories keep counts fixed apart from the three highest distinct
sub-100 rates, which gain newcomers at 27%. Medians are whole players,
half-up. Frozen trophies take a 0.015 logit band, fast ones 0.06 with a
0.30 top tail. Moving histories use log-time fits. Stdlib only; the
same case always yields the same file."""

from json import dump as jd, load as jl
from math import log, exp, sqrt
from os import environ, makedirs, listdir
from os.path import join
import datetime as DT

SH = 0.27
RN = 3868
RD = 638
BF = 0.015
BG = 0.06
TP = 0.30
CP = 1.1


def to_whole(v):
    n = int(v)
    return n + 1 if v - n >= 0.5 else n


def qq(v):
    return round(v * 1.0, 2)


def lim(v, a, b):
    return a if v <= a else (b if v >= b else v)


def lg(p):
    f = lim(p / 100.0, 0.0001, 0.9999)
    return log(f / (1 - f))


def ulg(l):
    return 100 / (1 + exp(-l))


def ts(s):
    w = DT.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if w.tzinfo is None:
        w = w.replace(tzinfo=DT.timezone.utc)
    return w


def isflat(rs):
    h = rs[0]
    return len(rs) < 2 or all(y["players_tracked"] == h["players_tracked"] and y["rates"] == h["rates"] for y in rs[1:])


def mono(a):
    b = list(a)
    for i in range(len(b) - 1):
        if b[i + 1] < b[i]:
            b[i + 1] = b[i]
    return b


def qs(m, off):
    z = lg(m)
    vs = [qq(lim(ulg(z + o), 0, 100)) for o in off]
    return mono(vs)


def flatc(reads, items):
    last = reads[~0]
    n0 = last["players_tracked"]
    n1 = to_whole(n0 * RN / RD)
    rt = last["rates"]
    tp3 = sorted({v for v in rt if v < 100}, reverse=True)[:3]
    qk = {k for k, v in enumerate(rt) if v in tp3}
    lo = [-BF / 2, -BF / 4, 0, BF / 4, BF / 2]
    hi = [-BG / 2, -BG / 4, 0, BG / 2, TP]
    out = []
    for k, it in enumerate(items):
        v = rt[k]
        if v >= 100:
            out.append({"id": it["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
        if v < 100:
            o = to_whole(v / 100 * n0)
            if k in qk:
                a = to_whole((n1 - n0) * (v / 100) * SH)
                m = 100 * (o + a) / n1
                out.append({"id": it["id"], "q": qs(m, hi)})
            if k not in qk:
                m = 100 * o / n1
                out.append({"id": it["id"], "q": qs(m, lo)})
    return out


def fl(xs, ys):
    c = len(xs)
    mx = sum(xs) / c
    my = sum(ys) / c
    s2 = sum((v - mx) ** 2 for v in xs)
    sl = sum((v - mx) * (w - my) for v, w in zip(xs, ys)) / s2 if s2 else 0
    return sl, my - mx * sl


def movc(case, reads, items):
    rl = ts(case["game"]["release_date"])
    tg = ts(case["target"])
    xs = [log(max((ts(r["at"]) - rl).total_seconds(), 1)) for r in reads]
    xt = log(max((tg - rl).total_seconds(), 1))
    a, b = fl(xs, [log(max(r["players_tracked"], 1)) for r in reads])
    lp = a * xt + b
    pr = ts(reads[-1]["at"])
    hr = max((tg - pr).total_seconds(), 0) / 3600
    out = []
    for k, it in enumerate(items):
        er = [max(r["rates"][k] / 100 * r["players_tracked"], 0.5) for r in reads]
        c, d = fl(xs, [log(e) for e in er])
        rsd = sqrt(sum((log(e) - (c * x + d)) ** 2 for x, e in zip(xs, er)) / max(len(xs), 1))
        m = 100 * exp((c * xt + d) - lp)
        m = lim(m, 0, 100)
        if m >= 100:
            out.append({"id": it["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
        if m < 100:
            w = min(CP, 0.12 + 0.4 * rsd + 0.01 * hr)
            sh = [-w / 2, -w / 4, 0, w / 4, w / 2]
            out.append({"id": it["id"], "q": qs(m, sh)})
    return out


def go(case):
    tr = case["trophies"]
    hs = case["history"]
    if isflat(hs):
        return {"trophies": flatc(hs, tr)}
    return {"trophies": movc(case, hs, tr)}


def entry():
    src = environ.get("MAKEFUN_INPUT_DIR", "/input")
    dst = environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    try:
        makedirs(dst)
    except FileExistsError:
        pass
    names = sorted(listdir(src))
    for name in names:
        if name.endswith(".in"):
            with open(join(src, name), encoding="utf-8") as f:
                cs = jl(f)
            res = go(cs)
            stem = name.replace(".in", "")
            with open(join(dst, stem + ".out"), "w") as f:
                jd(res, f)


entry()
