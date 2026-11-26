"""
MF Tracker — motor de análise em Python.

Mesma lógica do MF-Tracker.html (funções indicators, factorRow, computeNiche, analyze,
bankPlan e evaluateDue), para rodar sem navegador: no backtest e no coletor.

Diferença declarada: notícias e contexto global (fatores "noticias" e "macro") não entram aqui.
No HTML eles só ajustam a chance em no máximo ±4 e ±3 pontos e não têm histórico para testar.
"""
import math
import numpy as np
import pandas as pd

VERSAO_MOTOR = "0.1.0"

CLASSES = {
    "crypto": {"label": "Cripto", "cost": 0.003},
    "meme": {"label": "Memecoins", "cost": 0.006},
    "b3": {"label": "B3", "cost": 0.001},
    "us": {"label": "EUA", "cost": 0.001},
    "fx": {"label": "Câmbio", "cost": 0.005},
    "dex": {"label": "Memecoins novas (DEX)", "cost": 0.01},
}
MIN_TICKET = {"crypto": 10, "meme": 10, "b3": None, "us": 10, "fx": 50, "dex": 5}
DEFAULT_W = {"tendencia": 1.2, "momento": 1.0, "rsi": 0.8, "bollinger": 0.6, "volume": 0.5,
             "psicologia": 0.8, "nicho": 0.6, "noticias": 0.7, "macro": 0.5}
BT_KEYS = ["tendencia", "momento", "rsi", "bollinger", "volume", "psicologia", "nicho"]
ALL_KEYS = BT_KEYS + ["noticias", "macro"]
WARM = 100
HORIZONTES = [3, 7, 14]
DEFAULT_SETTINGS = {"banks": [50, 100, 300, 500, 1000, 2500, 5000, 10000], "minP": 55, "minEdge": 2, "kelly": 0.25}


def clip(x, a=-1.0, b=1.0):
    return np.minimum(b, np.maximum(a, x))


def _sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()


def _rstd(x, n):
    return pd.Series(x).rolling(n).std(ddof=0).to_numpy()


def _rsi(c, n=14):
    o = np.full(len(c), np.nan)
    g = l = 0.0
    for i in range(1, len(c)):
        d = c[i] - c[i - 1]
        G, L = max(d, 0.0), max(-d, 0.0)
        if i <= n:
            g += G; l += L
            if i == n:
                g /= n; l /= n
                o[i] = 100 - 100 / (1 + g / (l or 1e-12))
        else:
            g = (g * (n - 1) + G) / n
            l = (l * (n - 1) + L) / n
            o[i] = 100 - 100 / (1 + g / (l or 1e-12))
    return o


def indicators(df):
    """df: colunas t (segundos UTC), o, h, l, c, v; ordenado do mais antigo ao mais novo."""
    C = df["c"].to_numpy(float); H = df["h"].to_numpy(float); L = df["l"].to_numpy(float); V = df["v"].to_numpy(float)
    prev = np.r_[np.nan, C[:-1]]
    ret = np.r_[0.0, C[1:] / C[:-1] - 1]
    tr = np.where(np.isnan(prev), H - L, np.maximum.reduce([H - L, np.abs(H - prev), np.abs(L - prev)]))
    lv = np.log1p(np.maximum(V, 0))
    lvm, lvs = _sma(lv, 20), _rstd(lv, 20)
    with np.errstate(invalid="ignore", divide="ignore"):
        volz = np.where(lvs > 0, (lv - lvm) / lvs, 0.0)
    streak = np.zeros(len(C))
    for i in range(1, len(C)):
        s = np.sign(C[i] - C[i - 1]); p = streak[i - 1]
        streak[i] = 0 if s == 0 else (p + s if np.sign(p) == s else s)
    hi252 = pd.Series(C).rolling(252, min_periods=1).max().to_numpy()
    roc10 = np.r_[np.full(10, np.nan), C[10:] / C[:-10] - 1] if len(C) > 10 else np.full(len(C), np.nan)
    return {"C": C, "H": H, "L": L, "V": V, "ret": ret, "sma20": _sma(C, 20), "sma50": _sma(C, 50), "sma100": _sma(C, 100),
            "rsi": _rsi(C), "sd20": _rstd(C, 20), "atr": _sma(tr, 14), "vol30": _rstd(ret, 30), "volz": volz,
            "roc10": roc10, "streak": streak, "dd": 1 - C / hi252}


def factor_matrix(cls, df, I, fng=None):
    """Matriz N x 7 com os fatores técnicos (nicho = 0, preenchido depois). Linhas < WARM-1 ficam NaN."""
    N = len(I["C"])
    v = np.where(np.isfinite(I["vol30"]) & (I["vol30"] != 0), I["vol30"], 0.02)
    c = I["C"]
    with np.errstate(invalid="ignore", divide="ignore"):
        tend = clip(0.6 * np.tanh((I["sma20"] / I["sma50"] - 1) / (v * 3)) + 0.4 * np.tanh((c / I["sma100"] - 1) / (v * 6)))
        mom = np.tanh(I["roc10"] / (v * math.sqrt(10)))
        rsi = clip((50 - I["rsi"]) / 25)
        pb = np.where(I["sd20"] > 0, (c - (I["sma20"] - 2 * I["sd20"])) / (4 * I["sd20"]), 0.5)
        bol = clip(-(pb - 0.5) * 2)
        vol = np.where(I["V"] > 0, np.tanh(I["volz"] / 2) * np.sign(I["ret"]), 0.0)
        st, dd = I["streak"], I["dd"]
        psi_price = clip(0.5 * clip(-st / 5) + 0.5 * np.where(dd > 0.12, clip((dd - 0.12) * 4, 0, 1),
                                                                 -clip((0.03 - dd) * 20, 0, 1) * clip(st / 4, 0, 1)))
    psi = psi_price.copy()
    if fng is not None and cls in ("crypto", "meme"):
        days = (df["t"].to_numpy() // 86400).astype(int)
        for i, d in enumerate(days):
            g = fng.get(int(d))
            if g is not None:
                psi[i] = float(clip((50 - g) / 35))
    F = np.column_stack([tend, mom, rsi, bol, vol, psi, np.zeros(N)])
    F[~np.isfinite(F)] = 0.0
    F[: WARM - 1] = np.nan
    return F


def compute_niche(group):
    """group: lista de dicts {df, F} da mesma classe. Preenche a coluna nicho (6) com a média do momento dos pares no mesmo dia."""
    maps = []
    for a in group:
        days = (a["df"]["t"].to_numpy() // 86400).astype(int)
        m = {}
        for i, d in enumerate(days):
            if np.isfinite(a["F"][i, 1]):
                m[d] = a["F"][i, 1]
        maps.append(m)
    for gi, a in enumerate(group):
        days = (a["df"]["t"].to_numpy() // 86400).astype(int)
        for i, d in enumerate(days):
            if not np.isfinite(a["F"][i, 0]):
                continue
            peers = [m[d] for j, m in enumerate(maps) if j != gi and d in m]
            a["F"][i, 6] = float(clip(np.mean(peers))) if peers else 0.0


def combine(F, w):
    ws = np.array([w[k] for k in BT_KEYS])
    return F @ ws / ws.sum()


# Encolhimento: com n casos parecidos, só n/(n+K) do desvio em relação à taxa base é levado a sério.
# K=100 foi fixado antes de olhar o resultado; o backtest de 2026-11-26 mostrou chances infladas em ~10 p.p. sem isso.
SHRINK_K = 100


def analyze_at(cls, I, F, i, w, settings=DEFAULT_SETTINGS, cal_shift=0.0, horizons=HORIZONTES, shrink_k=SHRINK_K):
    """Análise como o HTML faria no fechamento do candle i, usando só dados até i (sem olhar o futuro)."""
    cost = CLASSES[cls]["cost"]
    C = I["C"]
    sHist = combine(F, w)
    sTech = sHist[i]
    if not np.isfinite(sTech):
        return None
    d = 1 if sTech >= 0 else -1
    res = []
    for h in horizons:
        j = np.arange(WARM, max(WARM, i - h))       # j + h < i: o resultado do análogo já era conhecido
        j = j[np.isfinite(sHist[j])]
        if len(j) < 40:
            continue
        fwd = C[j + h] / C[j] - 1
        base = ((d * fwd > cost).sum() + 1) / (len(j) + 2)
        same = np.sign(sHist[j]) == d
        dist = np.abs(sHist[j] - sTech)
        band = 0.08
        sel = same & (dist <= band)
        while band <= 0.5:
            sel = same & (dist <= band)
            if sel.sum() >= 20:
                break
            band += 0.04
        n = int(sel.sum())
        if n < 12:
            continue
        rr = d * fwd[sel] - cost
        wins, losses = rr[rr > 0], -rr[rr <= 0]
        k = n / (n + shrink_k) if shrink_k else 1.0
        p_raw = ((rr > 0).sum() + 1) / (n + 2)
        exp_base = float((d * fwd - cost).mean())
        res.append({"h": h, "n": n, "band": band, "base": base, "p_raw": p_raw, "exp_raw": float(rr.mean()), "shrink": k,
                    "p0": base + (p_raw - base) * k, "exp": exp_base + (float(rr.mean()) - exp_base) * k,
                    "avgWin": float(wins.mean()) if len(wins) else 0.0, "medWin": float(np.median(wins)) if len(wins) else 0.0,
                    "avgLoss": float(losses.mean()) if len(losses) else 0.0, "worst": float(rr.min())})
    if not res:
        return None
    for r in res:
        r["rank"] = r["exp"] / math.sqrt(r["h"])
    best = max(res, key=lambda r: r["rank"])
    p = float(clip(best["p0"] + cal_shift, 0.02, 0.98))
    price = C[i]
    atrPct = I["atr"][i] / price if np.isfinite(I["atr"][i]) else 0.0
    stopPct = max(best["avgLoss"] * 1.2, 1.5 * atrPct)
    b = best["avgWin"] / best["avgLoss"] if best["avgLoss"] > 0 else 1.0
    kelly = max(0.0, p - (1 - p) / b) if b > 0 else 0.0  # sem nenhum ganho entre os casos parecidos: não aposta
    frac = min(0.10, kelly * settings["kelly"])
    edge = p - best["base"]
    isOpp = (p * 100 >= settings["minP"] and edge * 100 >= settings["minEdge"] and best["exp"] > 0 and frac > 0 and abs(sTech) >= 0.12)
    if cls == "dex" and d < 0:
        isOpp = False
    return {"i": i, "dir": d, "sTech": float(sTech), "best": best, "p": p, "price": float(price), "frac": frac, "edge": edge,
            "isOpp": isOpp, "stopPct": stopPct, "target": float(price * (1 + d * best["medWin"])),
            "stop": float(price * (1 - d * stopPct)), "factors": dict(zip(BT_KEYS, map(float, F[i]))), "cost": cost}


def bank_plan(cls, an, settings=DEFAULT_SETTINGS, price_brl=None):
    out = []
    best = an["best"]
    minT = price_brl if cls == "b3" else MIN_TICKET.get(cls, 10)
    for bank in settings["banks"]:
        amount = round(bank * an["frac"] * 100) / 100
        ok, why = amount > 0, ""
        if not ok:
            why = "sem vantagem"
        elif minT and amount < minT:
            ok, why = False, "abaixo do mínimo por ordem"
        out.append({"bank": bank, "amount": amount, "ok": ok, "why": why})
    return out


def learn(w, factors, move, eta=0.08):
    """Pesos multiplicativos, como evaluateDue no HTML: fator que apontou certo ganha peso."""
    for k, v in factors.items():
        w[k] = float(clip(w[k] * math.exp(eta * v * move), 0.25, 3))


def calibration_shift(closed):
    if len(closed) < 10:
        return 0.0
    pred = np.mean([e["p"] for e in closed]); act = np.mean([1.0 if e["hit"] else 0.0 for e in closed])
    k = len(closed) / (len(closed) + 30)
    return float(clip((act - pred) * k, -0.15, 0.15))
