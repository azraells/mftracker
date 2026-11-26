"""Núcleo do modelo de aprendizado, reutilizável pela auditoria (mesma lógica do e_extremos_ml.py, com opções):
- embargo: o alvo de todo exemplo de treino termina pelo menos `embargo_h` horas antes do início do mês de teste;
- funding defasado 1 candle a mais (só usa a taxa já publicada);
- placebo: rótulos embaralhados dentro de cada mês, rótulos deslocados no tempo, ou preços trocados por passeio aleatório com a mesma vol;
- retirar grupos de variáveis;
- treino com um ponto a cada 4 h por ativo, previsão em todas as horas do mês de teste."""
import numpy as np, pandas as pd
import lib as L

GRUPOS = {"retornos": ["r1", "r4", "r24", "r72", "r168"], "volatilidade": ["vrat", "vol"], "osciladores": ["rsi", "dma24", "dma168"],
          "volume": ["volu"], "calendário": ["hs", "hc", "ds", "dc"], "BTC": ["btc1", "btc24"], "funding": ["fund"]}


def logit_fit(X, y, lam=1.0, it=25):
    p = X.shape[1]; w = np.zeros(p)
    for _ in range(it):
        z = np.clip(X @ w, -30, 30); pr = 1 / (1 + np.exp(-z))
        g = X.T @ (pr - y) + lam * np.r_[0, w[1:]]
        S = pr * (1 - pr)
        Hm = (X * S[:, None]).T @ X + lam * np.diag(np.r_[0, np.ones(p - 1)])
        try:
            step = np.linalg.solve(Hm, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def auc(p, y):
    p = np.asarray(p, float); y = np.asarray(y, int)
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / n).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / dn)


def feats_1h(c, v):
    lr = np.log(c); F = {}
    for k in (1, 4, 24, 72, 168):
        F[f"r{k}"] = lr.diff(k)
    vc = lr.diff().rolling(24).std(); vl = lr.diff().rolling(168).std()
    F["vrat"] = np.log(vc / vl); F["vol"] = np.log(vl)
    F["rsi"] = (rsi(c) - 50) / 50
    F["dma24"] = lr - lr.rolling(24).mean(); F["dma168"] = lr - lr.rolling(168).mean()
    F["volu"] = np.log((v + 1) / (v.rolling(168).mean() + 1))
    h = c.index.hour; F["hs"] = np.sin(2 * np.pi * h / 24); F["hc"] = np.cos(2 * np.pi * h / 24)
    dw = c.index.dayofweek; F["ds"] = np.sin(2 * np.pi * dw / 7); F["dc"] = np.cos(2 * np.pi * dw / 7)
    return pd.DataFrame(F, index=c.index)


def passeio(h, seed):
    """Troca o preço por um passeio aleatório com a mesma vol de 1 semana do ativo (sem nenhuma estrutura)."""
    rng = np.random.default_rng(seed)
    r = np.log(h["c"]).diff()
    sd = r.rolling(168).std().bfill().values
    z = rng.standard_normal(len(h)) * sd
    c = np.exp(np.log(h["c"].iloc[0]) + np.cumsum(z))
    out = h.copy(); out["c"] = c
    out["h"] = c * (h["h"] / h["c"]).values; out["l"] = c * (h["l"] / h["c"]).values
    return out


def painel(Hh, placebo=None, seed=1):
    ativos = L.CLASSES["crypto"] + L.CLASSES["meme"]
    H = {a: L.horario(a) for a in ativos}
    if placebo == "passeio":
        H = {a: passeio(h, seed + i) for i, (a, h) in enumerate(H.items())}
    btc = H["BTC"]
    btc_r = pd.DataFrame({"btc1": np.log(btc["c"]).diff(), "btc24": np.log(btc["c"]).diff(24)})
    partes = []
    for a in ativos:
        h = H[a]
        F = feats_1h(h["c"], h["v"])
        F = F.join(btc_r)
        f = L.funding(a)
        if f is not None and placebo != "passeio":
            fz = f[~f.index.duplicated()].reindex(F.index, method="ffill").shift(1)  # 1 candle a mais de folga
            F["fund"] = (fz - fz.rolling(24 * 30).mean()) / fz.rolling(24 * 30).std()
        else:
            F["fund"] = 0.0
        F["y"] = np.log(h["c"]).shift(-Hh) - np.log(h["c"])
        F["a"] = a; F["cls"] = L.CLS_DE[a]
        F["o_"] = h["o"]; F["h_"] = h["h"]; F["l_"] = h["l"]; F["c_"] = h["c"]
        partes.append(F)
    P = pd.concat(partes).sort_index()
    P = P.replace([np.inf, -np.inf], np.nan)
    P = P.dropna(subset=[c for c in P.columns if c not in ("y",)])
    return P


def walk_forward(P, Hh, embargo_h=None, tirar=(), placebo=None, seed=1, passo_treino=4):
    embargo_h = Hh if embargo_h is None else embargo_h
    cols = [c for c in P.columns if c not in ("y", "a", "cls", "o_", "h_", "l_", "c_") and c not in tirar]
    rng = np.random.default_rng(seed)
    meses = P.index.to_period("M"); ums = sorted(meses.unique())
    ok_treino = (P.index.hour % passo_treino == 0) & P["y"].notna().values
    y_all = P["y"].values.copy()
    if placebo == "desloca":
        # cada ativo recebe os rótulos de outro momento: deslocamento circular de 45 a 120 dias
        for a in P["a"].unique():
            ix = np.where(P["a"].values == a)[0]
            k = int(rng.integers(45 * 24, 120 * 24))
            y_all[ix] = np.roll(P["y"].values[ix], k)
    if placebo == "embaralha":
        for m in ums:
            ix = np.where((meses == m) & P["y"].notna().values)[0]
            y_all[ix] = y_all[rng.permutation(ix)]
    out = []
    for i, m in enumerate(ums):
        if i < 12:
            continue
        inicio = m.start_time
        lim = inicio - pd.Timedelta(hours=Hh + embargo_h)  # o alvo do último exemplo de treino termina antes do embargo
        tr = ok_treino & (P.index < lim)
        if tr.sum() < 2000:
            continue
        te = (meses == m) & P["y"].notna().values
        Xtr = P.loc[tr, cols]; mu = Xtr.mean(); sd = Xtr.std().replace(0, 1)
        X = np.c_[np.ones(tr.sum()), ((Xtr - mu) / sd).clip(-5, 5).values]
        yt = (y_all[tr] > 0).astype(float)
        w = logit_fit(X, yt, lam=len(yt) * 0.01)
        Xs = np.c_[np.ones(te.sum()), ((P.loc[te, cols] - mu) / sd).clip(-5, 5).values]
        p = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        Q = P.loc[te, ["a", "cls", "r1", "r4", "o_", "h_", "l_", "c_"]].copy()
        Q["p"] = p; Q["base"] = yt.mean(); Q["y"] = y_all[te]
        out.append(Q)
        L.conta()
    return pd.concat(out)


def operacoes(Q, mg, nao_sobrepor=None):
    """Seleciona sinais com |p − base| > margem. nao_sobrepor = h: mantém um ponto a cada h horas por ativo."""
    lado = np.where(Q.p > Q.base + mg, 1, np.where(Q.p < Q.base - mg, -1, 0))
    R = Q[lado != 0].copy(); R["lado"] = lado[lado != 0]
    if nao_sobrepor:
        R = R[R.index.hour % nao_sobrepor == 0] if nao_sobrepor <= 24 else R
        if nao_sobrepor == 24:
            R = R[R.index.hour == 0]
    R["bruto"] = R.lado * R.y
    return R
