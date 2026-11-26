"""Placebo do modelo diário do l_horizontes (24 h), refeito sem contaminação.
No l_horizontes, os rótulos foram deslocados de 45 a 120 pregões PARA TRÁS. Com isso, o rótulo de um dia passou a ser um retorno
do passado, que já está dentro das variáveis r60, r120 e r250: o placebo "via" a resposta, e por isso deu AUC 0,547.
Aqui o deslocamento é para a FRENTE e maior que a janela mais longa (300 a 600 pregões), em 3 sementes."""
import numpy as np, pandas as pd
import lib as L, ml_core as M

COLS = ["r1", "r5", "r20", "r60", "r120", "r250", "vrat", "vol", "rsi", "dma20", "dma100", "volu", "ds", "dc"]


def feats_d(d):
    c = d.c; lr = np.log(c); F = {}
    for k in (1, 5, 20, 60, 120, 250):
        F[f"r{k}"] = lr.diff(k)
    F["vrat"] = np.log(lr.diff().rolling(5).std() / lr.diff().rolling(60).std()); F["vol"] = np.log(lr.diff().rolling(60).std())
    F["rsi"] = (M.rsi(c) - 50) / 50; F["dma20"] = lr - lr.rolling(20).mean(); F["dma100"] = lr - lr.rolling(100).mean()
    F["volu"] = np.log((d.v + 1) / (d.v.rolling(20).mean() + 1))
    dw = c.index.dayofweek; F["ds"] = np.sin(2 * np.pi * dw / 5); F["dc"] = np.cos(2 * np.pi * dw / 5)
    return pd.DataFrame(F, index=c.index)


partes = []
for cls in ("b3", "us", "fx"):
    for a in L.CLASSES[cls]:
        d = L.diario(a); d = d[d.c > 0]
        F = feats_d(d); F["y"] = np.log(d.c).shift(-1) - np.log(d.c); F["a"] = a
        partes.append(F)
P = pd.concat(partes).sort_index().replace([np.inf, -np.inf], np.nan).dropna(subset=COLS)
meses = P.index.to_period("M"); ums = sorted(meses.unique())
out = {}
for seed in (None, 51, 52, 53):
    y_all = P.y.values.copy()
    if seed is not None:
        rng = np.random.default_rng(seed)
        for a in P.a.unique():
            ix = np.where(P.a.values == a)[0]
            y_all[ix] = np.roll(P.y.values[ix], -int(rng.integers(300, 600)))  # rótulo de 1 a 2 anos à frente
    pp, yy = [], []
    for i, m in enumerate(ums):
        if i < 24:
            continue
        tr = (meses < m) & (P.index < m.start_time - pd.Timedelta(days=4)) & np.isfinite(y_all)
        te = (meses == m) & np.isfinite(y_all)
        if tr.sum() < 2000 or te.sum() == 0:
            continue
        Xt = P.loc[tr, COLS]; mu, sd = Xt.mean(), Xt.std().replace(0, 1)
        w = M.logit_fit(np.c_[np.ones(tr.sum()), ((Xt - mu) / sd).clip(-5, 5).values], (y_all[tr] > 0).astype(float), lam=tr.sum() * 0.01)
        pp.append(1 / (1 + np.exp(-np.c_[np.ones(te.sum()), ((P.loc[te, COLS] - mu) / sd).clip(-5, 5).values] @ w))); yy.append(y_all[te])
        L.conta()
    out["real" if seed is None else f"placebo semente {seed}"] = M.auc(np.concatenate(pp), (np.concatenate(yy) > 0).astype(int))
print(out)
L.salva("l_placebo_diario.json", out)
L.salva("l_placebo_n.json", {"n": L.N_TESTES})
