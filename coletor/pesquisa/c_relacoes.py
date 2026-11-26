"""Estudo C: relações entre ativos. Tudo escolhido na 1ª metade, medido na 2ª.
C1 BTC puxa as altcoins? (1 h) · C2 ranking entre criptos, neutro ao mercado · C3 pares que voltam à média"""
import itertools, math
import numpy as np, pandas as pd
import lib as L

out = []
H = {a: L.horario(a) for a in L.CLASSES["crypto"] + L.CLASSES["meme"]}
R = pd.concat({a: np.log(h["c"]).diff() for a, h in H.items()}, axis=1)
btc = R["BTC"]
vol_b = btc.rolling(24 * 30).std()
for k in (1.0, 2.0, 3.0):
    for a in [x for x in R.columns if x != "BTC"]:
        cls = L.CLS_DE[a]
        df = pd.DataFrame({"b": btc, "x": R[a], "xn": R[a].shift(-1), "vb": vol_b}).dropna()
        z = df["b"] / df["vb"]
        sinal = np.sign(df["b"]) * (z.abs() >= k)
        sinal = sinal * (np.sign(df["x"] - df["b"]) != np.sign(df["b"]))
        pnl = (sinal * df["xn"]).loc[sinal != 0]
        m = len(df) // 2; corte = df.index[m]
        p1, p2 = pnl[pnl.index < corte], pnl[pnl.index >= corte]
        t2, nd = L.t_por_dia(p2.values, p2.index) if len(p2) > 5 else (0.0, 0)
        out.append(dict(estudo="C1", cls=cls, desc=f"BTC andou ≥{k} desvios na hora e {a} ficou para trás: seguir o BTC por 1 h",
                        n1=len(p1), m1=p1.mean(), t1=L.t_stat(p1), n2=len(p2), m2=p2.mean(), t2=t2, custo=L.CUSTO[cls], k=k, ativo=a))
        L.conta()
cc = []
for a in [x for x in R.columns if x != "BTC"]:
    df = pd.DataFrame({"b": btc, "xn": R[a].shift(-1)}).dropna()
    m = len(df) // 2
    cc.append(dict(ativo=a, corr1=df.iloc[:m].corr().iloc[0, 1], corr2=df.iloc[m:].corr().iloc[0, 1]))
pd.DataFrame(cc).to_csv(L.OUT + "/c1_corr.csv", index=False)
print(pd.DataFrame(cc).round(4).to_string())

P = pd.concat({a: h["c"] for a, h in H.items()}, axis=1)
P4 = P.resample("4h").last()
r4 = np.log(P4).diff()
res_c2 = []
for Lb in (1, 6, 18, 42):
    for Hb in (1, 6, 18, 42):
        for lado in (1, -1):
            past = np.log(P4).diff(Lb)
            rank = past.rank(axis=1, pct=True)
            ok = past.notna().sum(axis=1) >= 6
            w = ((rank >= 2 / 3).astype(float) - (rank <= 1 / 3).astype(float)) * lado
            w = w.where(ok, 0.0)
            w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            reb = np.arange(len(w)) % Hb == 0
            w = w.where(pd.Series(reb, index=w.index), np.nan).ffill().fillna(0)
            ret = (w.shift(1) * r4).sum(axis=1)
            giro = w.diff().abs().sum(axis=1)
            custo = giro * 0.5 * (0.003 + 0.006) / 2 / 1.0
            liq = ret - custo
            m = len(liq) // 2
            d1 = liq.iloc[:m].groupby(liq.index[:m].normalize()).sum(); d2 = liq.iloc[m:].groupby(liq.index[m:].normalize()).sum()
            b2 = ret.iloc[m:].groupby(ret.index[m:].normalize()).sum()
            res_c2.append(dict(L=Lb * 4, H=Hb * 4, lado="momento" if lado == 1 else "reversão", sh1=d1.mean() / d1.std() * math.sqrt(365),
                               t1=L.t_stat(d1), sh2=d2.mean() / d2.std() * math.sqrt(365), t2=L.t_stat(d2), ano2=d2.mean() * 365, bruto2=b2.mean() * 365,
                               giro_dia=giro.mean() * 6))
            L.conta(9)
c2 = pd.DataFrame(res_c2)
c2.to_csv(L.OUT + "/c2_ranking.csv", index=False)
best = c2.loc[c2["t1"].idxmax()]
out.append(dict(estudo="C2", cls="crypto+meme", desc=f"ranking neutro: {best['lado']} de {best['L']} h, troca a cada {best['H']} h (escolhido na 1ª metade entre 32)",
                n1=None, m1=best["sh1"], t1=best["t1"], n2=None, m2=best["sh2"], t2=best["t2"], ano2=best["ano2"], bruto2=best["bruto2"]))
print(c2.round(3).sort_values("t1", ascending=False).head(10).to_string())

PARES = [("ETH", "BTC"), ("SOL", "ETH"), ("XRP", "BTC"), ("DOGE", "SHIB"), ("ITUB4", "BBAS3"), ("PETR4", "VALE3"), ("QQQ", "SPY"), ("MSFT", "AAPL"), ("BOVA11", "ITUB4")]
res_c3 = []
for x, y in PARES:
    cls = L.CLS_DE[x]
    dx, dy = L.diario(x)["c"], L.diario(y)["c"]
    df = pd.concat({"x": dx, "y": dy}, axis=1).dropna()
    df = df[(df > 0).all(axis=1)]
    lr = np.log(df["x"] / df["y"])
    rx, ry = np.log(df["x"]).diff(), np.log(df["y"]).diff()
    m = len(df) // 2
    grade = []
    for W, k, T in itertools.product((20, 60), (1.5, 2.0, 2.5), (5, 20)):
        z = (lr - lr.rolling(W).mean()) / lr.rolling(W).std()
        pos = np.zeros(len(df)); p = 0.0; dias = 0
        zz = z.values
        for i in range(len(df)):
            if p == 0 and np.isfinite(zz[i]) and abs(zz[i]) >= k:
                p = -np.sign(zz[i]); dias = 0
            elif p != 0:
                dias += 1
                if (np.sign(zz[i]) != -p) or dias >= T or not np.isfinite(zz[i]):
                    p = 0.0
            pos[i] = p
        pos = np.r_[0, pos[:-1]]
        ret = pos * (rx.values - ry.values) / 2
        giro = np.abs(np.diff(np.r_[0, pos]))
        liq = np.nan_to_num(ret) - giro * L.CUSTO[cls]
        s = pd.Series(liq, index=df.index)
        grade.append(dict(W=W, k=k, T=T, sh1=s.iloc[:m].mean() / (s.iloc[:m].std() + 1e-12), t1=L.t_stat(s.iloc[:m][s.iloc[:m] != 0]),
                          t2=L.t_stat(s.iloc[m:][s.iloc[m:] != 0]), ano2=s.iloc[m:].mean() * (365 if cls in ("crypto", "meme") else 252),
                          n2=int((giro[m:] > 0).sum() / 2), s=s))
        L.conta()
    g = max(grade, key=lambda r: r["sh1"])
    res_c3.append(dict(par=f"{x}/{y}", W=g["W"], k=g["k"], T=g["T"], t1=g["t1"], t2=g["t2"], ano2=g["ano2"], n2=g["n2"]))
    out.append(dict(estudo="C3", cls=cls, desc=f"par {x}/{y}: z de {g['W']} dias, entra em |z|≥{g['k']}, sai na média ou em {g['T']} dias",
                    n1=None, m1=None, t1=g["t1"], n2=g["n2"], m2=None, t2=g["t2"], ano2=g["ano2"]))
print(pd.DataFrame(res_c3).round(3).to_string())

df = pd.DataFrame(out)
df.to_csv(L.OUT + "/c_relacoes.csv", index=False)
c1 = df[df.estudo == "C1"]
print(c1.groupby(["k", "cls"])[["t1", "t2", "m2"]].mean().round(4).to_string())
print("C1 com t2>=2:", int((c1.t2 >= 2).sum()), "de", len(c1), "| m2 médio (bruto por operação):", c1.m2.mean())
print("simulações:", L.N_TESTES)
L.salva("c_n.json", {"n": L.N_TESTES})
