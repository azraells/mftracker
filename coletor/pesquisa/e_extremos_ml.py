"""Estudo E.
E1 Depois de um dia extremo (|retorno| ≥ k desvios de 20 dias): o preço continua ou volta? (diário, todos)
   Escolhe k, prazo e sentido na 1ª metade de cada classe; mede na 2ª.
E2 Modelo de aprendizado (regressão logística com penalização, escrita em numpy), treino só com o passado
   (walk-forward: retreina todo mês, usa tudo o que veio antes). Dois cenários:
   - cripto e memecoins em 1 h, previsão de 4 h e 24 h, um ponto a cada 4 h;
   - todos os ativos no diário, previsão de 5 dias.
   Mede: AUC (0,5 = cara ou coroa), acerto contra taxa base e resultado líquido operando só quando a
   probabilidade passa de 0,5 + margem."""
import itertools, math
import numpy as np, pandas as pd
import lib as L

out = []

# ---------------------------------------------------------------- E1
for cls, ativos in L.CLASSES.items():
    evs = []
    for a in ativos:
        d = L.diario(a); d = d[d["c"] > 0]
        r = d["c"].pct_change()
        sd = r.rolling(20).std().shift(1)
        z = r / sd
        for H in (1, 3, 5):
            fwd = d["c"].shift(-H) / d["c"] - 1
            evs.append(pd.DataFrame({"z": z, "fwd": fwd, "H": H, "a": a}).dropna())
        L.conta(3)
    E = pd.concat(evs)
    datas = E.index.unique().sort_values(); corte = datas[len(datas) // 2]
    grade = []
    for k, H, lado in itertools.product((2.0, 2.5, 3.0), (1, 3, 5), (1, -1)):
        s = E[(E.H == H) & (E.z.abs() >= k)]
        pnl = lado * np.sign(s.z) * s.fwd - L.CUSTO[cls]
        p1, p2 = pnl[pnl.index < corte], pnl[pnl.index >= corte]
        if len(p1) < 20 or len(p2) < 10:
            continue
        t1, _ = L.t_por_dia(p1.values, p1.index); t2, nd = L.t_por_dia(p2.values, p2.index)
        grade.append(dict(k=k, H=H, lado="continua" if lado == 1 else "volta", n1=len(p1), m1=p1.mean(), t1=t1, n2=len(p2), m2=p2.mean(), t2=t2))
        L.conta()
    g = max(grade, key=lambda r: r["t1"])
    out.append(dict(estudo="E1", cls=cls, desc=f"depois de dia com |retorno| ≥ {g['k']} desvios, aposta que {g['lado']} por {g['H']} dia(s)",
                    n1=g["n1"], m1=g["m1"], t1=g["t1"], n2=g["n2"], m2=g["m2"], t2=g["t2"]))
    print(cls, pd.DataFrame(grade).round(4).sort_values("t1", ascending=False).head(4).to_string())


# ---------------------------------------------------------------- E2 utilidades
def logit_fit(X, y, lam=1.0, it=25):
    n, p = X.shape
    w = np.zeros(p)
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
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / n).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / dn)


def feats(c, v, passo, extra=None):
    """passo = quantos candles formam '1 dia' (24 no 1 h, 1 no diário)."""
    lr = np.log(c)
    F = {}
    for k in (1, 4, 24, 72, 168) if passo == 24 else (1, 5, 20, 60, 120, 250):
        F[f"r{k}"] = lr.diff(k)
    vc = lr.diff().rolling(passo).std() if passo > 1 else lr.diff().rolling(5).std()
    vl = lr.diff().rolling(passo * 7 if passo > 1 else 60).std()
    F["vrat"] = np.log(vc / vl)
    F["vol"] = np.log(vl)
    F["rsi"] = (rsi(c) - 50) / 50
    for k in ((24, 168) if passo == 24 else (20, 100)):
        F[f"dma{k}"] = lr - lr.rolling(k).mean()
    F["volu"] = np.log((v + 1) / (v.rolling(passo * 7 if passo > 1 else 20).mean() + 1))
    if passo == 24:
        h = c.index.hour; F["hs"] = np.sin(2 * np.pi * h / 24); F["hc"] = np.cos(2 * np.pi * h / 24)
    dw = c.index.dayofweek; F["ds"] = np.sin(2 * np.pi * dw / 7); F["dc"] = np.cos(2 * np.pi * dw / 7)
    df = pd.DataFrame(F, index=c.index)
    if extra is not None:
        df = df.join(extra)
    return df


def walk_forward(painel, nome, H_lbl, passo_amostra, custo_de, margens=(0.0, 0.02, 0.05)):
    """painel: DataFrame com colunas de atributos + 'y' (retorno futuro) + 'a' (ativo) + 'cls', índice = data."""
    painel = painel.replace([np.inf, -np.inf], np.nan).dropna()
    painel = painel.iloc[::passo_amostra] if passo_amostra > 1 else painel
    cols = [c for c in painel.columns if c not in ("y", "a", "cls", "yfim")]
    meses = painel.index.to_period("M")
    ums = sorted(meses.unique())
    pred = []
    for i, m in enumerate(ums):
        treino = painel[(meses < m) & (painel["yfim"] < m)]  # o alvo do treino também terminou antes do mês de teste
        if len(ums[:i]) < 12 or len(treino) < 2000:
            continue
        teste = painel[meses == m]
        mu = treino[cols].mean(); sd = treino[cols].std().replace(0, 1)
        Xt = np.c_[np.ones(len(treino)), ((treino[cols] - mu) / sd).clip(-5, 5).values]
        yt = (treino["y"] > 0).astype(float).values
        w = logit_fit(Xt, yt, lam=len(treino) * 0.01)
        Xs = np.c_[np.ones(len(teste)), ((teste[cols] - mu) / sd).clip(-5, 5).values]
        p = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        base = yt.mean()
        pred.append(pd.DataFrame({"p": p, "base": base, "y": teste["y"].values, "a": teste["a"].values, "cls": teste["cls"].values}, index=teste.index))
        L.conta()
    P = pd.concat(pred)
    yb = (P["y"] > 0).astype(int).values
    res = dict(nome=nome, H=H_lbl, n=len(P), de=str(P.index[0].date()), auc=auc(P["p"].values, yb), base=float(yb.mean()))
    res["auc_anos"] = {str(k): round(auc(g["p"].values, (g["y"] > 0).astype(int).values), 3) for k, g in P.groupby(P.index.year)}
    for mg in margens:
        lado = np.where(P["p"] > P["base"] + mg, 1, np.where(P["p"] < P["base"] - mg, -1, 0))
        sel = lado != 0
        custo = P["cls"].map(custo_de).values
        pnl = lado[sel] * P["y"].values[sel] - custo[sel]
        bruto = lado[sel] * P["y"].values[sel]
        hit = float((lado[sel] * P["y"].values[sel] > 0).mean()) if sel.any() else np.nan
        base_dir = float(np.mean(np.where(lado[sel] > 0, P["base"].values[sel], 1 - P["base"].values[sel]))) if sel.any() else np.nan
        t, nd = L.t_por_dia(pnl, P.index[sel]) if sel.sum() > 10 else (0.0, 0)
        tb, _ = L.t_por_dia(bruto, P.index[sel]) if sel.sum() > 10 else (0.0, 0)
        res[f"m{mg}"] = dict(ops=int(sel.sum()), acerto=hit, base_dir=base_dir, bruto=float(bruto.mean()) if sel.any() else np.nan,
                             liq=float(pnl.mean()) if sel.any() else np.nan, t_liq=t, t_bruto=tb)
    return res, P


ml = []
# E2a: cripto e memecoins em 1 h
ativos = L.CLASSES["crypto"] + L.CLASSES["meme"]
btc = L.horario("BTC")
btc_r = pd.DataFrame({"btc1": np.log(btc["c"]).diff(), "btc24": np.log(btc["c"]).diff(24)})
for Hh in (4, 24):
    partes = []
    for a in ativos:
        h = L.horario(a)
        f = L.funding(a)
        extra = btc_r if a != "BTC" else None
        F = feats(h["c"], h["v"], 24, extra)
        if f is not None:
            fz = f[~f.index.duplicated()].reindex(F.index, method="ffill")
            F["fund"] = (fz - fz.rolling(24 * 30).mean()) / fz.rolling(24 * 30).std()
        else:
            F["fund"] = 0.0
        if a == "BTC":
            F["btc1"] = F["r1"]; F["btc24"] = F["r24"]
        F["y"] = np.log(h["c"]).shift(-Hh) - np.log(h["c"])
        F["yfim"] = F.index + pd.Timedelta(hours=Hh)
        F["yfim"] = F["yfim"].dt.to_period("M")
        F["a"] = a; F["cls"] = L.CLS_DE[a]
        partes.append(F)
    painel = pd.concat(partes).sort_index()
    painel["yfim"] = painel["yfim"].astype("period[M]")
    r, P = walk_forward(painel, "cripto+memecoins, 1 h", f"{Hh} h", 4, L.CUSTO)
    ml.append(r); P.to_csv(L.OUT + f"/e2_pred_cripto_{Hh}h.csv")
    print(r)

# E2b: diário, todos os ativos, 5 dias
partes = []
for cls, lst in L.CLASSES.items():
    for a in lst:
        d = L.diario(a); d = d[d["c"] > 0]
        F = feats(d["c"], d["v"], 1)
        F["y"] = np.log(d["c"]).shift(-5) - np.log(d["c"])
        F["yfim"] = (F.index + pd.Timedelta(days=7)).to_period("M")
        F["a"] = a; F["cls"] = cls
        partes.append(F)
painel = pd.concat(partes).sort_index()
r, P = walk_forward(painel, "todos os ativos, diário", "5 dias", 1, L.CUSTO)
ml.append(r); P.to_csv(L.OUT + "/e2_pred_diario_5d.csv")
print(r)

pd.DataFrame(out).to_csv(L.OUT + "/e1_extremos.csv", index=False)
L.salva("e2_ml.json", ml)
print(pd.DataFrame(out).round(4).to_string())
print("simulações:", L.N_TESTES)
L.salva("e_n.json", {"n": L.N_TESTES})
