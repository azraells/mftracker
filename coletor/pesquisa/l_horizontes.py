"""Horizontes de 12 a 72 h, compra e venda (26/11/2026).
Universo: cripto e memes em 1 h (cache da Binance) nos prazos de 12, 24, 48 e 72 h; B3, EUA e dólar no diário (24/48/72 h =
1/2/3 pregões; sem 12 h). Uma decisão a cada prazo por ativo (sem sobreposição). t agrupado por dia pela SOMA (lib.t_por_dia_soma).
Escolha (margem, N) só na 1ª metade: cripto/memes até 2023; diário até 2019. Conferência: cripto/memes 2024–26; diário 2020–26.
Métodos:
  M1 modelo de aprendizado (ml_core no 1 h; logística diária no diário), walk-forward mensal, embargo = prazo.
     Compra se p ≥ base + margem, venda se p ≤ base − margem; margem ∈ {0,03; 0,05; 0,08; 0,10}.
  M2 tendência nos dois lados: compra se o retorno dos últimos N períodos > 0, venda se < 0; N ∈ {prazo, 4 × prazo}.
  M3 reversão nos dois lados: o contrário do M2, só quando o movimento de N passou de 1 desvio.
  M4 M1 a favor do regime do painel: compra só com regime dentro, venda só com regime fora (cripto: média 50;
     memes: rompimento 20/10; B3: momento 21). EUA e dólar não têm regra no painel: M4 não se aplica.
Custos de ida e volta: cripto 0,20% e 0,30%; memes 0,20% e 0,60%; B3, EUA e dólar 0,10%.
Venda: em perpétuo recebe +funding (positivo = vendido recebe), somado no período; B3 paga aluguel de 2% ao ano;
EUA, 1% ao ano; dólar sem custo de carregamento (simplificação).
Placebo do M1: rótulos deslocados 45 a 120 dias por ativo no 1 h; no diário, 300 a 600 pregões para a frente (prazo de 24 h).
Uso: python l_horizontes.py  → saida/l_horizontes.json"""
import itertools, json, math, os
import numpy as np, pandas as pd
import lib as L, ml_core as M

HORAS = (12, 24, 48, 72)
CUSTOS = {"crypto": (0.002, 0.003), "meme": (0.002, 0.006), "b3": (0.001,), "us": (0.001,), "fx": (0.001,)}
CORTE = {"1h": pd.Timestamp("2024-01-01"), "1d": pd.Timestamp("2020-01-01")}
MARGENS = (0.03, 0.05, 0.08, 0.10)
res = {"linhas": [], "placebo": {}, "escolhas": {}}


def grade_tempo(idx, H, diario):
    """True nos pontos de decisão: a cada H horas (1 h) ou a cada H/24 pregões (diário), contados por ativo."""
    if diario:
        return None
    return ((idx.asi8 // 3_600_000_000_000) % H == 0)


def regime_diario(c, cls):
    """1 = dentro, 0 = fora, decidido no fechamento de ontem (vale hoje). c: fechamentos diários."""
    if cls == "crypto":
        st = (c > c.rolling(50).mean()).astype(float)
    elif cls == "b3":
        st = (c / c.shift(21) - 1 > 0).astype(float)
    else:
        hi = c.rolling(20).max().shift(1); lo = c.rolling(10).min().shift(1); s = np.zeros(len(c)); pos = 0.0
        for i in range(len(c)):
            if np.isfinite(hi.iloc[i]) and c.iloc[i] > hi.iloc[i]: pos = 1.0
            elif np.isfinite(lo.iloc[i]) and c.iloc[i] < lo.iloc[i]: pos = 0.0
            s[i] = pos
        st = pd.Series(s, index=c.index)
    return st.shift(1)


def banca(D, liq_col, frac=0.10):
    """R$ 1.000, 10% da banca por operação; operações abertas no mesmo instante dividem a mesma banca."""
    if D.empty:
        return dict(final=1000.0, ao_ano=0.0, maior_queda=0.0)
    g = D.groupby(D.index)[liq_col].apply(lambda x: (x * frac).sum()).clip(lower=-0.99)
    eq = 1000 * (1 + g).cumprod()
    anos = max((g.index[-1] - g.index[0]).days / 365, 0.25)
    return dict(final=float(eq.iloc[-1]), ao_ano=float((eq.iloc[-1] / 1000) ** (1 / anos) - 1), maior_queda=float((eq / eq.cummax() - 1).min()))


def resumo(D, cls, meses):
    """D: operações com colunas lado (+1/−1), y (retorno log do ativo no prazo), carrego (a favor da posição), base_pos."""
    if len(D) < 10:
        return dict(ops=len(D))
    br = D.lado * D.y + D.carrego
    out = dict(ops=len(D), ops_mes=len(D) / max(meses, 1), acerto=float((br > 0).mean()), base=float(D.base_pos.mean()),
               bruto=float(br.mean()), t_bruto=L.t_por_dia_soma(br.values, D.index)[0], pior_op=float(br.min()))
    for c in CUSTOS[cls]:
        liq = br - c
        out[f"liq_{c*100:.2f}%"] = float(liq.mean()); out[f"t_{c*100:.2f}%"] = L.t_por_dia_soma(liq.values, D.index)[0]
    c0 = CUSTOS[cls][0]
    m = (br - c0).groupby(D.index.to_period("M")).mean()
    out["pior_mes"] = dict(mes=str(m.idxmin()), media_por_op=float(m.min()))
    return out


def avalia(D, cls, nome, metodo, H, mercado, corte, extra=None):
    """Registra compras, vendas e juntas, em 1ª e 2ª metade; banca só na 2ª metade (com o custo mais baixo)."""
    c0 = CUSTOS[cls][0]
    for lado_nome, S in (("compras", D[D.lado > 0]), ("vendas", D[D.lado < 0]), ("juntas", D)):
        S1, S2 = S[S.index < corte], S[S.index >= corte]
        m1 = max((S1.index.max() - S1.index.min()).days / 30.4, 1) if len(S1) else 1
        m2 = max((S2.index.max() - S2.index.min()).days / 30.4, 1) if len(S2) else 1
        S2b = S2.assign(liq=S2.lado * S2.y + S2.carrego - c0)
        res["linhas"].append(dict(metodo=metodo, prazo_h=H, mercado=mercado, lado=lado_nome, regra=nome, **(extra or {}),
                                  primeira=resumo(S1, cls, m1), segunda=resumo(S2, cls, m2), banca_2a=banca(S2b, "liq")))
        L.conta(2)


# ---------------------------------------------------------------- 1 h: cripto e memes
for H in HORAS:
    P = M.painel(H)
    Q = M.walk_forward(P, H)
    if H == 24:
        Qp = M.walk_forward(P, H, placebo="desloca", seed=41)
        res["placebo"]["cripto+memes 24 h (rótulos deslocados)"] = M.auc(Qp.p, (Qp.y > 0).astype(int))
    res["placebo"][f"cripto+memes {H} h AUC real"] = M.auc(Q.p, (Q.y > 0).astype(int))
    partes = []
    for a, g in Q.groupby("a"):
        g = g.sort_index().copy()
        cls = L.CLS_DE[a]
        cd = g.c_.groupby(g.index.normalize()).last()
        g["dentro"] = regime_diario(cd, cls).reindex(g.index.normalize()).values
        f = L.funding(a)
        fh = (f[~f.index.duplicated()].sort_index().reindex(g.index, method="ffill").fillna(0.0) / 8) if f is not None else pd.Series(0.0, index=g.index)
        g["fund_H"] = fh.rolling(H).sum().shift(-H).values  # funding pago/recebido durante as próximas H horas
        lr = np.log(g.c_)
        sd1 = lr.diff().rolling(24 * 30).std()
        for N in (H, 4 * H):
            g[f"ret{N}"] = lr.diff(N); g[f"z{N}"] = g[f"ret{N}"] / (sd1 * math.sqrt(N))
        g = g[grade_tempo(g.index, H, False)]
        partes.append(g)
    G = pd.concat(partes).sort_index()
    G = G[G.y.notna()]
    for mercado, cls in (("Cripto", "crypto"), ("Memecoins", "meme")):
        X = G[G.cls == cls]
        corte = CORTE["1h"]
        base_up = float((X[X.index < corte].y > 0).mean())

        def ops(sinal):
            lado = np.sign(sinal).astype(float)
            D = X.assign(lado=lado)[lado != 0].copy()
            D["carrego"] = np.where(D.lado < 0, D.fund_H.fillna(0.0), -D.fund_H.fillna(0.0) * 0)  # compra no spot: sem funding
            D["base_pos"] = np.where(D.lado > 0, base_up, 1 - base_up)
            return D
        # M1 e M4: margem escolhida na 1ª metade (maior líquido médio com o custo mais baixo, ≥ 30 operações)
        for metodo, filtro in (("M1 modelo", None), ("M4 modelo a favor do regime", "regime")):
            esc, melhor = None, -9
            for mg in MARGENS:
                s = np.where(X.p >= X.base + mg, 1, np.where(X.p <= X.base - mg, -1, 0))
                if filtro:
                    s = np.where(((s > 0) & (X.dentro == 1)) | ((s < 0) & (X.dentro == 0)), s, 0)
                D = ops(s); D1 = D[D.index < corte]
                L.conta()
                if len(D1) >= 30:
                    v = float((D1.lado * D1.y + D1.carrego - CUSTOS[cls][0]).mean())
                    if v > melhor: melhor, esc = v, mg
            esc = esc if esc is not None else 0.05
            res["escolhas"][f"{metodo} {H} h {mercado}"] = esc
            s = np.where(X.p >= X.base + esc, 1, np.where(X.p <= X.base - esc, -1, 0))
            if filtro:
                s = np.where(((s > 0) & (X.dentro == 1)) | ((s < 0) & (X.dentro == 0)), s, 0)
            avalia(ops(s), cls, f"margem {esc}", metodo, H, mercado, corte, dict(margem=esc))
            if metodo.startswith("M4") and H == 48:  # conferência do caso "48 h só compras em regime de alta" (todas as margens)
                for mg in MARGENS:
                    s2 = np.where((X.p >= X.base + mg) & (X.dentro == 1), 1, 0)
                    avalia(ops(s2), cls, f"só compras em regime de alta, margem {mg}", "conferência 48 h", H, mercado, corte, dict(margem=mg))
        # M2 e M3: N escolhido na 1ª metade
        for metodo in ("M2 tendência", "M3 reversão"):
            esc, melhor = None, -9
            for N in (H, 4 * H):
                s = np.sign(X[f"ret{N}"]) if metodo.startswith("M2") else -np.sign(X[f"ret{N}"]) * (X[f"z{N}"].abs() >= 1)
                D = ops(s.fillna(0).values); D1 = D[D.index < corte]
                L.conta()
                if len(D1) >= 30:
                    v = float((D1.lado * D1.y + D1.carrego - CUSTOS[cls][0]).mean())
                    if v > melhor: melhor, esc = v, N
            esc = esc or H
            res["escolhas"][f"{metodo} {H} h {mercado}"] = esc
            s = np.sign(X[f"ret{esc}"]) if metodo.startswith("M2") else -np.sign(X[f"ret{esc}"]) * (X[f"z{esc}"].abs() >= 1)
            avalia(ops(s.fillna(0).values), cls, f"N = {esc} h", metodo, H, mercado, corte, dict(N=esc))
    print("1 h", H, "pronto", flush=True)


# ---------------------------------------------------------------- diário: B3, EUA, dólar
def feats_d(d):
    c = d.c; lr = np.log(c); F = {}
    for k in (1, 5, 20, 60, 120, 250):
        F[f"r{k}"] = lr.diff(k)
    F["vrat"] = np.log(lr.diff().rolling(5).std() / lr.diff().rolling(60).std()); F["vol"] = np.log(lr.diff().rolling(60).std())
    F["rsi"] = (M.rsi(c) - 50) / 50; F["dma20"] = lr - lr.rolling(20).mean(); F["dma100"] = lr - lr.rolling(100).mean()
    F["volu"] = np.log((d.v + 1) / (d.v.rolling(20).mean() + 1))
    dw = c.index.dayofweek; F["ds"] = np.sin(2 * np.pi * dw / 5); F["dc"] = np.cos(2 * np.pi * dw / 5)
    return pd.DataFrame(F, index=c.index)


DCLS = {"b3": L.CLASSES["b3"], "us": L.CLASSES["us"], "fx": L.CLASSES["fx"]}
NOME = {"b3": "B3", "us": "EUA", "fx": "Dólar"}
for H in (24, 48, 72):
    k = H // 24
    partes = []
    for cls, ativos in DCLS.items():
        for a in ativos:
            d = L.diario(a); d = d[d.c > 0]
            F = feats_d(d)
            F["y"] = np.log(d.c).shift(-k) - np.log(d.c)
            F["a"] = a; F["cls"] = cls; F["c_"] = d.c
            F["dentro"] = regime_diario(d.c, cls).values if cls == "b3" else np.nan
            for N in (k, 4 * k):
                F[f"ret{N}"] = np.log(d.c).diff(N)
                F[f"z{N}"] = F[f"ret{N}"] / (np.log(d.c).diff().rolling(60).std() * math.sqrt(N))
            F["pos"] = np.arange(len(F))
            partes.append(F)
    P = pd.concat(partes).sort_index().replace([np.inf, -np.inf], np.nan)
    cols = ["r1", "r5", "r20", "r60", "r120", "r250", "vrat", "vol", "rsi", "dma20", "dma100", "volu", "ds", "dc"]
    P = P.dropna(subset=cols)
    meses = P.index.to_period("M"); ums = sorted(meses.unique())
    for placebo in ((False, True) if H == 24 else (False,)):
        y_all = P.y.values.copy()
        if placebo:
            rng = np.random.default_rng(43)
            for a in P.a.unique():
                ix = np.where(P.a.values == a)[0]
                # para a FRENTE e além da janela mais longa (250 pregões): deslocar para trás põe um retorno passado no rótulo,
                # que já está dentro de r60/r120/r250, e contamina o placebo (foi o que deu AUC 0,547 na 1ª rodada)
                y_all[ix] = np.roll(P.y.values[ix], -int(rng.integers(300, 600)))
        pred = []
        for i, m in enumerate(ums):
            if i < 24:
                continue
            tr = (meses < m) & (P.index < m.start_time - pd.Timedelta(days=2 * k + 2)) & np.isfinite(y_all)
            if tr.sum() < 2000:
                continue
            te = (meses == m)
            Xt = P.loc[tr, cols]; mu, sd = Xt.mean(), Xt.std().replace(0, 1)
            w = M.logit_fit(np.c_[np.ones(tr.sum()), ((Xt - mu) / sd).clip(-5, 5).values], (y_all[tr] > 0).astype(float), lam=tr.sum() * 0.01)
            p = 1 / (1 + np.exp(-np.c_[np.ones(te.sum()), ((P.loc[te, cols] - mu) / sd).clip(-5, 5).values] @ w))
            pred.append(P.loc[te].assign(p=p, base=float((y_all[tr] > 0).mean()), y_pl=y_all[te]))
            L.conta()
        PPx = pd.concat(pred)
        if placebo:
            res["placebo"]["diário 24 h (rótulos deslocados)"] = M.auc(PPx.p, (PPx.y_pl > 0).astype(int))
            continue
        res["placebo"][f"diário {H} h AUC real"] = M.auc(PPx.p, (PPx.y > 0).astype(int))
        PPreal = PPx
    PP = PPreal[PPreal.y.notna() & (PPreal.pos % k == 0)]  # uma decisão a cada k pregões por ativo
    corte = CORTE["1d"]
    for cls in ("b3", "us", "fx"):
        X = PP[PP.cls == cls]
        if X.empty:
            continue
        base_up = float((X[X.index < corte].y > 0).mean())
        aluguel = {"b3": 0.02, "us": 0.01, "fx": 0.0}[cls]

        def ops(sinal):
            lado = np.sign(sinal).astype(float)
            D = X.assign(lado=lado)[lado != 0].copy()
            D["carrego"] = np.where(D.lado < 0, -aluguel * k / 252, 0.0)
            D["base_pos"] = np.where(D.lado > 0, base_up, 1 - base_up)
            return D
        for metodo, filtro in (("M1 modelo", None), ("M4 modelo a favor do regime", "regime")):
            if filtro and cls != "b3":
                continue
            esc, melhor = None, -9
            for mg in MARGENS:
                s = np.where(X.p >= X.base + mg, 1, np.where(X.p <= X.base - mg, -1, 0))
                if filtro:
                    s = np.where(((s > 0) & (X.dentro == 1)) | ((s < 0) & (X.dentro == 0)), s, 0)
                D = ops(s); D1 = D[D.index < corte]
                L.conta()
                if len(D1) >= 30:
                    v = float((D1.lado * D1.y + D1.carrego - CUSTOS[cls][0]).mean())
                    if v > melhor: melhor, esc = v, mg
            esc = esc if esc is not None else 0.05
            res["escolhas"][f"{metodo} {H} h {NOME[cls]}"] = esc
            s = np.where(X.p >= X.base + esc, 1, np.where(X.p <= X.base - esc, -1, 0))
            if filtro:
                s = np.where(((s > 0) & (X.dentro == 1)) | ((s < 0) & (X.dentro == 0)), s, 0)
            avalia(ops(s), cls, f"margem {esc}", metodo, H, NOME[cls], corte, dict(margem=esc))
        for metodo in ("M2 tendência", "M3 reversão"):
            esc, melhor = None, -9
            for N in (k, 4 * k):
                s = np.sign(X[f"ret{N}"]) if metodo.startswith("M2") else -np.sign(X[f"ret{N}"]) * (X[f"z{N}"].abs() >= 1)
                D = ops(s.fillna(0).values); D1 = D[D.index < corte]
                L.conta()
                if len(D1) >= 30:
                    v = float((D1.lado * D1.y + D1.carrego - CUSTOS[cls][0]).mean())
                    if v > melhor: melhor, esc = v, N
            esc = esc or k
            res["escolhas"][f"{metodo} {H} h {NOME[cls]}"] = esc
            s = np.sign(X[f"ret{esc}"]) if metodo.startswith("M2") else -np.sign(X[f"ret{esc}"]) * (X[f"z{esc}"].abs() >= 1)
            avalia(ops(s.fillna(0).values), cls, f"N = {esc} pregão(ões)", metodo, H, NOME[cls], corte, dict(N=esc))
    print("diário", H, "pronto", flush=True)

# ---------------------------------------------------------------- contador, Bonferroni e resumo
anterior = 6777  # acumulado do dia até o k_vendas
total = anterior + L.N_TESTES
tmin = next(t / 1000 for t in range(1000, 8000) if L.p_bilateral(t / 1000) <= 0.05 / total)
res["testes"] = dict(novos=L.N_TESTES, total_do_dia=total, t_minimo_bonferroni=tmin)
pos = []
for l in res["linhas"]:
    a, b = l["primeira"], l["segunda"]
    cls_c = [k for k in a if k.startswith("liq_")]
    for kc in cls_c:
        if a.get(kc, -1) > 0 and b.get(kc, -1) > 0:
            pos.append(dict(metodo=l["metodo"], prazo_h=l["prazo_h"], mercado=l["mercado"], lado=l["lado"], regra=l["regra"], custo=kc,
                            liq1=a[kc], liq2=b[kc], t2=b["t_" + kc[4:]], ops2=b["ops"], bonferroni=tmin, passa=b["t_" + kc[4:]] >= tmin))
res["positivos_nas_duas_metades"] = pos
L.salva("l_horizontes.json", res)
print(json.dumps(res["testes"]), "| positivos nas duas metades:", len(pos))
