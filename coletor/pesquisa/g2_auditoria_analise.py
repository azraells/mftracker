"""Auditoria, parte 2: análises sobre as previsões auditadas (g_pred_4h.pkl e g_pred_24h.pkl).
A1 t honesto (por dia; só sem sobreposição) · A4 margem escolhida só em 2022–2023 · A5 modelo × reversão simples
B7 cenários de custo + deslizamento e custo de equilíbrio · B8 entrada por ordem limitada (seleção adversa)
C9 filtro de regime (média de 50 dias) e só compras · C10 tamanho pela confiança dentro do regime de cripto
C11 saída dinâmica (quando a probabilidade volta para a base ou no prazo máximo), escolhida em 2022–2023."""
import math, json
import numpy as np, pandas as pd
import lib as L, ml_core as M

CORTE = pd.Timestamp("2024-01-01")
CUSTOS = (0.0004, 0.001, 0.0015, 0.002, 0.003)
res = {}
Q = {H: pd.read_pickle(L.OUT + f"/g_pred_{H}h.pkl") for H in (4, 24)}
for H in (4, 24):
    Q[H]["slip"] = Q[H].groupby("a").apply(lambda g: ((g.h_ - g.l_) / g.c_).median() * 0.1).reindex(Q[H].a).values  # por lado


def resumo(R, custo_rt=None):
    if len(R) < 5:
        return dict(ops=len(R))
    t_dia, nd = L.t_por_dia(R.bruto.values, R.index)
    d = dict(ops=len(R), dias=nd, acerto=float((R.bruto > 0).mean()), bruto=float(R.bruto.mean()), t_dia=t_dia, t_simples=L.t_stat(R.bruto.values))
    if custo_rt is not None:
        liq = R.bruto - custo_rt
        d["liq"] = float(liq.mean()); d["t_liq_dia"] = L.t_por_dia(liq.values, R.index)[0]
    return d


# ---------------------------------------------------------------- A1
a1 = {}
for H in (4, 24):
    a1[f"{H}h AUC"] = M.auc(Q[H].p, (Q[H].y > 0).astype(int))
    a1[f"{H}h AUC por ano"] = {str(k): round(M.auc(g.p, (g.y > 0).astype(int)), 3) for k, g in Q[H].groupby(Q[H].index.year)}
    for mg in (0.05, 0.10):
        todas = M.operacoes(Q[H], mg)
        sem = M.operacoes(Q[H], mg, nao_sobrepor=H)
        a1[f"{H}h margem {mg} todas as horas"] = resumo(todas)
        a1[f"{H}h margem {mg} sem sobreposição"] = resumo(sem)
        L.conta(2)
res["A1"] = a1

# ---------------------------------------------------------------- A4
a4 = {}
for H in (4, 24):
    grade = []
    for mg in np.round(np.arange(0.02, 0.151, 0.01), 2):
        R = M.operacoes(Q[H], mg, nao_sobrepor=H)
        R["custo"] = R.cls.map(L.CUSTO) + 2 * R.slip
        R1, R2 = R[R.index < CORTE], R[R.index >= CORTE]
        grade.append(dict(margem=float(mg), ops1=len(R1), liq1=float((R1.bruto - R1.custo).mean()) if len(R1) else np.nan,
                          ops2=len(R2), bruto2=float(R2.bruto.mean()) if len(R2) else np.nan, liq2=float((R2.bruto - R2.custo).mean()) if len(R2) else np.nan,
                          t2=L.t_por_dia((R2.bruto - R2.custo).values, R2.index)[0] if len(R2) > 10 else 0.0))
        L.conta()
    g = pd.DataFrame(grade)
    elegiveis = g[g.ops1 >= 100]
    esc = elegiveis.loc[elegiveis.liq1.idxmax()]
    a4[f"{H}h"] = dict(escolhida=float(esc.margem), liq_2022_23=float(esc.liq1), ops_2024_26=int(esc.ops2), bruto_2024_26=float(esc.bruto2),
                       liq_2024_26=float(esc.liq2), t_2024_26=float(esc.t2), grade=g.round(5).to_dict("records"))
res["A4"] = a4

# ---------------------------------------------------------------- A5
a5 = {}
for H in (4, 24):
    D = Q[H].copy()
    D["s"] = D.p - D.base
    D["rev4"] = -D.r4; D["rev1"] = -D.r1
    a5[f"{H}h correlação modelo × −retorno 4 h"] = float(D.s.corr(D.rev4))
    a5[f"{H}h correlação modelo × −retorno 1 h"] = float(D.s.corr(D.rev1))
    a5[f"{H}h AUC só da reversão de 4 h"] = M.auc(D.rev4, (D.y > 0).astype(int))
    a5[f"{H}h AUC só da reversão de 1 h"] = M.auc(D.rev1, (D.y > 0).astype(int))
    resid = np.empty(len(D))
    for m, g in D.groupby(D.index.to_period("M")):
        X = np.c_[np.ones(len(g)), g.rev4.values, g.rev1.values]
        b = np.linalg.lstsq(X, g.s.values, rcond=None)[0]
        resid[D.index.to_period("M") == m] = g.s.values - X @ b
    D["resid"] = resid
    a5[f"{H}h AUC do modelo sem a reversão (resíduo)"] = M.auc(D.resid, (D.y > 0).astype(int))
    a5[f"{H}h R² da reversão sobre o modelo"] = float(1 - np.var(resid) / np.var(D.s))
    Dn = D[D.index.hour % H == 0] if H == 4 else D[D.index.hour == 0]
    q = Dn.resid.quantile([0.05, 0.95])
    sel = Dn[(Dn.resid <= q.iloc[0]) | (Dn.resid >= q.iloc[1])]
    lado = np.sign(sel.resid); br = lado * sel.y
    a5[f"{H}h resíduo nos 10% extremos: bruto por operação"] = float(br.mean())
    a5[f"{H}h resíduo nos 10% extremos: t por dia"] = L.t_por_dia(br.values, sel.index)[0]
    q2 = Dn.rev4.quantile([0.05, 0.95])
    sel2 = Dn[(Dn.rev4 <= q2.iloc[0]) | (Dn.rev4 >= q2.iloc[1])]
    br2 = np.sign(sel2.rev4) * sel2.y
    a5[f"{H}h reversão simples nos 10% extremos: bruto por operação"] = float(br2.mean())
    a5[f"{H}h reversão simples nos 10% extremos: t por dia"] = L.t_por_dia(br2.values, sel2.index)[0]
    L.conta(4)
res["A5"] = a5

# ---------------------------------------------------------------- B7
b7 = {}
for H in (4, 24):
    for mg in (0.05, 0.10):
        R = M.operacoes(Q[H], mg, nao_sobrepor=H)
        for parte, S in (("2022–2026", R), ("2024–2026", R[R.index >= CORTE])):
            slip_rt = 2 * S.slip
            d = dict(ops=len(S), bruto=float(S.bruto.mean()), deslizamento_ida_volta=float(slip_rt.mean()),
                     custo_equilibrio=float((S.bruto - slip_rt).mean()))
            for c in CUSTOS:
                liq = S.bruto - slip_rt - c
                d[f"liq_{c*100:.2f}%"] = float(liq.mean()); d[f"t_{c*100:.2f}%"] = L.t_por_dia(liq.values, S.index)[0]
            b7[f"{H}h margem {mg} {parte}"] = d
            L.conta(len(CUSTOS))
res["B7"] = b7

# ---------------------------------------------------------------- B8
b8 = {}
for H in (4, 24):
    D = Q[H]
    por_ativo = {a: g.sort_index() for a, g in D.groupby("a")}
    for mg in (0.05, 0.10):
        R = M.operacoes(D, mg, nao_sobrepor=H)
        for x in (0.0005, 0.001):
            for k in (1, 2):
                linhas = []
                for ts, r in R.iterrows():
                    g = por_ativo[r.a]
                    j = g.index.searchsorted(ts)
                    if j + max(k, H) >= len(g):
                        continue
                    futuro = g.iloc[j + 1:j + 1 + k]
                    saida = g.c_.iloc[j + H] if H > 0 else np.nan
                    lim = r.c_ * (1 - x * r.lado)
                    pegou = (futuro.l_.min() <= lim) if r.lado > 0 else (futuro.h_.max() >= lim)
                    linhas.append(dict(pegou=bool(pegou), mercado=r.lado * math.log(saida / r.c_), limite=r.lado * math.log(saida / lim)))
                T = pd.DataFrame(linhas)
                f = T[T.pegou]; nf = T[~T.pegou]
                b8[f"{H}h margem {mg} limite {x*100:.2f}% em {k} h"] = dict(
                    sinais=len(T), executadas=float(T.pegou.mean()), bruto_a_mercado_todas=float(T.mercado.mean()),
                    bruto_a_mercado_das_executadas=float(f.mercado.mean()), bruto_a_mercado_das_perdidas=float(nf.mercado.mean()) if len(nf) else None,
                    bruto_com_limite_executadas=float(f.limite.mean()))
                L.conta()
res["B8"] = b8

# ---------------------------------------------------------------- C9
c9 = {}
for H in (4, 24):
    D = Q[H].copy()
    dia = D.groupby(["a", D.index.normalize()]).c_.last().rename("cd").reset_index().rename(columns={"level_1": "dt"})
    dia.columns = ["a", "dt", "cd"]
    dia["sma50"] = dia.groupby("a").cd.transform(lambda s: s.rolling(50).mean())
    dia["alta"] = (dia.cd > dia.sma50).astype(float)
    dia["dt_uso"] = dia.dt + pd.Timedelta(days=1)  # o regime de ontem vale hoje
    reg = dia.set_index(["a", "dt_uso"]).alta
    D["alta"] = [reg.get((a, d), np.nan) for a, d in zip(D.a, D.index.normalize())]
    for mg in (0.05, 0.10):
        R = M.operacoes(D, mg, nao_sobrepor=H)
        R["custo"] = R.cls.map(L.CUSTO) + 2 * R.slip
        R = R[R.alta.notna()]
        for nome, S in (("modelo sozinho", R), ("só a favor do regime", R[((R.lado > 0) & (R.alta == 1)) | ((R.lado < 0) & (R.alta == 0))]),
                        ("só compras", R[R.lado > 0]), ("só compras em regime de alta", R[(R.lado > 0) & (R.alta == 1)])):
            for parte, SS in (("2022–23", S[S.index < CORTE]), ("2024–26", S[S.index >= CORTE])):
                c9[f"{H}h margem {mg} · {nome} · {parte}"] = resumo(SS.assign(bruto=SS.bruto), custo_rt=SS.custo.mean() if len(SS) else None)
            L.conta(2)
res["C9"] = c9

# ---------------------------------------------------------------- C10
c10 = {}
D = Q[24]
cr = D[D.cls == "crypto"]
fim_dia = cr[cr.index.hour == 23]  # previsão feita no fechamento do dia (candle das 23 h fecha à meia-noite)
series = {}
for a, g in fim_dia.groupby("a"):
    g = g.copy()
    g.index = g.index.normalize()
    cd = g.c_
    sma = cd.rolling(50).mean()
    reg = (cd > sma).astype(float)
    r_next = cd.shift(-1) / cd - 1
    conf = np.clip((g.p - 0.5) / 0.1, 0, 1)
    cx = L.caixa_para(cd.index, "crypto")
    for nome, w in (("segurar", pd.Series(1.0, index=cd.index)), ("regime (média 50)", reg), ("regime × confiança do modelo", reg * conf),
                    ("só confiança do modelo", conf)):
        giro = w.diff().abs().fillna(w.iloc[0])
        ret = (w * r_next + (1 - w) * cx - giro * L.CUSTO["crypto"] / 2).dropna()
        series.setdefault(nome, {})[a] = ret
for nome, dd in series.items():
    S = pd.concat(dd, axis=1).mean(axis=1).dropna()
    S = S[S.index >= pd.Timestamp("2022-03-01")]
    rf = L.caixa_para(S.index, "crypto")
    m = L.metricas_serie(S.values, 365, rf)
    c10[nome] = dict(**m, de=str(S.index[0].date()), ate=str(S.index[-1].date()))
    series[nome]["_carteira"] = S
ref = series["regime (média 50)"]["_carteira"]
for nome in series:
    if nome != "regime (média 50)":
        S = series[nome]["_carteira"].reindex(ref.index).fillna(0)
        c10[nome]["t_vs_regime"] = L.t_dif_sharpe_mensal(S.values, ref.values, ref.index, L.caixa_para(ref.index, "crypto"))
    L.conta(4)
res["C10"] = c10

# ---------------------------------------------------------------- C11
c11 = []
D = Q[4]
por_ativo = {a: g.sort_index() for a, g in D.groupby("a")}
for mg in (0.05, 0.08, 0.10):
    for hmax in (4, 8, 12, 24):
        ops = []
        for a, g in por_ativo.items():
            p = g.p.values; b = g.base.values; c = g.c_.values; idx = g.index
            custo = L.CUSTO[L.CLS_DE[a]] + 2 * float(g.slip.iloc[0])
            i, n = 0, len(g)
            while i < n - 1:
                lado = 1 if p[i] > b[i] + mg else (-1 if p[i] < b[i] - mg else 0)
                if lado == 0:
                    i += 1; continue
                j = i + 1
                while j < n - 1 and j - i < hmax and ((lado > 0 and p[j] > b[j]) or (lado < 0 and p[j] < b[j])):
                    j += 1
                if (idx[j] - idx[i]) > pd.Timedelta(hours=hmax + 6):  # buraco nos dados
                    i = j; continue
                ops.append((idx[i], lado * math.log(c[j] / c[i]) - custo, j - i))
                i = j
        T = pd.DataFrame(ops, columns=["t", "liq", "horas"]).set_index("t")
        T1, T2 = T[T.index < CORTE], T[T.index >= CORTE]
        c11.append(dict(margem=mg, h_max=hmax, ops1=len(T1), liq1=float(T1.liq.mean()), ops2=len(T2), liq2=float(T2.liq.mean()),
                        t2=L.t_por_dia(T2.liq.values, T2.index)[0], horas_medias=float(T.horas.mean())))
        L.conta()
g = pd.DataFrame(c11)
esc = g[g.ops1 >= 100].sort_values("liq1", ascending=False).iloc[0]
res["C11"] = dict(escolhida=esc.to_dict(), grade=g.round(5).to_dict("records"))

L.salva("g2_auditoria.json", res)
L.salva("g2_n.json", {"n": L.N_TESTES})
print(json.dumps(res, indent=1, default=float, ensure_ascii=False)[:20000])
