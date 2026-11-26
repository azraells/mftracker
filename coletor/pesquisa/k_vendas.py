"""Vendas a descoberto (26/11/2026). Mesma régua: escolha na 1ª metade (até 2023), conferência na 2ª (2024–2026),
t por dia sem sobreposição, Bonferroni com o contador do dia.

Parte A — lado vendido do modelo de 4 h (previsões fora da amostra do ml_core, com embargo):
  vendas com margem ≤ −0,03 / −0,05 / −0,08 em quatro recortes:
  (a) todas; (b) só com o ativo FORA do regime (cripto abaixo da média de 50 dias; memes fora do rompimento de 20);
  (c) só depois de alta de 24 h acima de +5% e de +10%; (d) as compras equivalentes (espelho de cada recorte).
  Funding: no perpétuo, com funding positivo o COMPRADO paga e o VENDIDO recebe. Em 4 h, a posição recebe ou paga metade
  da taxa de 8 h. Vendido: +funding; comprado: −funding.
Parte B — regime ao contrário: vendido no perpétuo (sem alavancagem, tamanho pela vol) enquanto o regime está FORA.
  Cripto: abaixo da média de N dias; memes: fora do rompimento de N dias; B3: momento de N dias negativo (aluguel de 2% ao ano
  no lugar do funding). Variações escolhidas na 1ª metade; conferência na 2ª; comparação com caixa, comprado/caixa e
  comprado-dentro + vendido-fora; maior queda, pior semana, pior sequência de 4 semanas e Monte Carlo em blocos de 4 semanas.
Uso: python k_vendas.py   (gera saida/k_vendas.json; refaz as previsões de 4 h se saida/g_pred_4h.pkl não existir)"""
import itertools, json, math, os
import numpy as np, pandas as pd
import lib as L, ml_core as M

CORTE = pd.Timestamp("2024-01-01")
CUSTOS_A = (0.001, 0.002, 0.003)
res = {"parte_A": {}, "parte_B": {}}


def funding_horario(a, idx):
    f = L.funding(a)
    if f is None:
        return pd.Series(0.0, index=idx)
    f = f[~f.index.duplicated()].sort_index()
    return f.reindex(idx, method="ffill").fillna(0.0)


# ================================================================ Parte A
arq = os.path.join(L.OUT, "g_pred_4h.pkl")
if os.path.exists(arq):
    Q = pd.read_pickle(arq)
else:
    Q = M.walk_forward(M.painel(4), 4); Q.to_pickle(arq)
partes = []
for a, g in Q.groupby("a"):
    g = g.sort_index().copy()
    g["r24"] = g.c_ / g.c_.shift(24) - 1  # índice horário contínuo por ativo (previsões em todas as horas)
    cd = g.c_.groupby(g.index.normalize()).last()
    if L.CLS_DE[a] == "crypto":
        fora_d = (cd <= cd.rolling(50).mean()).astype(float)
    else:
        hi = cd.rolling(20).max().shift(1); lo = cd.rolling(10).min().shift(1); st = np.zeros(len(cd)); pos = 0.0
        for i in range(len(cd)):
            if np.isfinite(hi.iloc[i]) and cd.iloc[i] > hi.iloc[i]: pos = 1.0
            elif np.isfinite(lo.iloc[i]) and cd.iloc[i] < lo.iloc[i]: pos = 0.0
            st[i] = pos
        fora_d = pd.Series(1 - st, index=cd.index)
    g["fora"] = fora_d.shift(1).reindex(g.index.normalize()).values  # regime de ontem vale hoje
    g["fund4h"] = funding_horario(a, g.index).values * 0.5
    partes.append(g)
Q = pd.concat(partes).sort_index()
Q = Q[Q.index.hour % 4 == 0]  # sem sobreposição: uma decisão a cada 4 h por ativo


def t_dia(x, idx):
    """t agrupado por dia usando a SOMA do dia (pesa cada operação igual; a média por dia do lib.t_por_dia pesa cada dia igual
    e pode até trocar o sinal quando dias com poucas operações são extremos)."""
    s = pd.Series(np.asarray(x, float), index=pd.to_datetime(idx)).groupby(lambda t: t.normalize()).sum()
    return L.t_stat(s.values)


def resumo_A(S, lado):
    if len(S) < 10:
        return dict(ops=len(S))
    br = lado * S.y
    fu = lado * -1 * S.fund4h  # vendido (lado −1) recebe +funding; comprado paga
    d = dict(ops=len(S), acerto=float((br > 0).mean()), bruto=float(br.mean()), t_bruto=t_dia(br.values, S.index),
             funding_medio_4h=float(fu.mean()))
    for c in CUSTOS_A:
        liq = br - c
        d[f"liq_{c*100:.2f}%"] = float(liq.mean()); d[f"t_{c*100:.2f}%"] = t_dia(liq.values, S.index)
    d["liq_0.20%_com_funding"] = float((br - 0.002 + fu).mean())
    d["pior_operacao"] = float(br.min())
    mes = (br - 0.002).groupby(S.index.to_period("M"))
    pm = mes.mean().idxmin()
    d["pior_mes"] = dict(mes=str(pm), media_por_op=float(mes.mean().min()), soma=float(mes.sum()[pm]), ops=int(mes.size()[pm]))
    return d


A = {}
for mg in (0.03, 0.05, 0.08):
    for lado, nome_lado in ((-1, "vendas"), (1, "compras")):
        sinal = (Q.p <= Q.base - mg) if lado < 0 else (Q.p >= Q.base + mg)
        S0 = Q[sinal]
        recortes = {"todas": S0,
                    "regime a favor (venda fora / compra dentro)": S0[S0.fora == (1 if lado < 0 else 0)],
                    "depois de 24 h ≥ +5% (venda) / ≤ −5% (compra)": S0[(S0.r24 >= 0.05) if lado < 0 else (S0.r24 <= -0.05)],
                    "depois de 24 h ≥ +10% (venda) / ≤ −10% (compra)": S0[(S0.r24 >= 0.10) if lado < 0 else (S0.r24 <= -0.10)]}
        for rc, S in recortes.items():
            A[f"{nome_lado} · margem {mg} · {rc}"] = {"1ª metade (até 2023)": resumo_A(S[S.index < CORTE], lado),
                                                      "2ª metade (2024–2026)": resumo_A(S[S.index >= CORTE], lado)}
            L.conta(2)
res["parte_A"] = A


# ================================================================ Parte B
def estado(c, fam, p):
    if fam == "media":
        return (c > c.rolling(p).mean()).astype(float).values
    if fam == "momento":
        return (c / c.shift(p) - 1 > 0).astype(float).values
    hi = c.rolling(p).max().shift(1); lo = c.rolling(p // 2).min().shift(1); st = np.zeros(len(c)); pos = 0.0
    for i in range(len(c)):
        if np.isfinite(hi.iloc[i]) and c.iloc[i] > hi.iloc[i]: pos = 1.0
        elif np.isfinite(lo.iloc[i]) and c.iloc[i] < lo.iloc[i]: pos = 0.0
        st[i] = pos
    return st


def serie(a, cls, fam, p, modo, tam):
    """modo: 'vendido_fora' | 'comprado_dentro' | 'longshort'. Retorno diário com caixa, funding/aluguel e custo."""
    d = L.diario(a); d = d[d.c > 0]
    c = d.c; r = c.pct_change().fillna(0).values
    per = 365 if cls in ("crypto", "meme") else 252
    rv = pd.Series(r).rolling(20).std().values * math.sqrt(per)
    meio = len(c) // 2; tv = float(np.nanmedian(rv[:meio])); rvn = np.nan_to_num(rv, nan=tv)
    dentro = np.nan_to_num(estado(c, fam, p))
    tamanho = np.minimum(1.0, tv / rvn) if tam == "vol" else np.ones(len(c))
    if modo == "vendido_fora":
        w = -(1 - dentro) * tamanho
    elif modo == "comprado_dentro":
        w = dentro * tamanho
    else:
        w = (dentro - (1 - dentro)) * tamanho
    w = np.r_[0.0, w[:-1]]  # decide no fechamento, vale a partir do dia seguinte
    cx = L.caixa_para(c.index, cls)
    if cls in ("crypto", "meme"):
        f = L.funding(a)
        fd = (f[~f.index.duplicated()].groupby(f.index.normalize()).sum().reindex(c.index).fillna(0.0).values) if f is not None else np.zeros(len(c))
        extra = np.where(w < 0, -w * fd, 0.0)  # vendido no perpétuo recebe +funding; o lado comprado é no spot (sem funding), como na pesquisa
    else:
        extra = np.where(w < 0, w * 0.02 / 252, 0.0)  # aluguel de 2% ao ano sobre a parte vendida
    giro = np.abs(np.diff(np.r_[0.0, w]))
    ret = w * r + (1 - np.abs(w)) * cx + extra - giro * L.CUSTO[cls] / 2  # a garantia da venda não rende juro (conservador)
    return pd.Series(ret, index=c.index), pd.Series(cx, index=c.index), meio


GRADE_B = {"crypto": [("media", 50), ("media", 100), ("media", 200)], "meme": [("donchian", 20), ("donchian", 55)],
           "b3": [("momento", 21), ("momento", 63)]}


def mc_blocos(ret, n=2000, bloco=4, banca=1000.0, seed=7):
    sem = ((1 + ret).groupby(ret.index.to_period("W")).prod() - 1).values
    rng = np.random.default_rng(seed); nb = 13
    ini = rng.integers(0, len(sem) - bloco + 1, size=(n, nb))
    am = np.stack([sem[ini[:, j][:, None] + np.arange(bloco)] for j in range(nb)], axis=1).reshape(n, -1)[:, :52]
    cam = banca * np.cumprod(1 + am, axis=1); fim = cam[:, -1]
    return dict(mediana=float(np.median(fim)), p5=float(np.percentile(fim, 5)), p95=float(np.percentile(fim, 95)),
                perde50=float(np.mean(cam.min(axis=1) <= banca / 2)), positivo=float(np.mean(fim > banca)))


def metricas(S, rf, per):
    m = L.metricas_serie(S.values, per, rf)
    sem = (1 + S).groupby(S.index.to_period("W")).prod() - 1
    q4 = (1 + sem).rolling(4).apply(np.prod, raw=True) - 1
    return dict(ao_ano=m["cagr"], sharpe=m["sharpe"], maior_queda=m["mdd"], pior_semana=float(sem.min()), pior_4_semanas=float(q4.min()))


B = {}
for cls in ("crypto", "meme", "b3"):
    ativos = L.CLASSES[cls] if cls != "b3" else ["PETR4", "VALE3", "ITUB4", "MGLU3", "BOVA11"]
    per = 365 if cls in ("crypto", "meme") else 252
    grade = []
    for (fam, p), tam in itertools.product(GRADE_B[cls], ("fixo", "vol")):
        ss, cxs, cortes = {}, {}, []
        for a in ativos:
            s, cx, meio = serie(a, cls, fam, p, "vendido_fora", tam); ss[a] = s; cxs[a] = cx; cortes.append(s.index[meio])
        S = pd.concat(ss, axis=1).mean(axis=1).dropna(); CX = pd.concat(cxs, axis=1).mean(axis=1).reindex(S.index).fillna(0)
        corte = max(cortes)
        S1 = S[(S.index < corte) & (S.index >= S.index[0] + pd.Timedelta(days=400))]
        grade.append(dict(fam=fam, p=p, tam=tam, sh1=L.metricas_serie(S1.values, per, CX.loc[S1.index].values)["sharpe"], corte=corte))
        L.conta(len(ativos))
    g = max(grade, key=lambda x: x["sh1"])
    corte = g["corte"]
    linhas = {}
    for modo in ("vendido_fora", "comprado_dentro", "longshort"):
        ss, cxs = {}, {}
        for a in ativos:
            s, cx, _ = serie(a, cls, g["fam"], g["p"], modo, g["tam"]); ss[a] = s; cxs[a] = cx
        S = pd.concat(ss, axis=1).mean(axis=1).dropna(); CX = pd.concat(cxs, axis=1).mean(axis=1).reindex(S.index).fillna(0)
        S2, cx2 = S[S.index >= corte], CX[CX.index >= corte]
        linhas[modo] = dict(**metricas(S2, cx2.values, per), mc_blocos_4s=mc_blocos(S2))
        L.conta(len(ativos))
    caixa = CX[CX.index >= corte]
    linhas["caixa"] = dict(ao_ano=float((1 + caixa).prod() ** (per / len(caixa)) - 1))
    S_v = None
    ss = {a: serie(a, cls, g["fam"], g["p"], "vendido_fora", g["tam"])[0] for a in ativos}
    S_v = pd.concat(ss, axis=1).mean(axis=1).dropna(); S_v = S_v[S_v.index >= corte]
    linhas["t_vendido_fora_vs_caixa"] = L.t_stat(((1 + S_v).groupby(S_v.index.to_period("M")).prod() - (1 + caixa.reindex(S_v.index).fillna(0)).groupby(S_v.index.to_period("M")).prod()).values)
    B[cls] = dict(escolha=f"{g['fam']} {g['p']} · tamanho {g['tam']}", segunda_metade_desde=str(corte.date()), **linhas)
    print(cls, B[cls]["escolha"], {k: (round(v["ao_ano"], 3) if isinstance(v, dict) and "ao_ano" in v else v) for k, v in linhas.items()}, flush=True)
res["parte_B"] = B

# ================================================================ contador e Bonferroni
total_dia = 6623 + L.N_TESTES  # 6.623 simulações até o estudo do mini índice (26/11)
tmin = next(t / 1000 for t in range(1000, 8000) if L.p_bilateral(t / 1000) <= 0.05 / total_dia)
res["testes"] = dict(novos=L.N_TESTES, total_do_dia=total_dia, t_minimo_bonferroni=tmin)
L.salva("k_vendas.json", res)
print(json.dumps(res["testes"]))
