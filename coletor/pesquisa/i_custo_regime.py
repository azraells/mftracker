"""Custo baixo e entrada no regime (26/11/2026). Mesma régua: escolha só em 2022–23, t sem sobreposição agrupado por dia.
(a) só BTC e ETH, custos de ida e volta 0,10/0,15/0,20%, deslizamento novo (0,01% por lado) e antigo (0,1 × (H−L)/C por lado);
    modelo de 4 h (margem escolhida em 2022–23) e "só compras em regime de alta" (margem 0,05, fixada na auditoria).
(b) prazos de 48, 72 e 120 h em cripto e memecoins (modelo retreinado com esse alvo), uma operação por ativo de cada vez,
    sozinho e só comprando em regime de alta; margem escolhida em 2022–23.
(c) regra de regime de cripto (média 50 + saída por tempo de 20 dias + tamanho pela vol): entrar no dia do sinal ou esperar
    até 3 dias pelo primeiro sinal de alta do modelo (4 h ou 24 h); saída igual à da regra.
Usa os g_pred_4h.pkl / g_pred_24h.pkl gerados por `g1_auditoria_rodadas.py base`."""
import math, json
import numpy as np, pandas as pd
import lib as L, ml_core as M, a_tempo as A

CORTE = pd.Timestamp("2024-01-01")
res = {}


def regime_por_hora(Q):
    """1 se o fechamento diário de ontem estava acima da média de 50 dias (sem olhar o dia corrente)."""
    vals = np.full(len(Q), np.nan)
    for a, g in Q.groupby("a"):
        cd = g.c_.groupby(g.index.normalize()).last()
        alta = (cd > cd.rolling(50).mean()).astype(float).shift(1)  # regime de ontem vale hoje
        m = alta.reindex(g.index.normalize()).values
        vals[np.where(Q.a.values == a)[0]] = m
    return vals


def sequencial(R, Hh):
    """uma operação por ativo de cada vez: só aceita um sinal novo depois que o anterior fechou."""
    keep = np.zeros(len(R), bool); livre = {}
    for i, (ts, a) in enumerate(zip(R.index, R.a)):
        if ts >= livre.get(a, pd.Timestamp.min):
            keep[i] = True; livre[a] = ts + pd.Timedelta(hours=Hh)
    return R[keep]


def avalia(R, custos, slip_rt):
    d = dict(ops=len(R))
    if len(R) < 5:
        return d
    t, nd = L.t_por_dia(R.bruto.values, R.index)
    d.update(dias=nd, acerto=float((R.bruto > 0).mean()), bruto=float(R.bruto.mean()), t_bruto=t, deslizamento=float(np.mean(slip_rt)),
             equilibrio=float((R.bruto - slip_rt).mean()))
    for c in custos:
        liq = R.bruto - slip_rt - c
        d[f"liq_{c*100:.2f}%"] = float(liq.mean()); d[f"t_{c*100:.2f}%"] = L.t_por_dia(liq.values, R.index)[0]
    return d


# ---------------------------------------------------------------- (a)
Q4 = pd.read_pickle(L.OUT + "/g_pred_4h.pkl")
Q4["alta"] = regime_por_hora(Q4)
Q4["slip_antigo"] = Q4.groupby("a").apply(lambda g: ((g.h_ - g.l_) / g.c_).median() * 0.1).reindex(Q4.a).values
QB = Q4[Q4.a.isin(["BTC", "ETH"])]
CUSTOS = (0.001, 0.0015, 0.002)
a = {}
grade = []
for mg in np.round(np.arange(0.02, 0.151, 0.01), 2):
    R = M.operacoes(QB, mg, nao_sobrepor=4)
    R1 = R[R.index < CORTE]
    grade.append((mg, len(R1), float((R1.bruto - 0.0002 - 0.0015).mean()) if len(R1) else -9))
    L.conta()
ok = [g for g in grade if g[1] >= 100]
mg_esc = max(ok, key=lambda g: g[2])[0]
a["margem escolhida em 2022–23 (BTC e ETH, 4 h)"] = float(mg_esc)
for nome, filtro in ((f"modelo 4 h, margem {mg_esc}", None), ("só compras em regime de alta, margem 0,05", "compra_alta")):
    mg = mg_esc if filtro is None else 0.05
    R = M.operacoes(QB, mg, nao_sobrepor=4)
    if filtro == "compra_alta":
        R = R[(R.lado > 0) & (R.alta == 1)]
    for parte, S in (("2022–23", R[R.index < CORTE]), ("2024–26", R[R.index >= CORTE])):
        a[f"{nome} · {parte} · deslizamento novo (0,01% por lado)"] = avalia(S, CUSTOS, np.full(len(S), 0.0002))
        a[f"{nome} · {parte} · deslizamento antigo"] = avalia(S, CUSTOS, 2 * S.slip_antigo.values)
        L.conta(2 * len(CUSTOS))
res["a"] = a
print(json.dumps(a, indent=1, default=float, ensure_ascii=False)[:6000])

# ---------------------------------------------------------------- (b)
b = {}
for Hh in (48, 72, 120):
    P = M.painel(Hh)
    Q = M.walk_forward(P, Hh)
    Q["alta"] = regime_por_hora(Q)
    b[f"{Hh} h AUC"] = M.auc(Q.p, (Q.y > 0).astype(int))
    b[f"{Hh} h AUC por ano"] = {str(k): round(M.auc(g.p, (g.y > 0).astype(int)), 3) for k, g in Q.groupby(Q.index.year)}
    Q = Q[Q.index.hour == 0]  # uma decisão por dia, à meia-noite UTC
    for versao in ("modelo sozinho", "só compras em regime de alta"):
        grade = []
        for mg in (0.02, 0.03, 0.05, 0.07, 0.10):
            R = M.operacoes(Q, mg)
            if versao != "modelo sozinho":
                R = R[(R.lado > 0) & (R.alta == 1)]
            R = sequencial(R.sort_index(), Hh)
            R["custo"] = R.cls.map(L.CUSTO) + 0.0002
            R1, R2 = R[R.index < CORTE], R[R.index >= CORTE]
            grade.append(dict(margem=mg, ops1=len(R1), liq1=float((R1.bruto - R1.custo).mean()) if len(R1) else np.nan, R2=R2))
            L.conta()
        ok = [g for g in grade if g["ops1"] >= 30]
        if not ok:
            b[f"{Hh} h · {versao}"] = "menos de 30 operações em 2022–23 em todas as margens"
            continue
        g = max(ok, key=lambda r: r["liq1"])
        R2 = g["R2"]
        d = dict(margem=g["margem"], liq_2022_23=g["liq1"], ops_2024_26=len(R2))
        if len(R2) >= 5:
            for nome, S in (("cripto+memes", R2), ("BTC e ETH", R2[R2.a.isin(["BTC", "ETH"])])):
                if len(S) < 5:
                    continue
                t, nd = L.t_por_dia(S.bruto.values, S.index)
                d[nome] = dict(ops=len(S), acerto=float((S.bruto > 0).mean()), bruto=float(S.bruto.mean()), t_bruto=t,
                               liq_custo_programa=float((S.bruto - S.custo).mean()), t_liq=L.t_por_dia((S.bruto - S.custo).values, S.index)[0],
                               liq_015=float((S.bruto - 0.0017).mean()))
        b[f"{Hh} h · {versao}"] = d
    print(Hh, json.dumps({k: v for k, v in b.items() if k.startswith(str(Hh))}, default=float, ensure_ascii=False)[:2500], flush=True)
res["b"] = b

# ---------------------------------------------------------------- (c)
c = {}
Q24 = pd.read_pickle(L.OUT + "/g_pred_24h.pkl")
sinais_dia = {}
for nomeQ, QQ in (("4 h", Q4), ("24 h", Q24)):
    for mg in (0.02, 0.05):
        s = QQ[QQ.p > QQ.base + mg]
        sinais_dia[(nomeQ, mg)] = set(zip(s.a, s.index.normalize()))


def pesos_cripto(d):
    """média 50 + saída por tempo de 20 dias (espera 5 dias) + tamanho pela vol — igual ao f_saidas_mc.py."""
    cc = d["c"]; r = cc.pct_change().fillna(0).values
    rv = pd.Series(r).rolling(20).std().values * math.sqrt(365)
    meio = len(cc) // 2; tv = float(np.nanmedian(rv[:meio])); rvn = np.nan_to_num(rv, nan=tv)
    w = A.sinais(cc, "media", 50, rvn, tv)
    out = np.zeros_like(w); dentro = 0; bloq = 0
    for i in range(len(w)):
        if w[i] <= 0:
            dentro, bloq = 0, 0; continue
        if bloq > 0:
            bloq -= 1
            if bloq == 0: dentro = 0
            continue
        dentro += 1
        if dentro > 20:
            bloq = 5; continue
        out[i] = w[i]
    return out * np.minimum(1.0, tv / rvn), r


def atrasa(w, datas, ativo, sinais, maxd=3):
    """em cada entrada (0 -> >0), espera o primeiro dia com sinal de alta do modelo, até maxd dias; saída igual."""
    w2 = w.copy(); dif_precos = []
    i = 1
    while i < len(w):
        if w[i] > 0 and w[i - 1] <= 0:
            fim = i
            while fim < len(w) and w[fim] > 0:
                fim += 1
            k = 0
            while k < maxd and (ativo, datas[i + k]) not in sinais and i + k + 1 < fim:
                k += 1
            w2[i:i + k] = 0.0
            dif_precos.append((i, i + k))
            i = fim
        else:
            i += 1
    return w2, dif_precos


for chave in [("4 h", 0.02), ("4 h", 0.05), ("24 h", 0.02), ("24 h", 0.05)]:
    series = {"no dia do sinal": {}, "esperando o modelo": {}}
    difs = []
    for a in L.CLASSES["crypto"]:
        d = L.diario(a); d = d[d.index >= pd.Timestamp("2021-01-01")]
        w, r = pesos_cripto(d)
        datas = list(d.index)
        w2, pares = atrasa(w, datas, a, sinais_dia[chave])
        cx = L.caixa_para(d.index, "crypto")
        for nome, ww in (("no dia do sinal", w), ("esperando o modelo", w2)):
            ww = np.r_[0.0, ww[:-1]]
            giro = np.abs(np.diff(np.r_[0.0, ww]))
            series[nome][a] = pd.Series(ww * r + (1 - ww) * cx - giro * L.CUSTO["crypto"] / 2, index=d.index)
        for i0, i1 in pares:
            if d.index[i0] >= pd.Timestamp("2022-02-01"):
                difs.append(dict(data=d.index[i0], ativo=a, dias=i1 - i0, preco=math.log(d.c.iloc[i1] / d.c.iloc[i0])))
    D = pd.DataFrame(difs)
    linha = dict(entradas=len(D), dias_medios=float(D.dias.mean()), entrou_no_mesmo_dia=float((D.dias == 0).mean()),
                 diferenca_preco_media=float(D.preco.mean()), t_preco=L.t_stat(D.preco.values))
    for parte, ini, fimp in (("2022–23", "2022-02-01", "2023-12-31"), ("2024–26", "2024-01-01", "2026-12-31")):
        for nome in series:
            S = pd.concat(series[nome], axis=1).mean(axis=1)
            S = S[(S.index >= ini) & (S.index <= fimp)]
            m = L.metricas_serie(S.values, 365, L.caixa_para(S.index, "crypto"))
            linha[f"{parte} · {nome}"] = dict(ao_ano=m["cagr"], sharpe=m["sharpe"], maior_queda=m["mdd"])
        Sa = pd.concat(series["no dia do sinal"], axis=1).mean(axis=1); Sb = pd.concat(series["esperando o modelo"], axis=1).mean(axis=1)
        Sa, Sb = Sa[(Sa.index >= ini) & (Sa.index <= fimp)], Sb[(Sb.index >= ini) & (Sb.index <= fimp)]
        linha[f"{parte} · t (esperar − mesmo dia)"] = L.t_dif_sharpe_mensal(Sb.values, Sa.values, Sa.index, L.caixa_para(Sa.index, "crypto"))
    c[f"modelo {chave[0]}, margem {chave[1]}"] = linha
    L.conta(4)
res["c"] = c
print(json.dumps(c, indent=1, default=float, ensure_ascii=False))

L.salva("i_custo_regime.json", res)
L.salva("i_n.json", {"n": L.N_TESTES})
print("simulações:", L.N_TESTES)
