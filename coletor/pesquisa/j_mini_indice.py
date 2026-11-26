"""Mini índice, aproximado pelo Ibovespa (^BVSP) e pelo BOVA11 em candles de 1 h do Yahoo (dez/2023 a nov/2026).
Custo do WIN convertido em % do valor do contrato (1 ponto = R$ 0,20; índice ~165 mil pontos → contrato ~R$ 33 mil):
  emolumentos + registro ~R$ 0,30 por contrato por lado ≈ 0,0009%; deslizamento de 1 ponto por lado ≈ 0,0006%,
  1 tick (5 pontos) ≈ 0,0030%, 2 ticks (10 pontos) ≈ 0,0061%.
  Ida e volta: 1 ponto ≈ 0,003%; 1 tick ≈ 0,008%; 2 ticks ≈ 0,014%; e um cenário de estresse de 0,04%.
Testes (uma operação por dia no máximo; tudo fecha no último candle do pregão; escolha só na 1ª metade dos dias):
  T1 rompimento da faixa da 1ª (ou das 2 primeiras) hora(s), seguindo ou contrariando, com ou sem stop;
  T2 reversão × continuação: depois de k horas, se o movimento desde o fechamento anterior passou de x desvios;
  T3 só a última hora e só a primeira hora (comprado);
  T4 modelo de aprendizado (logística) decidindo às 11 h e às 13 h, alvo = retorno até o fechamento, treino só no passado (mensal);
  T5 filtro de regime (fechamento diário acima da média de 50 dias): compra só em alta, venda só em baixa."""
import itertools, math, json
import numpy as np, pandas as pd
import os
import lib as L, ml_core as M

# baixa o Ibovespa do Yahoo na primeira vez (precisa de internet e do pacote yfinance)
os.makedirs(L.OUT, exist_ok=True)
for _arq, _kw in (("bvsp_1h.pkl", dict(period="730d", interval="1h")), ("bvsp_1d.pkl", dict(period="max", interval="1d"))):
    if not os.path.exists(os.path.join(L.OUT, _arq)):
        import yfinance as yf
        yf.download("^BVSP", auto_adjust=False, progress=False, **_kw).to_pickle(os.path.join(L.OUT, _arq))

CUSTOS = {"1 ponto": 0.00003, "1 tick (5 pontos)": 0.00008, "2 ticks (10 pontos)": 0.00014, "estresse 0,04%": 0.0004}


def carrega_bvsp():
    d = pd.read_pickle(L.OUT + "/bvsp_1h.pkl")
    d.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in d.columns]
    d.index = d.index.tz_convert("America/Sao_Paulo").tz_localize(None)
    return d.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c"})[["o", "h", "l", "c"]]


def carrega_bova():
    h = L.horario("BOVA11")
    h.index = h.index - pd.Timedelta(hours=3)  # UTC -> Brasília (aproximado; B3 sem horário de verão desde 2019)
    return h[["o", "h", "l", "c"]]


def regime_diario():
    d = pd.read_pickle(L.OUT + "/bvsp_1d.pkl")
    c = d["Close"].iloc[:, 0] if isinstance(d["Close"], pd.DataFrame) else d["Close"]
    c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
    alta = (c > c.rolling(50).mean()).astype(float).shift(1)  # regime de ontem vale hoje
    sd = c.pct_change().rolling(20).std().shift(1)
    return alta, sd, c


ALTA, SD, CD = regime_diario()


def sessoes(h):
    """lista de (data, DataFrame do pregão, fechamento anterior) com ao menos 5 candles."""
    out = []
    ant = None
    for dt, g in h.groupby(h.index.normalize()):
        g = g.sort_index()
        if len(g) >= 5 and ant is not None:
            out.append((dt, g, ant))
        ant = g.c.iloc[-1]
    return out


def resultado(lst, dias_corte, nome, regime=False):
    """lst: lista de (data, lado, retorno log do lado comprado)."""
    if not lst:
        return None
    D = pd.DataFrame(lst, columns=["data", "lado", "r"]).set_index("data")
    if regime:
        al = ALTA.reindex(D.index).values
        D = D[((D.lado > 0) & (al == 1)) | ((D.lado < 0) & (al == 0))]
    D["bruto"] = D.lado * D.r
    return D


def resume(D, corte):
    out = {}
    for parte, S in (("1ª metade", D[D.index < corte]), ("2ª metade", D[D.index >= corte])):
        if len(S) < 10:
            out[parte] = dict(ops=len(S)); continue
        d = dict(ops=len(S), acerto=float((S.bruto > 0).mean()), bruto=float(S.bruto.mean()), t_bruto=L.t_stat(S.bruto.values))
        for k, c in CUSTOS.items():
            d[f"líquido {k}"] = float((S.bruto - c).mean()); d[f"t {k}"] = L.t_stat((S.bruto - c).values)
        pts = (S.bruto * CD.reindex(S.index).ffill().values)
        d["pontos por operação"] = float(np.nanmean(pts))
        d["R$ por contrato por operação (bruto)"] = float(np.nanmean(pts) * 0.20)
        d["dias"] = len(S)
        out[parte] = d
    return out


res = {}
for fonte, H in (("Ibovespa", carrega_bvsp()), ("BOVA11", carrega_bova())):
    S = sessoes(H)
    datas = [s[0] for s in S]
    corte = datas[len(datas) // 2]
    R = {}
    # T1 rompimento da faixa inicial
    grade = []
    for k, lado, stop in itertools.product((1, 2), ("segue", "contraria"), (False, True)):
        lst = []
        for dt, g, ant in S:
            hi, lo = g.h.iloc[:k].max(), g.l.iloc[:k].min()
            fim = g.c.iloc[-1]
            for i in range(k, len(g) - 1):
                c = g.c.iloc[i]
                if c > hi or c < lo:
                    d = 1 if c > hi else -1
                    if lado == "contraria": d = -d
                    saida = fim
                    if stop:
                        # stop sempre do lado contrário à posição, a uma distância igual à faixa (seguindo: o outro lado da faixa)
                        if lado == "segue":
                            nivel = lo if d > 0 else hi
                        else:
                            nivel = c - (hi - lo) if d > 0 else c + (hi - lo)
                        for j in range(i + 1, len(g)):
                            if (d > 0 and g.l.iloc[j] <= nivel) or (d < 0 and g.h.iloc[j] >= nivel):
                                saida = nivel; break
                    lst.append((dt, d, math.log(saida / c)))
                    break
        for reg in (False, True):
            DD = resultado(lst, corte, "T1", regime=reg)
            if DD is None or len(DD) < 20:
                continue
            D1 = DD[DD.index < corte]
            grade.append(dict(k=k, lado=lado, stop=stop, regime=reg, liq1=float((D1.bruto - CUSTOS["1 tick (5 pontos)"]).mean()), D=DD))
            L.conta()
    for reg in (False, True):
        g = max([x for x in grade if x["regime"] == reg], key=lambda x: x["liq1"])
        R[f"T1 rompimento · {'com regime' if reg else 'sem regime'}"] = dict(escolha=f"faixa de {g['k']} h, {g['lado']}, stop {'sim' if g['stop'] else 'não'}", **resume(g["D"], corte))
    # T2 reversão × continuação
    grade = []
    for k, x, lado in itertools.product((1, 2, 3), (0.5, 1.0, 1.5), ("segue", "contraria")):
        lst = []
        for dt, g, ant in S:
            sd = SD.get(dt, np.nan)
            if not np.isfinite(sd) or len(g) <= k + 1:
                continue
            mov = math.log(g.c.iloc[k - 1] / ant)
            z = mov / (sd * math.sqrt(min(k, 7) / 7 + 0.5))  # inclui a noite (o 1º candle do índice começa no fechamento anterior)
            if abs(z) >= x:
                d = int(np.sign(mov)) * (1 if lado == "segue" else -1)
                lst.append((dt, d, math.log(g.c.iloc[-1] / g.c.iloc[k - 1])))
        for reg in (False, True):
            DD = resultado(lst, corte, "T2", regime=reg)
            if DD is None or len(DD) < 20:
                continue
            D1 = DD[DD.index < corte]
            grade.append(dict(k=k, x=x, lado=lado, regime=reg, liq1=float((D1.bruto - CUSTOS["1 tick (5 pontos)"]).mean()) if len(D1) >= 10 else -9, D=DD))
            L.conta()
    for reg in (False, True):
        g = max([x for x in grade if x["regime"] == reg], key=lambda x: x["liq1"])
        R[f"T2 reversão/continuação · {'com regime' if reg else 'sem regime'}"] = dict(escolha=f"depois de {g['k']} h, movimento ≥ {g['x']} desvios, {g['lado']}", **resume(g["D"], corte))
    # T3 primeira e última hora (comprado), regras fixas
    for nome, fn in (("T3 só a última hora, comprado", lambda g, ant: math.log(g.c.iloc[-1] / g.c.iloc[-2])),
                     ("T3 só a primeira hora, comprado (inclui a noite)", lambda g, ant: math.log(g.c.iloc[0] / ant)),
                     ("T3 pregão sem a 1ª hora, comprado", lambda g, ant: math.log(g.c.iloc[-1] / g.c.iloc[0]))):
        lst = [(dt, 1, fn(g, ant)) for dt, g, ant in S]
        R[nome] = resume(resultado(lst, corte, nome), corte)
        L.conta()
    # T4 modelo de aprendizado
    linhas = []
    for dt, g, ant in S:
        sd = SD.get(dt, np.nan); al = ALTA.get(dt, np.nan)
        cd = CD[CD.index < dt]
        if len(cd) < 60 or not np.isfinite(sd):
            continue
        for k in (1, 3):  # decide no fim da 1ª hora (≈11 h) e da 3ª (≈13 h)
            if len(g) <= k + 1:
                continue
            c = g.c.iloc[:k]
            linhas.append(dict(data=dt, k=k, mov=math.log(c.iloc[-1] / ant) / sd, ult=math.log(g.c.iloc[k - 1] / g.o.iloc[k - 1]) / sd,
                               faixa=(g.h.iloc[:k].max() - g.l.iloc[:k].min()) / c.iloc[-1] / sd,
                               r1d=math.log(cd.iloc[-1] / cd.iloc[-2]) / sd, r5d=math.log(cd.iloc[-1] / cd.iloc[-6]) / sd,
                               r20d=math.log(cd.iloc[-1] / cd.iloc[-21]) / sd, d50=math.log(cd.iloc[-1] / cd.iloc[-50:].mean()) / sd,
                               vrat=math.log(sd / cd.pct_change().iloc[-60:].std()), dow=dt.dayofweek, alta=al,
                               y=math.log(g.c.iloc[-1] / g.c.iloc[k - 1])))
    P = pd.DataFrame(linhas).set_index("data")
    P["ds"] = np.sin(2 * np.pi * P.dow / 5); P["dc"] = np.cos(2 * np.pi * P.dow / 5)
    cols = ["mov", "ult", "faixa", "r1d", "r5d", "r20d", "d50", "vrat", "ds", "dc"]
    meses = P.index.to_period("M"); ums = sorted(meses.unique())
    pred = []
    for i, m in enumerate(ums):
        tr = P[meses < m]
        if len(tr) < 250:
            continue
        te = P[meses == m]
        mu, sdv = tr[cols].mean(), tr[cols].std().replace(0, 1)
        X = np.c_[np.ones(len(tr)), ((tr[cols] - mu) / sdv).clip(-5, 5).values]
        yt = (tr.y > 0).astype(float).values
        w = M.logit_fit(X, yt, lam=len(yt) * 0.05)
        Xs = np.c_[np.ones(len(te)), ((te[cols] - mu) / sdv).clip(-5, 5).values]
        pred.append(te.assign(p=1 / (1 + np.exp(-Xs @ w)), base=yt.mean()))
        L.conta()
    PP = pd.concat(pred)
    R["T4 modelo: AUC fora da amostra"] = dict(auc=M.auc(PP.p, (PP.y > 0).astype(int)), previsoes=len(PP), de=str(PP.index[0].date()))
    corte4 = PP.index[len(PP) // 2]
    grade = []
    for k, mg, reg in itertools.product((1, 3), (0.02, 0.05, 0.08), (False, True)):
        Q = PP[PP.k == k]
        lado = np.where(Q.p > Q.base + mg, 1, np.where(Q.p < Q.base - mg, -1, 0))
        D = pd.DataFrame({"lado": lado, "r": Q.y.values, "alta": Q.alta.values}, index=Q.index)[lado != 0]
        if reg:
            D = D[((D.lado > 0) & (D.alta == 1)) | ((D.lado < 0) & (D.alta == 0))]
        D["bruto"] = D.lado * D.r
        D1 = D[D.index < corte4]
        if len(D1) >= 15:
            grade.append(dict(k=k, mg=mg, regime=reg, liq1=float((D1.bruto - CUSTOS["1 tick (5 pontos)"]).mean()), D=D))
        L.conta()
    for reg in (False, True):
        cand = [x for x in grade if x["regime"] == reg]
        if cand:
            g = max(cand, key=lambda x: x["liq1"])
            R[f"T4 modelo · {'com regime' if reg else 'sem regime'}"] = dict(escolha=f"decide às {10 + g['k']} h, margem {g['mg']}", **resume(g["D"], corte4))
    res[fonte] = R
    print(fonte, json.dumps(R, default=float, ensure_ascii=False)[:5000], flush=True)

L.salva("j_mini_indice.json", res)
L.salva("j_n.json", {"n": L.N_TESTES})
print("simulações:", L.N_TESTES)
