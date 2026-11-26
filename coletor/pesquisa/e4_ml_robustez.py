"""E4: robustez do sinal de alta confiança (4 h e 24 h, cripto e memecoins): por ano, mercado, lado e ativo,
e banca operando só margens 0,08/0,10 com custo do programa, 10% por operação (24 h: um sinal por ativo por vez).
ATENÇÃO: superado pela auditoria (g2) — margem escolhida olhando o resultado e sem deslizamento."""
import numpy as np, pandas as pd
import lib as L

res = {}
for arq, H, horas in (("e2_pred_cripto_4h.csv", "4 h", 4), ("e2_pred_cripto_24h.csv", "24 h", 24)):
    P = pd.read_csv(L.OUT + "/" + arq, index_col=0, parse_dates=True)
    for mg in (0.08, 0.10):
        lado = np.where(P.p > P.base + mg, 1, np.where(P.p < P.base - mg, -1, 0))
        Q = P[lado != 0].copy(); Q["lado"] = lado[lado != 0]
        Q["bruto"] = Q.lado * Q.y; Q["custo"] = Q.cls.map(L.CUSTO); Q["liq"] = Q.bruto - Q.custo
        chave = f"{H} margem {mg}"
        tab = {}
        for nome, g in (("ano", Q.index.year), ("mercado", Q.cls), ("lado", np.where(Q.lado > 0, "compra", "venda")), ("ativo", Q.a)):
            tt = Q.groupby(g).agg(ops=("liq", "size"), acerto=("bruto", lambda x: (x > 0).mean()), bruto=("bruto", "mean"), liq=("liq", "mean"))
            tab[nome] = tt.round(4).reset_index().to_dict("records")
        Q = Q.sort_index()
        if horas == 24:
            keep, livre = [], {}
            for ts, r in Q.iterrows():
                if ts >= livre.get(r.a, pd.Timestamp.min):
                    keep.append(True); livre[r.a] = ts + pd.Timedelta(hours=24)
                else:
                    keep.append(False)
            Q = Q[keep]
        por_hora = Q.groupby(Q.index).apply(lambda g: (g.liq * (0.10 / max(len(g), 1)) * min(len(g), 3)).sum())
        eq = (1 + por_hora).cumprod() * 1000
        anos = (por_hora.index[-1] - por_hora.index[0]).days / 365
        tab["banca"] = dict(inicio=str(por_hora.index[0].date()), fim=str(por_hora.index[-1].date()), final=float(eq.iloc[-1]),
                            ao_ano=float((eq.iloc[-1] / 1000) ** (1 / anos) - 1), maior_queda=float((eq / eq.cummax() - 1).min()), ops=int(len(Q)))
        tab["banca_por_ano"] = {str(k): float((1 + v).prod() - 1) for k, v in por_hora.groupby(por_hora.index.year)}
        res[chave] = tab
        L.conta(len(tab))
        print("=====", chave)
        for k in ("ano", "mercado", "lado"):
            print(pd.DataFrame(tab[k]).to_string())
        print("banca:", tab["banca"], tab["banca_por_ano"])
L.salva("e4_ml_robustez.json", res)
L.salva("e4_n.json", {"n": L.N_TESTES})
