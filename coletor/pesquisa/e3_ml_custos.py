"""E3: o sinal paga o custo? Usa as previsões fora da amostra salvas (e2_pred_*), varia margem e custo.
A margem é observada em 2022-2023 e conferida em 2024-2026 (diário: antes/depois de 2020)."""
import numpy as np, pandas as pd
import lib as L

linhas = []
for arq, H in (("e2_pred_cripto_4h.csv", "4 h"), ("e2_pred_cripto_24h.csv", "24 h"), ("e2_pred_diario_5d.csv", "5 dias")):
    P = pd.read_csv(L.OUT + "/" + arq, index_col=0, parse_dates=True)
    corte = pd.Timestamp("2024-01-01") if "cripto" in arq else pd.Timestamp("2020-01-01")
    for mg in (0.03, 0.05, 0.08, 0.10, 0.12, 0.15):
        lado = np.where(P.p > P.base + mg, 1, np.where(P.p < P.base - mg, -1, 0))
        sel = lado != 0
        Q = P[sel].copy(); Q["lado"] = lado[sel]; Q["bruto"] = Q.lado * Q.y
        Q["acerto"] = (Q.bruto > 0).astype(float)
        Q["base_dir"] = np.where(Q.lado > 0, Q.base, 1 - Q.base)
        for parte, R in (("treino da margem (até o corte)", Q[Q.index < corte]), ("conferência (depois do corte)", Q[Q.index >= corte])):
            if len(R) < 30:
                continue
            t, nd = L.t_por_dia(R.bruto.values, R.index)
            linhas.append(dict(H=H, margem=mg, parte=parte, ops=len(R), ops_dia=len(R) / max(nd, 1), acerto=R.acerto.mean(), base_dir=R.base_dir.mean(),
                               bruto=R.bruto.mean(), t_bruto=t, **{f"liq_{int(c*10000)}bp": R.bruto.mean() - c for c in (0.0004, 0.001, 0.002, 0.003)},
                               so_compra_bruto=R[R.lado > 0].bruto.mean(), so_venda_bruto=R[R.lado < 0].bruto.mean()))
        L.conta()
df = pd.DataFrame(linhas)
df.to_csv(L.OUT + "/e3_ml_custos.csv", index=False)
pd.set_option("display.width", 300)
print(df.round(4).to_string())
L.salva("e3_n.json", {"n": L.N_TESTES})
