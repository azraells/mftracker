"""Estudo D1: carry de financiamento neutro ao mercado.
Compra o spot e vende o perpétuo no mesmo valor: o preço se cancela e sobra a taxa de financiamento
(recebe quando é positiva, paga quando é negativa). Custos: montar e desmontar as duas pernas = 0,30%
(spot 0,10% + perpétuo 0,05%, ida e volta) e rebalanceamento de 0,02% por semana.
Variante com filtro: só fica montado quando a média dos últimos 3 dias está acima de um limiar
escolhido na 1ª metade. Não inclui a variação da base (diferença spot × perpétuo), que não temos."""
import math
import numpy as np, pandas as pd
import lib as L

MONTA = 0.003
REB_SEMANA = 0.0002
linhas, series = [], {}
for a in L.CLASSES["crypto"] + L.CLASSES["meme"]:
    f = L.funding(a)
    if f is None or len(f) < 300:
        continue
    f = f[~f.index.duplicated()]
    por_dia = f.groupby(f.index.normalize()).sum()
    por_dia = por_dia.asfreq("D").fillna(0.0)
    m = len(por_dia) // 2
    media3 = por_dia.rolling(3).mean().shift(1)
    melhor, best_sh = None, -9
    for th in (None, 0.0, 0.00015, 0.0003, 0.0006):   # por dia (≈ 0, 0,005%, 0,01%, 0,02% por 8 h)
        on = pd.Series(1.0, index=por_dia.index) if th is None else (media3 > th).astype(float)
        giro = on.diff().abs().fillna(on.iloc[0])
        liq = on * por_dia - giro * MONTA / 2 - on * REB_SEMANA / 7
        sh = liq.iloc[:m].mean() / (liq.iloc[:m].std() + 1e-12)
        L.conta()
        if sh > best_sh:
            best_sh, melhor, serie = sh, th, liq
    s2 = serie.iloc[m:]
    sempre = (por_dia - REB_SEMANA / 7).iloc[m:]
    sempre.iloc[0] -= MONTA
    for nome, s in (("sempre montado", sempre), (f"filtro escolhido ({'sempre' if melhor is None else f'média 3 d > {melhor*100/3:.4f}% por 8 h'})", s2)):
        eq = (1 + s).cumprod()
        mes = s.groupby(s.index.to_period("M")).sum()
        linhas.append(dict(ativo=a, cls=L.CLS_DE[a], regra=nome, de=str(s.index[0].date()), ano=float(s.mean() * 365),
                           mdd=float((eq / eq.cummax() - 1).min()), meses=len(mes), meses_neg=int((mes < 0).sum()),
                           pior_mes=float(mes.min()), t=L.t_stat(s.values)))
    series[a] = s2
df = pd.DataFrame(linhas)
# carteira igual entre ativos (2ª metade de cada um, alinhado por data)
C = pd.concat(series, axis=1).fillna(0.0)
C = C.loc[C.index >= pd.Timestamp("2024-01-01")]
port = C.mean(axis=1)
eq = (1 + port).cumprod(); mes = port.groupby(port.index.to_period("M")).sum()
tot = dict(ativo="carteira igual", cls="—", regra="filtro escolhido, 2024 em diante", de=str(port.index[0].date()), ano=float(port.mean() * 365),
           mdd=float((eq / eq.cummax() - 1).min()), meses=len(mes), meses_neg=int((mes < 0).sum()), pior_mes=float(mes.min()), t=L.t_stat(port.values))
df = pd.concat([df, pd.DataFrame([tot])])
df.to_csv(L.OUT + "/d_funding_carry.csv", index=False)
port.to_csv(L.OUT + "/d_funding_carteira.csv")
pd.set_option("display.width", 250)
print(df.round(4).to_string())
print("simulações:", L.N_TESTES)
L.salva("d_n.json", {"n": L.N_TESTES})
