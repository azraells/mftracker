"""Confere o efeito overnight com os candles de 1 h: noite = abertura do 1º candle do dia / fechamento do último
candle do dia anterior; pregão = fechamento do último / abertura do 1º. Também mede cada hora do pregão."""
import numpy as np, pandas as pd
import lib as L

for cls in ("b3", "us"):
    on, intra, horas = [], [], []
    for a in L.CLASSES[cls]:
        h = L.horario(a)
        if h is None:
            continue
        loc = h.index - pd.Timedelta(hours=3 if cls == "b3" else 5)
        dia = loc.normalize()
        g = h.groupby(dia)
        o = g["o"].first(); c = g["c"].last()
        on.append((o / c.shift(1) - 1).rename(a)); intra.append((c / o - 1).rename(a))
        r = h["c"] / h["o"] - 1
        horas.append(pd.Series(r.values, index=loc.hour).groupby(level=0).mean().rename(a))
        L.conta(2)
    ON = pd.concat(on, axis=1).mean(axis=1).dropna(); IN = pd.concat(intra, axis=1).mean(axis=1).dropna()
    print(cls, "dias", len(ON), "| noite média %.4f%% t %.2f | pregão média %.4f%% t %.2f" % (ON.mean() * 100, L.t_stat(ON), IN.mean() * 100, L.t_stat(IN)))
    print("  por hora local (média do retorno do candle, %):", (pd.concat(horas, axis=1).mean(axis=1) * 100).round(4).to_dict())
L.salva("b1_n.json", {"n": L.N_TESTES})
