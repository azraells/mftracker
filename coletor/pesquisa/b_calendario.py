"""Estudo B: efeitos de calendário. Cada efeito é medido na 1ª metade e conferido na 2ª.
B1 overnight × pregão (ações, diário com abertura e fechamento)
B2 dia da semana (todos, diário)
B3 hora do dia (cripto e memecoins, 1 h): escolhe as 6 melhores horas na 1ª metade
B4 virada do mês (ações): último dia útil + 3 primeiros, regra fixada antes (literatura)
B5 fim de semana em cripto"""
import math
import numpy as np, pandas as pd
import lib as L

linhas = []


def registra(estudo, cls, desc, n1, m1, t1, n2, m2, t2, custo_por_dia=None, extra=None):
    d = dict(estudo=estudo, cls=cls, desc=desc, n1=n1, m1=m1, t1=t1, n2=n2, m2=m2, t2=t2, custo_dia=custo_por_dia)
    if extra: d.update(extra)
    linhas.append(d)


def metade(s):
    m = len(s) // 2
    return s.iloc[:m], s.iloc[m:]


for cls in ("b3", "us"):
    on_all, in_all = [], []
    for a in L.CLASSES[cls]:
        d = L.diario(a)
        d = d[(d["o"] > 0) & (d["c"] > 0)]
        on = (d["o"] / d["c"].shift(1) - 1).dropna()
        intra = (d["c"] / d["o"] - 1).loc[on.index]
        on_all.append(on.rename(a)); in_all.append(intra.rename(a))
        L.conta(2)
    ON = pd.concat(on_all, axis=1).mean(axis=1).dropna(); IN = pd.concat(in_all, axis=1).mean(axis=1).dropna()
    for nome, s in (("só a noite (compra no fechamento, vende na abertura)", ON), ("só o pregão (compra na abertura, vende no fechamento)", IN)):
        a1, a2 = metade(s)
        registra("B1", cls, nome, len(a1), a1.mean(), L.t_stat(a1), len(a2), a2.mean(), L.t_stat(a2), custo_por_dia=L.CUSTO[cls],
                 extra=dict(ano2=(1 + a2).prod() ** (252 / len(a2)) - 1, ano2_liq=(1 + a2 - L.CUSTO[cls]).prod() ** (252 / len(a2)) - 1,
                            ano2_liq_barato=(1 + a2 - L.CUSTO_BARATO[cls]).prod() ** (252 / len(a2)) - 1))

for cls, ativos in L.CLASSES.items():
    rs = []
    for a in ativos:
        d = L.diario(a); rs.append(d["c"].pct_change().rename(a)); L.conta()
    R = pd.concat(rs, axis=1).mean(axis=1).dropna()
    r1, r2 = metade(R)
    g1 = r1.groupby(r1.index.dayofweek).mean()
    melhor = int(g1.idxmax()); pior = int(g1.idxmin())
    for dia, rot in ((melhor, "melhor dia da 1ª metade"), (pior, "pior dia da 1ª metade")):
        s1 = r1[r1.index.dayofweek == dia]; s2 = r2[r2.index.dayofweek == dia]
        o2 = r2[r2.index.dayofweek != dia]
        nomes = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
        registra("B2", cls, f"{rot}: {nomes[dia]}", len(s1), s1.mean(), L.t_stat(s1), len(s2), s2.mean() - o2.mean(),
                 (s2.mean() - o2.mean()) / math.sqrt(s2.var() / len(s2) + o2.var() / len(o2)))

for cls in ("crypto", "meme"):
    rs = []
    for a in L.CLASSES[cls]:
        h = L.horario(a)
        rs.append(np.log(h["c"]).diff().rename(a)); L.conta(24)
    R = pd.concat(rs, axis=1).mean(axis=1).dropna()
    r1, r2 = metade(R)
    g1 = r1.groupby(r1.index.hour).mean()
    top = sorted(g1.sort_values(ascending=False).index[:6].tolist())
    bot = sorted(g1.sort_values().index[:6].tolist())
    for horas, rot in ((top, "6 melhores horas (UTC) da 1ª metade"), (bot, "6 piores horas (UTC) da 1ª metade")):
        def diario_sub(r):
            m = r[r.index.hour.isin(horas)]
            return m.groupby(m.index.normalize()).sum()
        s1, s2 = diario_sub(r1), diario_sub(r2)
        resto2 = r2[~r2.index.hour.isin(horas)]; resto2 = resto2.groupby(resto2.index.normalize()).sum() * (6 / 18)
        dif = (s2 - resto2.reindex(s2.index).fillna(0))
        blocos = sum(1 for i, x in enumerate(horas) if (x - 1) % 24 not in horas)
        registra("B3", cls, f"{rot}: {horas}", len(s1), s1.mean(), L.t_stat(s1), len(s2), dif.mean(), L.t_stat(dif),
                 custo_por_dia=blocos * L.CUSTO[cls], extra=dict(bruto2_dia=s2.mean(), blocos=blocos))

for cls in ("b3", "us"):
    rs = []
    for a in L.CLASSES[cls]:
        d = L.diario(a); rs.append(d["c"].pct_change().rename(a)); L.conta()
    R = pd.concat(rs, axis=1).mean(axis=1).dropna()
    per = R.index.to_period("M")
    pos = R.groupby(per).cumcount(); tam = R.groupby(per).transform("size")
    janela = (pos >= tam - 1) | (pos <= 2)
    r1, r2 = metade(R); j1, j2 = metade(janela)
    a1, b1 = r1[j1.values], r1[~j1.values]; a2, b2 = r2[j2.values], r2[~j2.values]
    t1 = (a1.mean() - b1.mean()) / math.sqrt(a1.var() / len(a1) + b1.var() / len(b1))
    t2 = (a2.mean() - b2.mean()) / math.sqrt(a2.var() / len(a2) + b2.var() / len(b2))
    registra("B4", cls, "virada do mês (último dia útil + 3 primeiros) contra o resto", len(a1), a1.mean() - b1.mean(), t1, len(a2), a2.mean() - b2.mean(), t2,
             extra=dict(dentro2=a2.mean(), fora2=b2.mean()))

for cls in ("crypto", "meme"):
    rs = []
    for a in L.CLASSES[cls]:
        d = L.diario(a); rs.append(d["c"].pct_change().rename(a)); L.conta()
    R = pd.concat(rs, axis=1).mean(axis=1).dropna()
    r1, r2 = metade(R)
    def dif(r):
        w = r[r.index.dayofweek >= 5]; u = r[r.index.dayofweek < 5]
        return w.mean() - u.mean(), (w.mean() - u.mean()) / math.sqrt(w.var() / len(w) + u.var() / len(u)), len(w)
    m1, t1, n1 = dif(r1); m2, t2, n2 = dif(r2)
    registra("B5", cls, "sábado e domingo contra dias úteis", n1, m1, t1, n2, m2, t2)

df = pd.DataFrame(linhas)
df.to_csv(L.OUT + "/b_calendario.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(df.to_string())
print("simulações:", L.N_TESTES)
L.salva("b_n.json", {"n": L.N_TESTES})
