"""H: soma todas as simulações e aplica Bonferroni (no total de simulações e no número de hipóteses
conferidas fora da amostra). Separa "melhora o risco" de "prevê direção"."""
import glob, json, math, os
import numpy as np, pandas as pd
import lib as L

total = sum(json.load(open(f))["n"] for f in glob.glob(L.OUT + "/*_n.json"))
H = []  # (categoria, hipótese, t fora da amostra)
a = pd.read_csv(L.OUT + "/f_final.csv")
for _, r in a.iterrows():
    if r.estrategia != "comprar e segurar" and np.isfinite(r.t_vs_segurar) and "saída base" not in r.estrategia:
        H.append(("melhora o risco", f"{L.NOME[r.cls]}: {r.estrategia} contra segurar (Sharpe acima do caixa)", r.t_vs_segurar))
b = pd.read_csv(L.OUT + "/b_calendario.csv")
for _, r in b.iterrows():
    H.append(("prevê direção (calendário)", f"{L.NOME[r.cls]}: {r.desc}", r.t2))
c = pd.read_csv(L.OUT + "/c_relacoes.csv")
for _, r in c.iterrows():
    H.append(("prevê direção (relações)", f"{r.desc}", r.t2))
e = pd.read_csv(L.OUT + "/e1_extremos.csv")
for _, r in e.iterrows():
    H.append(("prevê direção (dias extremos)", f"{L.NOME[r.cls]}: {r.desc}", r.t2))
ml = json.load(open(L.OUT + "/e2_ml.json", encoding="utf-8"))
for r in ml:
    for mg in ("m0.0", "m0.02", "m0.05"):
        H.append(("prevê direção (modelo, bruto)", f"{r['nome']} {r['H']} margem {mg[1:]}: retorno bruto", r[mg]["t_bruto"]))
        H.append(("prevê direção (modelo, líquido)", f"{r['nome']} {r['H']} margem {mg[1:]}: retorno líquido", r[mg]["t_liq"]))
df = pd.DataFrame(H, columns=["categoria", "hipotese", "t"])
df["p"] = df.t.apply(L.p_bilateral)
m = len(df)
df["bonf_hipoteses"] = df.p <= 0.05 / m
df["bonf_total"] = df.p <= 0.05 / total
df["positivo"] = df.t > 0
df.to_csv(L.OUT + "/h_bonferroni.csv", index=False)
z = lambda n: next(t for t in np.arange(1, 8, 0.001) if L.p_bilateral(t) <= 0.05 / n)
info = dict(simulacoes=int(total), hipoteses_fora_da_amostra=m, t_minimo_hipoteses=round(z(m), 2), t_minimo_total=round(z(total), 2))
L.salva("h_info.json", info)
print(info)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110)
print(df[df.t.abs() >= 2].sort_values("t", ascending=False).to_string())
