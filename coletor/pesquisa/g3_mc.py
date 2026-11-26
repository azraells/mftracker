"""Auditoria D: Monte Carlo das estratégias de regime com R$ 1.000 por 1 ano, comparando três fontes de semanas:
(i) só a 2ª metade (como antes), (ii) a série inteira, em semanas soltas, (iii) a série inteira, em blocos de
4 semanas seguidas (preserva sequências de alta e de queda). 2.000 caminhos cada."""
import math
import numpy as np, pandas as pd
import lib as L, a_tempo as A

ESCOLHAS = {"crypto": ("media", 50, "tempo", 20, "vol"), "meme": ("donchian", 20, "tempo", 60, "vol"),
            "b3": ("momento", 21, "tempo", 60, "vol"), "us": ("conjunto", None, "base", None, "fixo"), "fx": ("donchian", 55, "vol_pct", 0.8, "fixo")}


def pesos(d, cls, fam, p, saida, ps, tam):  # mesma lógica do f_saidas_mc.py
    c = d["c"]; r = c.pct_change().fillna(0).values
    per = 365 if cls in ("crypto", "meme") else 252
    rv = pd.Series(r).rolling(20).std().values * math.sqrt(per)
    meio = len(c) // 2
    tv = float(np.nanmedian(rv[:meio])); rvn = np.nan_to_num(rv, nan=tv)
    w = np.nan_to_num(A.sinais(c, fam, p, rvn, tv), nan=0.0)
    if saida != "base":
        atr = (d["h"] - d["l"]).rolling(14).mean().values
        lim = np.nanquantile(rv[:meio], ps) if saida == "vol_pct" else None
        out = np.zeros_like(w); dentro = 0; pico = 0.0; bloq = 0
        for i in range(len(w)):
            if w[i] <= 0:
                dentro, pico, bloq = 0, 0.0, 0; out[i] = 0.0; continue
            if bloq > 0:
                bloq -= 1; out[i] = 0.0
                if bloq == 0: dentro, pico = 0, c.iloc[i]
                continue
            dentro += 1; pico = max(pico, c.iloc[i]); sair = False
            if saida == "tempo" and dentro > ps: sair = True
            if saida == "stop_atr" and np.isfinite(atr[i]) and c.iloc[i] < pico - ps * atr[i]: sair = True
            if saida == "vol_pct" and rvn[i] > lim:
                out[i] = 0.0; continue
            if sair: bloq = 5; out[i] = 0.0
            else: out[i] = w[i]
        w = out
    if tam == "vol" and fam not in ("vol", "conjunto_vol", "media_vol"):
        w = w * np.minimum(1.0, tv / rvn)
    w = np.r_[0.0, w[:-1]]
    cx = L.caixa_para(c.index, cls)
    giro = np.abs(np.diff(np.r_[0.0, w]))
    return pd.Series(w * r + (1 - w) * cx - giro * L.CUSTO[cls] / 2, index=c.index), meio


def mc(sem, bloco, n=2000, banca=1000.0, seed=7):
    rng = np.random.default_rng(seed)
    sem = np.asarray(sem); k = 52
    nb = int(math.ceil(k / bloco))
    ini = rng.integers(0, len(sem) - bloco + 1, size=(n, nb))
    am = np.stack([sem[ini[:, j][:, None] + np.arange(bloco)] for j in range(nb)], axis=1).reshape(n, -1)[:, :k]
    cam = banca * np.cumprod(1 + am, axis=1)
    fim = cam[:, -1]
    return dict(mediana=float(np.median(fim)), p5=float(np.percentile(fim, 5)), p95=float(np.percentile(fim, 95)),
                perde50=float(np.mean(cam.min(axis=1) <= banca / 2)), positivo=float(np.mean(fim > banca)))


linhas = []
for cls, (fam, p, sd, ps, tam) in ESCOLHAS.items():
    dados = {a: L.diario(a) for a in L.CLASSES[cls]}
    dados = {a: d[d["c"] > 0] for a, d in dados.items() if len(d) > 600}
    bh, st, cortes = {}, {}, []
    for a, d in dados.items():
        s, meio = pesos(d, cls, "segurar", None, "base", None, "fixo"); bh[a] = s.iloc[260:]; cortes.append(d.index[meio])
        st[a] = pesos(d, cls, fam, p, sd, ps, tam)[0].iloc[260:]
    corte = max(cortes)
    for nome, ser in (("comprar e segurar", pd.concat(bh, axis=1).mean(axis=1)), (f"{fam} {p or ''} · saída {sd} {ps or ''} · tamanho {tam}", pd.concat(st, axis=1).mean(axis=1))):
        ser = ser.dropna()
        sem_all = ((1 + ser).groupby(ser.index.to_period("W")).prod() - 1).values
        s2 = ser[ser.index >= corte]
        sem_2 = ((1 + s2).groupby(s2.index.to_period("W")).prod() - 1).values
        for fonte, sem, bloco in (("2ª metade, semanas soltas", sem_2, 1), ("série inteira, semanas soltas", sem_all, 1), ("série inteira, blocos de 4 semanas", sem_all, 4)):
            linhas.append(dict(mercado=L.NOME[cls], estrategia=nome, fonte=fonte, semanas=len(sem), de=str(ser.index[0].date()), **mc(sem, bloco)))
            L.conta()
df = pd.DataFrame(linhas)
df.to_csv(L.OUT + "/g3_mc.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 50)
print(df.round(3).to_string())
L.salva("g3_n.json", {"n": L.N_TESTES})
