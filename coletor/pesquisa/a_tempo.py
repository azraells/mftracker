"""Estudo A: quando colocar e tirar o dinheiro (prazo de dias a meses), com caixa rendendo juro.
Famílias: média móvel, momento de série temporal, conjunto de prazos, rompimento de canal (Donchian),
controle de volatilidade, e conjunto × volatilidade. Parâmetros escolhidos na 1ª metade de cada ativo
(critério: ganho médio de Sharpe sobre segurar, na classe); só a 2ª metade conta."""
import itertools, math
import numpy as np, pandas as pd
import lib as L


def sinais(c, fam, p, rv, tv):
    n = len(c)
    if fam == "segurar":
        return np.ones(n)
    if fam == "media":
        m = c.rolling(p).mean()
        return (c > m).astype(float).values
    if fam == "momento":
        return (c / c.shift(p) - 1 > 0).astype(float).values
    if fam == "conjunto":
        lbs = [20, 40, 60, 120, 180, 250]
        return np.nanmean(np.vstack([(c / c.shift(k) - 1 > 0).astype(float).values for k in lbs]), axis=0)
    if fam == "donchian":
        hi = c.rolling(p).max().shift(1); lo = c.rolling(p // 2).min().shift(1)
        w = np.zeros(n); pos = 0.0
        for i in range(n):
            if np.isfinite(hi.iloc[i]) and c.iloc[i] > hi.iloc[i]: pos = 1.0
            elif np.isfinite(lo.iloc[i]) and c.iloc[i] < lo.iloc[i]: pos = 0.0
            w[i] = pos
        return w
    if fam == "vol":
        return np.minimum(1.0, tv * p / rv)
    if fam == "conjunto_vol":
        base = sinais(c, "conjunto", None, rv, tv)
        return base * np.minimum(1.0, tv * p / rv)
    if fam == "media_vol":
        base = sinais(c, "media", 200, rv, tv)
        return base * np.minimum(1.0, tv * p / rv)
    raise ValueError(fam)


GRADE = {
    "segurar": [None],
    "media": [50, 100, 150, 200],
    "momento": [21, 63, 126, 252],
    "conjunto": [None],
    "donchian": [20, 55, 100],
    "vol": [0.5, 0.75, 1.0],
    "conjunto_vol": [0.5, 0.75, 1.0],
    "media_vol": [0.5, 0.75, 1.0],
}


def simula(d, cls, fam, p):
    c = d["c"]
    r = c.pct_change().fillna(0).values
    per = 365 if cls in ("crypto", "meme") else 252
    rv = pd.Series(r).rolling(20).std().values * math.sqrt(per)
    rv = np.where(np.isfinite(rv) & (rv > 0), rv, np.nan)
    meio = len(c) // 2
    tv = float(np.nanmedian(rv[:meio]))  # vol "normal" medida só na 1ª metade
    w = sinais(c, fam, p, np.nan_to_num(rv, nan=tv), tv)
    w = np.nan_to_num(w, nan=0.0)
    w = np.r_[0.0, w[:-1]]  # decide no fechamento, vale a partir do próximo dia
    cx = L.caixa_para(c.index, cls)
    giro = np.abs(np.diff(np.r_[0.0, w]))
    ret = w * r + (1 - w) * cx - giro * L.CUSTO[cls] / 2
    return ret, meio, per, float(np.mean(giro) * per), cx


def main():
    res, escolhas, linhas = {}, {}, []
    for cls, ativos in L.CLASSES.items():
        dados = {a: L.diario(a) for a in ativos}
        dados = {a: d[d["c"] > 0] for a, d in dados.items() if len(d) > 600}
        # 1ª metade: escolhe parâmetro por família
        tab = {}
        for fam, ps in GRADE.items():
            for p in ps:
                dif = []
                for a, d in dados.items():
                    ret, meio, per, _, cx = simula(d, cls, fam, p)
                    bh = simula(d, cls, "segurar", None)[0]
                    m1 = L.metricas_serie(ret[260:meio], per, cx[260:meio]); b1 = L.metricas_serie(bh[260:meio], per, cx[260:meio])
                    dif.append(m1["sharpe"] - b1["sharpe"])
                    L.conta()
                tab[(fam, p)] = float(np.nanmean(dif))
        for fam in GRADE:
            if fam == "segurar":
                continue
            melhor = max(GRADE[fam], key=lambda p: tab[(fam, p)])
            escolhas[(cls, fam)] = melhor
        # 2ª metade
        for fam in GRADE:
            p = escolhas.get((cls, fam), None)
            por_ativo = []
            for a, d in dados.items():
                ret, meio, per, giro, cx = simula(d, cls, fam, p)
                bh = simula(d, cls, "segurar", None)[0]
                idx = d.index[meio:]
                m2 = L.metricas_serie(ret[meio:], per, cx[meio:]); b2 = L.metricas_serie(bh[meio:], per, cx[meio:])
                t = L.t_dif_sharpe_mensal(ret[meio:], bh[meio:], idx, cx[meio:]) if fam != "segurar" else 0.0
                por_ativo.append(dict(ativo=a, **{k: m2[k] for k in m2}, bh_sharpe=b2["sharpe"], bh_mdd=b2["mdd"], bh_cagr=b2["cagr"], t=t, giro=giro, de=str(idx[0].date())))
                L.conta()
            dfp = pd.DataFrame(por_ativo)
            res[(cls, fam)] = dfp
            linhas.append(dict(cls=cls, fam=fam, p=p, n=len(dfp),
                               dsh=float((dfp["sharpe"] - dfp["bh_sharpe"]).mean()),
                               dmdd=float((dfp["mdd"] - dfp["bh_mdd"]).mean()),
                               melhor_sh=int((dfp["sharpe"] > dfp["bh_sharpe"]).sum()),
                               menor_q=int((dfp["mdd"] > dfp["bh_mdd"]).sum()),
                               cagr=float(dfp["cagr"].mean()), bh_cagr=float(dfp["bh_cagr"].mean()),
                               t_medio=float(dfp["t"].mean()), t_pool=float(dfp["t"].mean() * math.sqrt(len(dfp))) if fam != "segurar" else 0.0,
                               giro=float(dfp["giro"].mean()), de=dfp["de"].min()))
    df = pd.DataFrame(linhas)
    df.to_csv(L.OUT + "/a_tempo.csv", index=False)
    pd.concat({f"{k[0]}|{k[1]}": v for k, v in res.items()}).to_csv(L.OUT + "/a_tempo_ativos.csv")
    print(df.to_string())
    print("simulações:", L.N_TESTES)
    L.salva("a_tempo_n.json", {"n": L.N_TESTES})


if __name__ == "__main__":
    main()
