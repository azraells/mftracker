"""Estudo F + G.
F: para as famílias de entrada/saída do estudo A, testa formas de sair e de dimensionar, sempre na
   carteira igual da classe (honesto: ativos da mesma classe andam juntos):
   saída base | + saída por tempo | + stop móvel em ATR | + saída quando a vol passa de um percentil;
   tamanho fixo (0/1) × tamanho pela vol. A combinação é escolhida na 1ª metade; só a 2ª conta.
G: Monte Carlo com 2000 caminhos de 1 ano por bootstrap de blocos semanais, usando só os retornos da
   2ª metade (fora da amostra), para cada estratégia final e para comprar e segurar."""
import itertools, math, json
import numpy as np, pandas as pd
import lib as L
import a_tempo as A

# todas as famílias do estudo A, com o parâmetro que a 1ª metade escolheu (nada escolhido olhando a 2ª)
_a = pd.read_csv(L.OUT + "/a_tempo.csv")
FAMS = {}
for _, r in _a[_a.fam != "segurar"].iterrows():
    p = None if pd.isna(r.p) else (int(r.p) if float(r.p) >= 2 else float(r.p))
    FAMS.setdefault(r.cls, []).append((r.fam, p))
SAIDAS = [("base", None), ("tempo", 20), ("tempo", 60), ("stop_atr", 3.0), ("stop_atr", 5.0), ("vol_pct", 0.8), ("vol_pct", 0.9)]
TAMANHO = ["fixo", "vol"]


def pesos(d, cls, fam, p, saida, ps, tam):
    c = d["c"]; r = c.pct_change().fillna(0).values
    per = 365 if cls in ("crypto", "meme") else 252
    rv = pd.Series(r).rolling(20).std().values * math.sqrt(per)
    meio = len(c) // 2
    tv = float(np.nanmedian(rv[:meio]))
    rvn = np.nan_to_num(rv, nan=tv)
    w = np.nan_to_num(A.sinais(c, fam, p, rvn, tv), nan=0.0)
    if saida != "base":
        atr = (d["h"] - d["l"]).rolling(14).mean().values
        lim = np.nanquantile(rv[:meio], ps) if saida == "vol_pct" else None
        out = np.zeros_like(w); dentro = 0; pico = 0.0; bloqueado = 0
        for i in range(len(w)):
            if w[i] <= 0:
                dentro, pico, bloqueado = 0, 0.0, 0
                out[i] = 0.0; continue
            if bloqueado > 0:  # depois de uma saída, espera 5 dias e recomeça a contagem
                bloqueado -= 1; out[i] = 0.0
                if bloqueado == 0: dentro, pico = 0, c.iloc[i]
                continue
            dentro += 1; pico = max(pico, c.iloc[i])
            sair = False
            if saida == "tempo" and dentro > ps: sair = True
            if saida == "stop_atr" and np.isfinite(atr[i]) and c.iloc[i] < pico - ps * atr[i]: sair = True
            if saida == "vol_pct" and rvn[i] > lim:
                out[i] = 0.0; continue  # fica fora enquanto a vol estiver alta (volta quando cair)
            if sair:
                bloqueado = 5; out[i] = 0.0
            else:
                out[i] = w[i]
        w = out
    if tam == "vol" and fam not in ("vol", "conjunto_vol", "media_vol"):
        w = w * np.minimum(1.0, tv / rvn)
    w = np.r_[0.0, w[:-1]]
    cx = L.caixa_para(c.index, cls)
    giro = np.abs(np.diff(np.r_[0.0, w]))
    return pd.Series(w * r + (1 - w) * cx - giro * L.CUSTO[cls] / 2, index=c.index), meio, pd.Series(cx, index=c.index)


def carteira(series):
    """carteira igual entre os ativos disponíveis em cada dia."""
    return pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna()


def mc(ret_d, anos=1, n=2000, banca=1000.0, seed=7):
    rng = np.random.default_rng(seed)
    sem = (1 + ret_d).groupby(ret_d.index.to_period("W")).prod() - 1
    sem = sem.values
    k = 52 * anos
    amostras = rng.choice(sem, size=(n, k), replace=True)
    fim = banca * np.prod(1 + amostras, axis=1)
    caminhos = banca * np.cumprod(1 + amostras, axis=1)
    minimo = caminhos.min(axis=1)
    return dict(mediana=float(np.median(fim)), p5=float(np.percentile(fim, 5)), p95=float(np.percentile(fim, 95)),
                prob_perde50=float(np.mean(minimo <= banca * 0.5)), prob_positivo=float(np.mean(fim > banca)), caminhos=n, semanas=len(sem))


finais, linhas = {}, []
for cls, fams in FAMS.items():
    dados = {a: L.diario(a) for a in L.CLASSES[cls]}
    dados = {a: d[d["c"] > 0] for a, d in dados.items() if len(d) > 600}
    per = 365 if cls in ("crypto", "meme") else 252
    bh_s, cortes, cx_s = {}, {}, {}
    for a, d in dados.items():
        s, meio, cxa = pesos(d, cls, "segurar", None, "base", None, "fixo"); bh_s[a] = s; cortes[a] = d.index[meio]; cx_s[a] = cxa
    corte = max(cortes.values())  # 2ª metade da carteira só depois do corte de todos os ativos
    BH = carteira(bh_s); CX = carteira(cx_s).reindex(BH.index).fillna(0.0)
    BH1, BH2 = BH[(BH.index < corte) & (BH.index >= BH.index[0] + pd.Timedelta(days=400))], BH[BH.index >= corte]
    grade = []
    for (fam, p), (sd, ps), tam in itertools.product(fams, SAIDAS, TAMANHO):
        ss = {a: pesos(d, cls, fam, p, sd, ps, tam)[0] for a, d in dados.items()}
        S = carteira(ss)
        S1, S2 = S.loc[BH1.index], S.loc[BH2.index]
        m1 = L.metricas_serie(S1.values, per, CX.loc[S1.index].values)
        grade.append(dict(fam=fam, p=p, saida=sd, ps=ps, tam=tam, sh1=m1["sharpe"], S=S))
        L.conta(len(dados))
    g = max(grade, key=lambda r: r["sh1"])
    S2 = g["S"].loc[BH2.index]
    # a mesma família, sem saídas extras e tamanho fixo (para ver o que as saídas acrescentam)
    gb = next(r for r in grade if r["fam"] == g["fam"] and r["saida"] == "base" and r["tam"] == "fixo")
    for nome, ser in (("comprar e segurar", BH2), (f"{g['fam']} {g['p']} · saída {g['saida']} {g['ps'] or ''} · tamanho {g['tam']}", S2),
                      (f"{gb['fam']} {gb['p']} · saída base · tamanho fixo", gb["S"].loc[BH2.index])):
        rf2 = CX.loc[BH2.index].values
        m = L.metricas_serie(ser.values, per, rf2)
        t = L.t_dif_sharpe_mensal(ser.values, BH2.values, BH2.index, rf2) if nome != "comprar e segurar" else 0.0
        lin = dict(cls=cls, estrategia=nome, caixa_ano=float(np.mean(rf2) * per), de=str(BH2.index[0].date()), ate=str(BH2.index[-1].date()), **m, t_vs_segurar=t, **{f"mc_{k}": v for k, v in mc(ser).items()})
        linhas.append(lin)
    finais[cls] = dict(escolha={k: g[k] for k in ("fam", "p", "saida", "ps", "tam")}, top5_1a_metade=[{k: r[k] for k in ("fam", "p", "saida", "ps", "tam", "sh1")} for r in sorted(grade, key=lambda r: -r["sh1"])[:5]])
    print(cls, finais[cls]["escolha"])

# candidatos fora do estudo A: carry de funding (carteira igual) e noite nos EUA (custo barato)
fc = pd.read_csv(L.OUT + "/d_funding_carteira.csv", index_col=0, parse_dates=True).iloc[:, 0]
m = L.metricas_serie(fc.values, 365, L.juro_eua_diario().reindex(fc.index, method="ffill").fillna(0).values)
linhas.append(dict(cls="crypto", estrategia="carry de funding (spot comprado + perpétuo vendido), carteira igual", de=str(fc.index[0].date()), ate=str(fc.index[-1].date()), **m, t_vs_segurar=np.nan, **{f"mc_{k}": v for k, v in mc(fc).items()}))
on = []
for a in L.CLASSES["us"]:
    d = L.diario(a); d = d[(d["o"] > 0) & (d["c"] > 0)]
    on.append((d["o"] / d["c"].shift(1) - 1).rename(a))
ON = pd.concat(on, axis=1).mean(axis=1).dropna(); ON2 = ON.iloc[len(ON) // 2:]
for nome, cu in (("EUA só à noite, custo 0,10%", L.CUSTO["us"]), ("EUA só à noite, custo 0,05%", L.CUSTO_BARATO["us"]), ("EUA só à noite, custo 0,02%", 0.0002)):
    s = ON2 - cu
    m = L.metricas_serie(s.values, 252, L.caixa_para(s.index, "us"))
    linhas.append(dict(cls="us", estrategia=nome, de=str(s.index[0].date()), ate=str(s.index[-1].date()), **m, t_vs_segurar=np.nan, **{f"mc_{k}": v for k, v in mc(s).items()}))

df = pd.DataFrame(linhas)
df.to_csv(L.OUT + "/f_final.csv", index=False)
L.salva("f_escolhas.json", finais)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60)
print(df[["cls", "estrategia", "cagr", "sharpe", "mdd", "t_vs_segurar", "mc_mediana", "mc_p5", "mc_p95", "mc_prob_perde50", "mc_prob_positivo"]].round(3).to_string())
print("simulações:", L.N_TESTES)
L.salva("f_n.json", {"n": L.N_TESTES})
