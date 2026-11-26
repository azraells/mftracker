"""
MF Tracker — simulação da última semana, palpite por palpite.

Pedido do Luis (26/11/2026): voltar uma semana, fazer palpites com os dados disponíveis em cada data e hora,
conferir depois se acertaram e identificar os padrões que levaram a acertos e erros.

Como funciona:
  * candles de 1 hora: cripto e memecoins (Binance, cache do backtest_intradiario), B3, EUA e dólar (Yahoo, 1h, 2 anos);
  * em cada ponto da semana (cripto e dólar a cada 4 h; bolsas a cada hora de pregão) calcula 12 técnicas do guia
    de análise técnica (Murphy) só com dados até aquele momento, e forma o palpite como o guia manda:
    a DIREÇÃO vem da tendência diária, e os osciladores e padrões servem para o momento de entrada;
  * a chance do palpite vem do ano anterior à semana (mesma regra, mesmo mercado, mesma concordância entre técnicas);
  * também prevê a faixa de 50% (volatilidade ajustada, alfa do faixa.py);
  * confere em dois prazos: curto (4 candles) e um dia;
  * depois procura padrões: em que situações acertou e errou, e se esse padrão também valia no ano anterior
    (se não valia, provavelmente foi sorte da semana).

Uso: python semana.py               (dados reais)
     python semana.py --sintetico   (passeio aleatório, teste do código)
"""
import argparse, json, math, os, sys, time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

import backtest as B
import backtest_intradiario as BI

BRT = timezone(timedelta(hours=-3))
YAHOO_1H = [("PETR4", "b3", "PETR4.SA"), ("VALE3", "b3", "VALE3.SA"), ("ITUB4", "b3", "ITUB4.SA"), ("BBAS3", "b3", "BBAS3.SA"),
            ("BOVA11", "b3", "BOVA11.SA"), ("AAPL", "us", "AAPL"), ("NVDA", "us", "NVDA"), ("MSFT", "us", "MSFT"),
            ("TSLA", "us", "TSLA"), ("SPY", "us", "SPY"), ("QQQ", "us", "QQQ"), ("USD-BRL", "fx", "BRL=X")]
CUSTO = {"crypto": 0.003, "meme": 0.006, "b3": 0.001, "us": 0.001, "fx": 0.005}
MERCADO = {"crypto": "Cripto", "meme": "Memecoins", "b3": "B3", "us": "EUA", "fx": "Dólar"}
DIAS_SEMANA = 7
ALFA_FAIXA = 0.75
TECNICAS = {
    "mm_rapida": "Médias de 9 e 21 candles (cruzamento)",
    "tendencia_diaria": "Tendência diária (preço contra a média de 20 dias)",
    "rsi_na_tendencia": "RSI a favor da tendência (comprar sobrevendido em alta, vender sobrecomprado em baixa)",
    "rsi_extremo": "RSI extremo contra o movimento (abaixo de 30 compra, acima de 70 vende)",
    "estocastico": "Estocástico %D saindo de 20 ou de 80",
    "macd": "Histograma do MACD (lado do zero)",
    "bollinger": "Bandas de Bollinger (fora da banda, aposta na volta)",
    "rompimento": "Rompimento da máxima ou mínima de 20 candles",
    "rompimento_volume": "Rompimento confirmado por volume alto",
    "adx_regime": "ADX: segue a tendência se ADX > 25, aposta na volta se ADX < 20",
    "candle_filtrado": "Padrão de candle (engolfo, martelo) com estocástico em extremo",
    "pivo_diario": "Rompimento da máxima ou mínima do dia anterior",
}


# ---------------- dados ----------------
def carregar_yahoo():
    import yfinance as yf
    out = []
    for sym, cls, tk in YAHOO_1H:
        path = os.path.join(B.HIST, f"{sym}-1h.csv")
        try:
            if B.cache_ok(path, 6):
                df, origem = pd.read_csv(path), "cache"
            else:
                h = yf.Ticker(tk).history(period="730d", interval="1h", auto_adjust=True)
                if h is None or h.empty:
                    raise RuntimeError("Yahoo não devolveu dados")
                idx = h.index.tz_convert("UTC") if h.index.tz is not None else h.index.tz_localize("UTC")
                df = pd.DataFrame({"t": np.asarray((idx - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1), dtype="int64"), "o": h["Open"].to_numpy(), "h": h["High"].to_numpy(),
                                   "l": h["Low"].to_numpy(), "c": h["Close"].to_numpy(), "v": h["Volume"].to_numpy()})
                df = df.dropna(subset=["c"]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
                os.makedirs(B.HIST, exist_ok=True); df.to_csv(path, index=False); origem = "yahoo"
            B.log(f"{sym:7s} {len(df):6d} candles de 1h  {datetime.fromtimestamp(df['t'].iloc[0], timezone.utc):%Y-%m-%d} a "
                  f"{datetime.fromtimestamp(df['t'].iloc[-1], timezone.utc):%Y-%m-%d %H:%M} UTC  ({origem})")
            out.append({"sym": sym, "cls": cls, "df": df})
        except Exception as e:
            B.log(f"{sym:7s} FALHOU: {e}")
    return out


def sinteticos():
    out = BI.sinteticos()
    rng = np.random.default_rng(5)
    agora = int(time.time()) // 3600 * 3600
    for sym, cls, _ in YAHOO_1H:
        ts = [t for t in range(agora - 500 * 86400, agora, 3600)
              if datetime.fromtimestamp(t, timezone.utc).weekday() < 5 and (13 <= datetime.fromtimestamp(t, timezone.utc).hour < 20 or cls == "fx")]
        n = len(ts); r = rng.normal(0, 0.006, n); c = 50 * np.exp(np.cumsum(r)); o = np.r_[50, c[:-1]]
        out.append({"sym": sym, "cls": cls, "df": pd.DataFrame({"t": ts, "o": o, "h": np.maximum(o, c) * 1.002, "l": np.minimum(o, c) * 0.998,
                                                                 "c": c, "v": rng.lognormal(10, 0.5, n)})})
    return out


# ---------------- indicadores ----------------
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def indicadores(df):
    C, H, Lw, V, T = (df[k].to_numpy(float) for k in ("c", "h", "l", "v", "t"))
    O = df["o"].to_numpy(float)
    s = pd.Series(C)
    dias = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert("America/Sao_Paulo").dt.date
    bpd = max(1, int(round(pd.Series(dias).value_counts().median())))            # candles por dia
    d = np.r_[0, np.diff(C)]
    up = pd.Series(np.maximum(d, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    dn = pd.Series(np.maximum(-d, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50).to_numpy()
    lo14 = pd.Series(Lw).rolling(14).min(); hi14 = pd.Series(H).rolling(14).max()
    k = 100 * (s - lo14) / (hi14 - lo14).replace(0, np.nan)
    stoch_d = k.rolling(3).mean().rolling(3).mean().to_numpy()
    macd = ema(C, 12) - ema(C, 26); hist = macd - ema(macd, 9)
    m20 = s.rolling(20).mean().to_numpy(); sd20 = s.rolling(20).std(ddof=0).to_numpy()
    prevC = np.r_[C[0], C[:-1]]
    tr = np.maximum.reduce([H - Lw, np.abs(H - prevC), np.abs(Lw - prevC)])
    pdm = np.r_[0, np.where((H[1:] - H[:-1]) > (Lw[:-1] - Lw[1:]), np.maximum(H[1:] - H[:-1], 0), 0)]
    ndm = np.r_[0, np.where((Lw[:-1] - Lw[1:]) > (H[1:] - H[:-1]), np.maximum(Lw[:-1] - Lw[1:], 0), 0)]
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pd.Series(pdm).ewm(alpha=1 / 14, adjust=False).mean() / atr
    ndi = 100 * pd.Series(ndm).ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    lv = np.log1p(np.maximum(V, 0)); volz = ((pd.Series(lv) - pd.Series(lv).rolling(20).mean()) / pd.Series(lv).rolling(20).std()).to_numpy()
    hi20 = pd.Series(H).rolling(20).max().shift(1).to_numpy(); lo20 = pd.Series(Lw).rolling(20).min().shift(1).to_numpy()
    dia_sma = s.rolling(20 * bpd, min_periods=10 * bpd).mean().to_numpy()
    # máxima e mínima do dia anterior (pregão anterior)
    g = pd.DataFrame({"dia": dias, "h": H, "l": Lw})
    agg = g.groupby("dia").agg(h=("h", "max"), l=("l", "min"))
    prev = agg.shift(1)
    ph = g["dia"].map(prev["h"]).to_numpy(float); pl = g["dia"].map(prev["l"]).to_numpy(float)
    lr = np.r_[0.0, np.diff(np.log(C))]
    sig = np.sqrt(pd.Series(lr ** 2).ewm(alpha=0.06, adjust=False).mean().to_numpy())
    corpo = C - O; corpo_prev = np.r_[0, corpo[:-1]]
    O_prev = np.r_[O[0], O[:-1]]; C_prev = prevC
    rng_ = np.maximum(H - Lw, 1e-12)
    eng_alta = (corpo > 0) & (corpo_prev < 0) & (C >= O_prev) & (O <= C_prev)
    eng_baixa = (corpo < 0) & (corpo_prev > 0) & (C <= O_prev) & (O >= C_prev)
    martelo = ((np.minimum(O, C) - Lw) >= 2 * np.abs(corpo)) & ((H - np.maximum(O, C)) <= 0.3 * rng_)
    estrela = ((H - np.maximum(O, C)) >= 2 * np.abs(corpo)) & ((np.minimum(O, C) - Lw) <= 0.3 * rng_)
    return {"C": C, "T": T, "bpd": bpd, "dias": np.array(dias), "rsi": rsi, "stoch_d": stoch_d, "hist": hist, "m20": m20, "sd20": sd20,
            "adx": adx, "volz": volz, "hi20": hi20, "lo20": lo20, "dia_sma": dia_sma, "ph": ph, "pl": pl, "sig": sig,
            "ema9": ema(C, 9), "ema21": ema(C, 21), "cand_alta": eng_alta | martelo, "cand_baixa": eng_baixa | estrela}


def votos(I, i):
    """Voto de cada técnica no candle i (+1 alta, -1 baixa, 0 neutro), só com dados até i."""
    C = I["C"]; c = C[i]; v = {}
    sg = lambda x: 0 if not np.isfinite(x) or x == 0 else (1 if x > 0 else -1)
    v["mm_rapida"] = sg(I["ema9"][i] - I["ema21"][i])
    td = sg(c - I["dia_sma"][i]); v["tendencia_diaria"] = td
    r = I["rsi"][i]
    v["rsi_na_tendencia"] = 1 if td > 0 and r < 40 else (-1 if td < 0 and r > 60 else 0)
    v["rsi_extremo"] = 1 if r < 30 else (-1 if r > 70 else 0)
    d0, d1 = I["stoch_d"][i], I["stoch_d"][i - 1]
    v["estocastico"] = 1 if (d1 < 20 <= d0) else (-1 if (d1 > 80 >= d0) else 0)
    v["macd"] = sg(I["hist"][i])
    lo, hi = I["m20"][i] - 2 * I["sd20"][i], I["m20"][i] + 2 * I["sd20"][i]
    v["bollinger"] = 1 if c < lo else (-1 if c > hi else 0)
    rp = 1 if c > I["hi20"][i] else (-1 if c < I["lo20"][i] else 0)
    v["rompimento"] = rp
    v["rompimento_volume"] = rp if (rp and np.isfinite(I["volz"][i]) and I["volz"][i] > 1) else 0
    a = I["adx"][i]
    v["adx_regime"] = v["mm_rapida"] if (np.isfinite(a) and a > 25) else ((v["bollinger"] or v["rsi_extremo"]) if (np.isfinite(a) and a < 20) else 0)
    sd = I["stoch_d"][i]
    v["candle_filtrado"] = 1 if (I["cand_alta"][i] and sd < 20) else (-1 if (I["cand_baixa"][i] and sd > 80) else 0)
    v["pivo_diario"] = 1 if (np.isfinite(I["ph"][i]) and c > I["ph"][i]) else (-1 if (np.isfinite(I["pl"][i]) and c < I["pl"][i]) else 0)
    return v


def palpite(v):
    """Como o guia manda: direção pela tendência diária; se não houver, pela maioria. Concordância = a favor - contra."""
    soma = sum(v.values())
    d = v["tendencia_diaria"] or (1 if soma > 0 else (-1 if soma < 0 else 0))
    if d == 0:
        return 0, 0
    conc = sum(1 for k, x in v.items() if k != "tendencia_diaria" and x == d) - sum(1 for k, x in v.items() if k != "tendencia_diaria" and x == -d)
    return d, conc


def faixa(I, i, h, rets_hist):
    """Faixa de 50% para h candles: quantis do retorno em desvios (vol^alfa) das últimas 1.200 janelas já encerradas."""
    if len(rets_hist) < 200:
        return None
    sigs = np.array([s for _, s in rets_hist]); rs = np.array([r for r, _ in rets_hist])
    z = rs / (sigs * math.sqrt(h)) ** ALFA_FAIXA
    esc = (I["sig"][i] * math.sqrt(h)) ** ALFA_FAIXA
    return float(np.quantile(z, 0.25) * esc), float(np.quantile(z, 0.75) * esc)


def bucket_conc(c):
    return "contra (≤ 0)" if c <= 0 else ("fraca (1 a 2)" if c <= 2 else "forte (3 ou mais)")


# ---------------- simulação ----------------
def simular(a, fim_semana, ini_semana):
    I = indicadores(a["df"]); C, T = I["C"], I["T"]; N = len(C)
    hs = {"curto": 4, "dia": I["bpd"]}
    passo = 4 if a["cls"] in ("crypto", "meme", "fx") else 1
    ano_ini = ini_semana - 365 * 86400
    linhas = []
    hist_faixa = {k: [] for k in hs}
    for i in range(max(30 * I["bpd"], 60), N):
        t = T[i]
        if t < ano_ini:
            continue
        semana = t >= ini_semana
        # janelas já encerradas para a faixa (sem olhar o futuro)
        for k, h in hs.items():
            j = i - h
            if j > 0:
                hist_faixa[k].append((C[i] / C[j] - 1, I["sig"][j])); hist_faixa[k] = hist_faixa[k][-1200:]
        if (i % passo) != 0:
            continue
        v = votos(I, i)
        d, conc = palpite(v)
        hora = datetime.fromtimestamp(int(t), BRT)
        rec = {"sym": a["sym"], "cls": a["cls"], "t": int(t), "w": {k: min(1.0, passo / h) for k, h in hs.items()}, "semana": semana, "dir": d, "conc": conc, "votos": v, "preco": float(C[i]),
               "rsi": float(I["rsi"][i]), "adx": float(I["adx"][i]) if np.isfinite(I["adx"][i]) else None, "sig": float(I["sig"][i]),
               "hora_brt": hora.hour, "dia_semana": hora.weekday()}
        for k, h in hs.items():
            if i + h < N and T[i + h] <= fim_semana:
                rec[f"r_{k}"] = float(C[i + h] / C[i] - 1)
                rec[f"faixa_{k}"] = faixa(I, i, h, hist_faixa[k]) if semana else None
        linhas.append(rec)
    return linhas


def n_efetivo(xs, k):
    """Palpites que se sobrepõem no tempo não são independentes: desconta a sobreposição do prazo e,
    como ativos do mesmo mercado andam juntos, conta cada hora de cada mercado uma vez só."""
    por_hora = {}
    for l in xs:
        por_hora[(l["cls"], l["t"])] = l["w"][k]
    return max(1.0, sum(por_hora.values()))


def taxa(ls, k, filtro=lambda l: True):
    """Acerto da direção (bruto) e líquido de custo, para palpites com direção."""
    xs = [l for l in ls if l["dir"] != 0 and f"r_{k}" in l and filtro(l)]
    if not xs:
        return None
    hit = np.array([l["dir"] * l[f"r_{k}"] > 0 for l in xs], float)
    liq = np.array([l["dir"] * l[f"r_{k}"] - CUSTO[l["cls"]] for l in xs])
    p = hit.mean(); n = len(xs); ne = n_efetivo(xs, k)
    return {"n": n, "acerto": p, "ic": 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / ne), "liq": float(liq.mean()), "liq_pos": float((liq > 0).mean())}


def taxa_tecnica(ls, k, nome):
    xs = [l for l in ls if l["votos"][nome] != 0 and f"r_{k}" in l]
    if not xs:
        return None
    hit = np.array([l["votos"][nome] * l[f"r_{k}"] > 0 for l in xs], float)
    p = hit.mean(); n = len(xs)
    return {"n": n, "acerto": p, "ic": 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n_efetivo(xs, k))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sintetico", action="store_true"); args = ap.parse_args()
    ini = time.time()
    if args.sintetico:
        ativos = sinteticos()
    else:
        ativos = [a for a in BI.carregar() if len(a["df"]) >= 3000] + carregar_yahoo()
    ativos = [a for a in ativos if len(a["df"]) >= 500]
    if not ativos:
        B.log("sem dados"); sys.exit(1)
    fim = max(int(a["df"]["t"].iloc[-1]) for a in ativos)
    ini_sem = fim - DIAS_SEMANA * 86400
    B.log(f"semana simulada: {datetime.fromtimestamp(ini_sem, BRT):%d/%m %H:%M} a {datetime.fromtimestamp(fim, BRT):%d/%m %H:%M} (horário de Brasília)")
    todas = []
    for a in ativos:
        ls = simular(a, fim, ini_sem); todas += ls
        n_sem = sum(1 for l in ls if l["semana"] and l["dir"] != 0)
        B.log(f"{a['sym']:7s} {n_sem:4d} palpites na semana, {sum(1 for l in ls if not l['semana']):5d} pontos no ano anterior")
    sem = [l for l in todas if l["semana"]]; ano = [l for l in todas if not l["semana"]]

    # chance de cada palpite: acerto do ano anterior com o mesmo mercado e a mesma faixa de concordância (encolhido para 50%)
    cal = {}
    for cls in set(l["cls"] for l in ano):
        for b in ("contra (≤ 0)", "fraca (1 a 2)", "forte (3 ou mais)"):
            st = taxa(ano, "curto", lambda l, c=cls, b=b: l["cls"] == c and bucket_conc(l["conc"]) == b)
            if st:
                k = st["n"] / (st["n"] + 100)
                cal[(cls, b)] = 0.5 + (st["acerto"] - 0.5) * k
    for l in sem:
        l["chance"] = cal.get((l["cls"], bucket_conc(l["conc"])), 0.5)

    hoje = datetime.now(BRT)
    L = ["---", "tipo: simulacao", "gerado_por: MF Tracker semana.py", f"data: {hoje:%Y-%m-%dT%H:%M}-03:00",
         f"dados: {'SINTÉTICOS (teste do código)' if args.sintetico else 'reais'}", "tags: [simulacao, trading, palpites]", "---", "",
         f"# Simulação da semana: palpites e conferência — {hoje:%d/%m/%Y}", ""]
    if args.sintetico:
        L += ["> **Dados sintéticos** (passeio aleatório). O resultado certo é acerto perto de 50% e nenhum padrão que se repita no ano anterior.", ""]
    L += [f"Semana de **{datetime.fromtimestamp(ini_sem, BRT):%d/%m/%Y %H:%M}** a **{datetime.fromtimestamp(fim, BRT):%d/%m/%Y %H:%M}** (Brasília). "
          "Em cada ponto, o programa só via os candles até aquela hora. O palpite segue o guia de análise técnica (Murphy): "
          "**a direção vem da tendência diária**, e as outras 11 técnicas dizem se o momento é bom (concordância). "
          "Prazos: **curto** = 4 candles de 1 h; **dia** = 1 dia de pregão (24 h em cripto e dólar). Acerto = preço andou na direção do palpite (antes do custo). "
          "A chance de cada palpite vem do ano anterior à semana, com a mesma regra.", ""]
    L += ["## Como os mercados andaram na semana", "", "| Ativo | Mercado | Variação na semana | Palpites | Acerto (curto) | Acerto (dia) |", "|---|---|---:|---:|---:|---:|"]
    for a in ativos:
        df = a["df"]; d0 = df[df["t"] >= ini_sem]
        if d0.empty:
            continue
        var = d0["c"].iloc[-1] / d0["c"].iloc[0] - 1
        ls = [l for l in sem if l["sym"] == a["sym"]]
        s1, s2 = taxa(ls, "curto"), taxa(ls, "dia")
        L.append(f"| {a['sym']} | {MERCADO[a['cls']]} | {B.pct(var, 1)} | {s1['n'] if s1 else 0} | {B.pc(s1['acerto']) if s1 else '—'} | {B.pc(s2['acerto']) if s2 else '—'} |")
    L += ["", "## Placar geral", "", "| Prazo | Mercado | Palpites | Acerto na semana | Margem (95%) | Acerto no ano anterior | Líquido médio na semana |", "|---|---|---:|---:|---:|---:|---:|"]
    for k, lab in (("curto", "4 candles"), ("dia", "1 dia")):
        for cls in [None] + sorted(set(l["cls"] for l in sem)):
            f = (lambda l: True) if cls is None else (lambda l, c=cls: l["cls"] == c)
            s, y = taxa(sem, k, f), taxa(ano, k, f)
            if s:
                L.append(f"| {lab} | {'**Todos**' if cls is None else MERCADO[cls]} | {s['n']} | {B.pc(s['acerto'])} | ±{s['ic']*100:.1f} p.p. | {B.pc(y['acerto']) if y else '—'} | {B.pct(s['liq'], 3)} |")
    L += ["", "## Cada técnica do guia: semana contra ano anterior (prazo curto)", "",
          "Só conta quando a técnica deu sinal. Uma técnica útil acerta acima de 50% **nos dois períodos**.", "",
          "| Técnica | Sinais na semana | Acerto na semana | Sinais no ano | Acerto no ano | Leitura |", "|---|---:|---:|---:|---:|---|"]
    for nome, desc in TECNICAS.items():
        s, y = taxa_tecnica(sem, "curto", nome), taxa_tecnica(ano, "curto", nome)
        if not s:
            continue
        if y and y["acerto"] - y["ic"] > 0.5 and s["acerto"] > 0.5:
            leit = "acertou na semana e tem histórico a favor"
        elif y and y["acerto"] + y["ic"] < 0.5:
            leit = "no ano erra mais do que acerta: sinal invertido"
        elif s["acerto"] - s["ic"] > 0.5:
            leit = "boa semana, mas no ano é cara ou coroa: provavelmente sorte"
        elif s["acerto"] + s["ic"] < 0.5:
            leit = "semana ruim, no ano é cara ou coroa: provavelmente azar"
        else:
            leit = "cara ou coroa"
        L.append(f"| {desc} | {s['n']} | {B.pc(s['acerto'])} | {y['n'] if y else 0} | {B.pc(y['acerto']) if y else '—'} | {leit} |")
    # padrões: situações
    L += ["", "## Em que situações acertou e errou (prazo curto)", "",
          "Para cada situação: acerto na semana e no ano anterior. **Padrão real** só quando os dois apontam para o mesmo lado com folga.", "",
          "| Situação | Palpites na semana | Acerto na semana | Acerto no ano | Leitura |", "|---|---:|---:|---:|---|"]
    def sigq(l):
        return l["sig"]
    sig_cut = {}
    for cls in set(l["cls"] for l in ano):
        xs = np.array([l["sig"] for l in ano if l["cls"] == cls])
        if len(xs) > 30:
            sig_cut[cls] = (np.quantile(xs, 1 / 3), np.quantile(xs, 2 / 3))
    def clima(l):
        c = sig_cut.get(l["cls"])
        return "?" if not c else ("calmo" if l["sig"] < c[0] else ("agitado" if l["sig"] > c[1] else "normal"))
    situacoes = [
        ("Palpite de alta (compra)", lambda l: l["dir"] > 0), ("Palpite de baixa (venda)", lambda l: l["dir"] < 0),
        ("Concordância contra (≤ 0)", lambda l: l["conc"] <= 0), ("Concordância fraca (1 a 2)", lambda l: 1 <= l["conc"] <= 2),
        ("Concordância forte (3 ou mais)", lambda l: l["conc"] >= 3),
        ("Mercado calmo", lambda l: clima(l) == "calmo"), ("Mercado normal", lambda l: clima(l) == "normal"), ("Mercado agitado", lambda l: clima(l) == "agitado"),
        ("ADX acima de 25 (tendência)", lambda l: l["adx"] is not None and l["adx"] > 25), ("ADX abaixo de 20 (lateral)", lambda l: l["adx"] is not None and l["adx"] < 20),
        ("RSI acima de 70", lambda l: l["rsi"] > 70), ("RSI abaixo de 30", lambda l: l["rsi"] < 30),
        ("Madrugada (0h a 8h Brasília)", lambda l: l["hora_brt"] < 8), ("Manhã e tarde (8h a 18h)", lambda l: 8 <= l["hora_brt"] < 18),
        ("Noite (18h a 24h)", lambda l: l["hora_brt"] >= 18), ("Fim de semana (cripto)", lambda l: l["dia_semana"] >= 5),
        ("Palpite contra o MACD", lambda l: l["votos"]["macd"] == -l["dir"]), ("Palpite a favor do MACD", lambda l: l["votos"]["macd"] == l["dir"]),
        ("Rompimento a favor do palpite", lambda l: l["votos"]["rompimento"] == l["dir"]), ("Bollinger contra o palpite", lambda l: l["votos"]["bollinger"] == -l["dir"]),
    ]
    achados = []
    for nome, f in situacoes:
        s, y = taxa(sem, "curto", f), taxa(ano, "curto", f)
        if not s or s["n"] < 10:
            continue
        real = y and ((y["acerto"] - y["ic"] > 0.5 and s["acerto"] > 0.5) or (y["acerto"] + y["ic"] < 0.5 and s["acerto"] < 0.5))
        if real:
            leit = "**padrão real**: " + ("ajuda a acertar" if s["acerto"] > 0.5 else "leva a errar")
        elif s["acerto"] - s["ic"] > 0.5:
            leit = "acertou na semana, mas não se repete no ano"
        elif s["acerto"] + s["ic"] < 0.5:
            leit = "errou na semana, mas não se repete no ano"
        else:
            leit = "sem diferença"
        achados.append((nome, s, y, leit))
        L.append(f"| {nome} | {s['n']} | {B.pc(s['acerto'])} | {B.pc(y['acerto']) if y else '—'} | {leit} |")
    # faixa
    L += ["", "## A faixa prevista acertou?", "", "| Prazo | Previsões | Caiu dentro da faixa de 50% |", "|---|---:|---:|"]
    for k, lab in (("curto", "4 candles"), ("dia", "1 dia")):
        xs = [l for l in sem if l.get(f"faixa_{k}") and f"r_{k}" in l]
        if xs:
            dentro = np.mean([l[f"faixa_{k}"][0] <= l[f"r_{k}"] <= l[f"faixa_{k}"][1] for l in xs])
            L.append(f"| {lab} | {len(xs)} | {B.pc(dentro)} (o certo é perto de 50%) |")
    # conclusões automáticas
    s_all, y_all = taxa(sem, "curto"), taxa(ano, "curto")
    reais = [a_ for a_ in achados if a_[3].startswith("**padrão real")]
    L += ["", "## O que a semana ensinou", ""]
    if s_all:
        L.append(f"- Acerto geral na semana: {B.pc(s_all['acerto'])} (±{s_all['ic']*100:.1f} p.p.), contra {B.pc(y_all['acerto']) if y_all else '—'} no ano anterior com a mesma regra. "
                 + ("Dentro da margem de cara ou coroa." if abs(s_all["acerto"] - 0.5) <= s_all["ic"] else "Fora da margem de cara ou coroa nesta semana."))
    L.append("- Padrões que se repetem no ano anterior: " + ("; ".join(f"{n} ({B.pc(s['acerto'])} na semana, {B.pc(y['acerto'])} no ano)" for n, s, y, _ in reais) if reais else "nenhum."))
    L += ["- Uma semana tem poucos palpites independentes: ativos do mesmo mercado andam juntos, então um padrão visto só na semana costuma ser sorte. Por isso cada achado é conferido no ano anterior.",
          "- Erros e acertos de cada palpite estão na nota de palpites, com as técnicas que votaram a favor e contra.", "",
          f"Tempo de execução: {time.time() - ini:.0f} s.", ""]
    md = "\n".join(L)
    # lista de palpites
    P = ["---", "tipo: simulacao", "gerado_por: MF Tracker semana.py", f"data: {hoje:%Y-%m-%dT%H:%M}-03:00", "tags: [simulacao, palpites]", "---", "",
         f"# Palpites da semana — {hoje:%d/%m/%Y}", "",
         "Cada linha: o que o programa via naquela hora, o palpite, a chance (do ano anterior) e o que aconteceu em 4 candles. "
         "A favor e contra: técnicas que concordaram ou discordaram da direção.", "",
         "| Quando (Brasília) | Ativo | Palpite | Chance | Preço | Faixa 50% | Depois de 4 candles | Resultado | A favor | Contra |", "|---|---|---|---:|---:|---|---:|---|---|---|"]
    curto = {"mm_rapida": "MM", "tendencia_diaria": "tend", "rsi_na_tendencia": "RSI-t", "rsi_extremo": "RSI-x", "estocastico": "esto", "macd": "MACD",
             "bollinger": "BB", "rompimento": "romp", "rompimento_volume": "romp+vol", "adx_regime": "ADX", "candle_filtrado": "candle", "pivo_diario": "pivô"}
    for l in sorted(sem, key=lambda x: (x["t"], x["sym"])):
        if l["dir"] == 0 or "r_curto" not in l:
            continue
        r = l["r_curto"]; ok = l["dir"] * r > 0
        fx = l.get("faixa_curto")
        a_f = ", ".join(curto[k] for k, x in l["votos"].items() if x == l["dir"] and k != "tendencia_diaria") or "—"
        c_f = ", ".join(curto[k] for k, x in l["votos"].items() if x == -l["dir"]) or "—"
        P.append(f"| {datetime.fromtimestamp(l['t'], BRT):%d/%m %H:%M} | {l['sym']} | {'alta' if l['dir'] > 0 else 'baixa'} | {B.pc(l['chance'])} | {l['preco']:.6g} | "
                 f"{(B.pct(fx[0], 2) + ' a ' + B.pct(fx[1], 2)) if fx else '—'} | {B.pct(r, 2)} | {'acerto' if ok else 'erro'} | {a_f} | {c_f} |")
    pal = "\n".join(P) + "\n"
    os.makedirs(B.SAIDA, exist_ok=True)
    open(os.path.join(B.SAIDA, "semana.md"), "w", encoding="utf-8").write(md)
    open(os.path.join(B.SAIDA, "semana_palpites.md"), "w", encoding="utf-8").write(pal)
    if not args.sintetico:
        try:
            vault = json.load(open(os.path.join(B.BASE, "config.json"), encoding="utf-8"))["vault"]
            if os.path.isdir(os.path.join(vault, "01 Projeto")):
                n1 = f"Simulação da semana {hoje:%Y-%m-%d %Hh%M}.md"; n2 = f"Palpites da semana {hoje:%Y-%m-%d %Hh%M}.md"
                open(os.path.join(vault, "01 Projeto", n1), "w", encoding="utf-8").write(md)
                open(os.path.join(vault, "01 Projeto", n2), "w", encoding="utf-8").write(pal)
                B.log(f"gravado no vault: 01 Projeto/{n1} e {n2}")
        except Exception as e:
            B.log(f"não gravei no vault: {e}")
    B.log(f"pronto em {time.time() - ini:.0f} s")


if __name__ == "__main__":
    main()
