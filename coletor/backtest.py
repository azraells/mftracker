"""
MF Tracker — backtest do método em anos de histórico.

Pergunta que ele responde: os fatores técnicos do MF Tracker têm vantagem real, depois dos custos,
ou os acertos seriam sorte?

Como funciona (sem olhar o futuro):
  * Para cada dia do histórico, o motor faz exatamente o que o HTML faz ao vivo, mas usando só os dados
    até aquele dia: calcula os fatores, procura casos parecidos no passado (só os que já tinham resultado
    conhecido), estima a chance e aplica os mesmos filtros (chance >= 55%, vantagem >= 2 p.p., etc.).
  * Se virar oportunidade, a "operação" fecha no prazo escolhido, com o preço real de fechamento, descontando custos.
  * Uma operação aberta por ativo de cada vez, como ao vivo.
  * Duas versões: "fixo" (pesos iniciais, sem autoajuste) e "aprendendo" (pesos e calibração se ajustam
    a cada operação encerrada, como o programa faz).

Fora do teste (não há histórico confiável grátis): notícias, contexto global e memecoins de DEX.

Uso:
  python backtest.py                 baixa o histórico (Binance via ccxt, Yahoo via yfinance) e roda
  python backtest.py --desde 2018    muda o início do histórico das ações (padrão 2012)
  python backtest.py --sintetico     roda com preços aleatórios (para testar o código; o resultado tem que dar "sem vantagem")
"""
import argparse, csv, json, math, os, sys, time, urllib.request
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

import motor as M

BASE = os.path.dirname(os.path.abspath(__file__))
DADOS = os.path.join(BASE, "dados")
HIST = os.path.join(DADOS, "historico")
SAIDA = os.path.join(DADOS, "backtest")
BRT = timezone(timedelta(hours=-3))
UA = {"User-Agent": "mf-tracker-backtest/0.1", "Accept": "application/json"}

ATIVOS = [
    # símbolo, classe, fonte, código na fonte
    ("BTC", "crypto", "ccxt", "BTC/USDT"), ("ETH", "crypto", "ccxt", "ETH/USDT"), ("SOL", "crypto", "ccxt", "SOL/USDT"), ("XRP", "crypto", "ccxt", "XRP/USDT"),
    ("DOGE", "meme", "ccxt", "DOGE/USDT"), ("SHIB", "meme", "ccxt", "SHIB/USDT"), ("PEPE", "meme", "ccxt", "PEPE/USDT"),
    ("BONK", "meme", "ccxt", "BONK/USDT"), ("WIF", "meme", "ccxt", "WIF/USDT"),
    ("PETR4", "b3", "yahoo", "PETR4.SA"), ("VALE3", "b3", "yahoo", "VALE3.SA"), ("ITUB4", "b3", "yahoo", "ITUB4.SA"), ("MGLU3", "b3", "yahoo", "MGLU3.SA"),
    ("BBAS3", "b3", "yahoo", "BBAS3.SA"), ("WEGE3", "b3", "yahoo", "WEGE3.SA"), ("BOVA11", "b3", "yahoo", "BOVA11.SA"),
    ("AAPL", "us", "yahoo", "AAPL"), ("QQQ", "us", "yahoo", "QQQ"), ("MSFT", "us", "yahoo", "MSFT"), ("NVDA", "us", "yahoo", "NVDA"),
    ("SPY", "us", "yahoo", "SPY"), ("TSLA", "us", "yahoo", "TSLA"), ("AMZN", "us", "yahoo", "AMZN"),
    ("USD-BRL", "fx", "yahoo", "BRL=X"),
]


def log(m):
    print(f"{datetime.now(BRT):%H:%M:%S} {m}", flush=True)


def get_json(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ------------------------------------------------------------------ dados
def cache_ok(path, horas=12):
    return os.path.exists(path) and time.time() - os.path.getmtime(path) < horas * 3600


def baixar_ccxt(par, desde_ms):
    import ccxt
    ultimo_erro = None
    for ex_id in ("binance", "kraken", "coinbaseexchange"):
        try:
            ex = getattr(ccxt, ex_id)({"enableRateLimit": True, "timeout": 30000})
            ex.load_markets()
            sym = par if par in ex.markets else par.replace("/USDT", "/USD")
            if sym not in ex.markets:
                continue
            rows, since = [], desde_ms
            while True:
                lote = ex.fetch_ohlcv(sym, "1d", since=since, limit=1000)
                if not lote:
                    break
                rows += lote
                nxt = lote[-1][0] + 86400000
                if nxt <= since or len(lote) < 2:
                    break
                since = nxt
                if since > time.time() * 1000:
                    break
            if rows:
                df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).drop_duplicates("t")
                df["t"] = df["t"] // 1000
                return df, ex_id
        except Exception as e:  # tenta a próxima exchange
            ultimo_erro = e
    raise RuntimeError(f"nenhuma exchange respondeu para {par}: {ultimo_erro}")


def baixar_yahoo(ticker, desde):
    import yfinance as yf
    h = yf.Ticker(ticker).history(start=f"{desde}-01-01", interval="1d", auto_adjust=True)
    if h is None or h.empty:
        raise RuntimeError(f"Yahoo sem dados para {ticker}")
    df = pd.DataFrame({"t": [int(pd.Timestamp(d.date()).tz_localize("UTC").timestamp()) for d in h.index],
                       "o": h["Open"].values, "h": h["High"].values, "l": h["Low"].values, "c": h["Close"].values,
                       "v": h["Volume"].fillna(0).values})
    return df.dropna(subset=["c"]), "yahoo"


def carregar(desde_acoes):
    os.makedirs(HIST, exist_ok=True)
    out = []
    for sym, cls, fonte, cod in ATIVOS:
        path = os.path.join(HIST, f"{sym}.csv")
        try:
            if cache_ok(path):
                df, origem = pd.read_csv(path), "cache"
            elif fonte == "ccxt":
                df, origem = baixar_ccxt(cod, int(datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp() * 1000))
            else:
                df, origem = baixar_yahoo(cod, desde_acoes)
            df = df.sort_values("t").reset_index(drop=True)
            # o candle de hoje ainda está em formação: fica fora
            df = df[df["t"] // 86400 < int(time.time()) // 86400].reset_index(drop=True)
            df.to_csv(path, index=False)
            log(f"{sym:8s} {len(df):5d} dias  {datetime.fromtimestamp(df['t'].iloc[0], timezone.utc):%Y-%m-%d} a "
                f"{datetime.fromtimestamp(df['t'].iloc[-1], timezone.utc):%Y-%m-%d}  ({origem})")
            out.append({"sym": sym, "cls": cls, "df": df})
        except Exception as e:
            log(f"{sym:8s} FALHOU: {e}")
    return out


def carregar_fng():
    path = os.path.join(HIST, "fng.json")
    try:
        if cache_ok(path):
            d = json.load(open(path, encoding="utf-8"))
        else:
            d = get_json("https://api.alternative.me/fng/?limit=0&format=json")["data"]
            json.dump(d, open(path, "w", encoding="utf-8"))
        return {int(x["timestamp"]) // 86400: int(x["value"]) for x in d}
    except Exception as e:
        log(f"Medo & Ganância indisponível ({e}); psicologia das criptos usa só o preço")
        return None


def sinteticos():
    rng = np.random.default_rng(42)
    out = []
    hoje = int(time.time()) // 86400 * 86400
    for sym, cls, *_ in ATIVOS:
        n = 2500
        r = rng.normal(0.0003, 0.03 if cls in ("crypto", "meme") else 0.015, n)
        c = 100 * np.exp(np.cumsum(r))
        o = np.r_[100, c[:-1]]
        df = pd.DataFrame({"t": hoje - (n - np.arange(n)) * 86400, "o": o, "h": np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n)),
                           "l": np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n)), "c": c, "v": rng.lognormal(10, 0.5, n)})
        out.append({"sym": sym, "cls": cls, "df": df})
    return out


# ------------------------------------------------------------------ simulação
def preparar(ativos, fng):
    for a in ativos:
        a["I"] = M.indicators(a["df"])
        a["F"] = M.factor_matrix(a["cls"], a["df"], a["I"], fng)
        a["days"] = (a["df"]["t"].to_numpy() // 86400).astype(int)
        a["idx"] = {int(d): i for i, d in enumerate(a["days"])}
    for cls in M.CLASSES:
        g = [a for a in ativos if a["cls"] == cls]
        if g:
            M.compute_niche(g)


def simular(ativos, aprender, usdbrl):
    w = dict(M.DEFAULT_W)
    closed, trades, abertas = [], [], {}
    todos = sorted(set().union(*[set(a["days"].tolist()) for a in ativos]))
    for d in todos:
        # 1) encerra as que vencem hoje
        for sym in [s for s, t in abertas.items() if t["exit_day"] == d]:
            t = abertas.pop(sym)
            a = t["a"]; C = a["I"]["C"]; i, h = t["i"], t["h"]
            ret = C[i + h] / C[i] - 1
            path = slice(i, i + h + 1)
            hi, lo = a["I"]["H"][path].max(), a["I"]["L"][path].min()
            t.update(ret=ret, retDir=t["dir"] * ret, retNet=t["dir"] * ret - t["cost"], hit=t["dir"] * ret - t["cost"] > 0,
                     long_net=ret - t["cost"], targetHit=(hi >= t["target"]) if t["dir"] > 0 else (lo <= t["target"]),
                     stopHit=(lo <= t["stop"]) if t["dir"] > 0 else (hi >= t["stop"]))
            if aprender:
                M.learn(w, t["factors"], math.copysign(1, ret) if ret != 0 else 1)
            closed.append(t); trades.append(t)
        cal = M.calibration_shift(closed) if aprender else 0.0
        # 2) procura oportunidades nos ativos sem operação aberta
        for a in ativos:
            i = a["idx"].get(d)
            if i is None or a["sym"] in abertas or i + 1 < M.WARM + 30:
                continue
            an = M.analyze_at(a["cls"], a["I"], a["F"], i, w, cal_shift=cal)
            if not an or not an["isOpp"]:
                continue
            h = an["best"]["h"]
            if i + h >= len(a["I"]["C"]):
                continue  # ainda não venceu: fica fora do placar
            preco_brl = an["price"] if a["cls"] == "b3" else None
            abertas[a["sym"]] = {"a": a, "sym": a["sym"], "cls": a["cls"], "i": i, "h": h, "exit_day": int(a["days"][i + h]),
                                 "date": datetime.fromtimestamp(int(a["days"][i]) * 86400, timezone.utc).strftime("%Y-%m-%d"),
                                 "dir": an["dir"], "p": an["p"], "base": an["best"]["base"], "p0": an["best"]["p0"], "n_analogos": an["best"]["n"],
                                 "exp": an["best"]["exp"], "frac": an["frac"], "cost": an["cost"], "target": an["target"], "stop": an["stop"],
                                 "score": an["sTech"], "factors": an["factors"], "banks": M.bank_plan(a["cls"], an, price_brl=preco_brl)}
    return trades, w


def ic_por_classe(ativos):
    """Correlação de postos entre a pontuação técnica (pesos iniciais) e o retorno seguinte, em todos os dias (sem filtro)."""
    out = {}
    for cls in M.CLASSES:
        g = [a for a in ativos if a["cls"] == cls]
        if not g:
            continue
        out[cls] = {}
        for h in M.HORIZONTES:
            vals = []
            for a in g:
                s = M.combine(a["F"], M.DEFAULT_W); C = a["I"]["C"]
                j = np.arange(M.WARM, len(C) - h)
                ok = np.isfinite(s[j])
                if ok.sum() < 60:
                    continue
                x = pd.Series(s[j][ok]); y = pd.Series(C[j + h][ok] / C[j][ok] - 1)
                vals.append(x.rank().corr(y.rank()))  # Spearman sem depender do scipy
            out[cls][h] = float(np.nanmean(vals)) if vals else None
    return out


# ------------------------------------------------------------------ relatório
def estat(ts):
    if not ts:
        return None
    r = np.array([t["retNet"] for t in ts]); hit = np.array([t["hit"] for t in ts], float)
    p = np.array([t["p"] for t in ts]); base = np.array([t["base"] for t in ts]); lg = np.array([t["long_net"] for t in ts])
    n = len(ts)
    sd = r.std(ddof=1) if n > 1 else float("nan")
    # z do acerto contra a taxa base (binomial aproximada)
    z_hit = (hit.mean() - base.mean()) / math.sqrt(max(base.mean() * (1 - base.mean()), 1e-9) / n)
    return {"n": n, "acerto": hit.mean(), "chance_prevista": p.mean(), "taxa_base": base.mean(), "brier": float(((p - hit) ** 2).mean()),
            "ret_medio": r.mean(), "t": r.mean() / sd * math.sqrt(n) if n > 1 and sd > 0 else float("nan"), "z_acerto": z_hit,
            "comprar_sempre": lg.mean(), "soma": r.sum(), "alvo": np.mean([t["targetHit"] for t in ts]), "stop": np.mean([t["stopHit"] for t in ts])}


def bancas(ts):
    out = []
    for bank in M.DEFAULT_SETTINGS["banks"]:
        eq, pico, mdd, ops = bank, bank, 0.0, 0
        for t in sorted(ts, key=lambda x: x["exit_day"]):
            b = next(x for x in t["banks"] if x["bank"] == bank)
            if not b["ok"]:
                continue
            ops += 1
            eq += b["amount"] * t["retNet"]
            pico = max(pico, eq); mdd = max(mdd, 1 - eq / pico)
        out.append({"bank": bank, "final": eq, "retorno": eq / bank - 1, "ops": ops, "mdd": mdd})
    return out


def pct(x, casas=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{casas}f}%".replace(".", ",")


def pc(x, casas=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{casas}f}%".replace(".", ",")


def num(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}".replace(".", ",")


def brl(x):
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def veredito(s):
    if not s or s["n"] < 30:
        return "Poucas operações para concluir qualquer coisa."
    if s["t"] >= 2 and s["z_acerto"] >= 2:
        return "Há sinal de vantagem estatística depois dos custos (t ≥ 2 e acerto acima da taxa base). Ainda assim, é passado: confirme ao vivo."
    if s["t"] >= 2:
        return "O retorno médio é positivo com significância, mas o acerto não supera a taxa base com folga: o ganho vem de poucos movimentos grandes."
    if s["ret_medio"] > 0:
        return "Retorno médio positivo, mas sem significância estatística (t < 2): não dá para separar de sorte."
    return "Sem vantagem: o retorno médio depois dos custos é zero ou negativo. Do jeito que está, o método não deveria receber dinheiro real."


def relatorio(ativos, res, ic, sintetico, ini):
    hoje = datetime.now(BRT)
    L = [f"---", f"tipo: backtest", f"gerado_por: MF Tracker backtest (motor {M.VERSAO_MOTOR})", f"data: {hoje:%Y-%m-%dT%H:%M}-03:00",
         f"dados: {'SINTÉTICOS (teste do código)' if sintetico else 'reais'}", f"tags: [backtest, metodologia]", f"---", "",
         f"# Backtest do MF Tracker — {hoje:%d/%m/%Y}", ""]
    if sintetico:
        L += ["> **Dados sintéticos (preços aleatórios).** Serve só para testar o código: o resultado certo aqui é \"sem vantagem\".", ""]
    L += ["## Resposta curta", ""]
    for nome, (ts, w) in res.items():
        s = estat(ts)
        L.append(f"- **{nome}**: {veredito(s)}" + (f" ({s['n']} operações, acerto {pc(s['acerto'])} contra taxa base {pc(s['taxa_base'])}, "
                                                      f"retorno médio {pct(s['ret_medio'], 2)} por operação, t = {num(s['t'])})" if s else ""))
    L += ["", "Como ler: **t** mede se o retorno médio é diferente de zero além do acaso (abaixo de 2 é indistinguível de sorte). "
          "**Taxa base** é quanto a mesma direção acertaria sem filtro nenhum. **Comprar sempre** é o resultado de apenas comprar nos mesmos dias.", ""]
    L += ["## Dados usados", "", "| Ativo | Classe | Dias | De | Até |", "|---|---|---:|---|---|"]
    for a in ativos:
        t = a["df"]["t"]
        L.append(f"| {a['sym']} | {M.CLASSES[a['cls']]['label']} | {len(t)} | {datetime.fromtimestamp(t.iloc[0], timezone.utc):%Y-%m-%d} | "
                 f"{datetime.fromtimestamp(t.iloc[-1], timezone.utc):%Y-%m-%d} |")
    L += ["", "Fora do teste: notícias e contexto global (sem histórico grátis confiável; ao vivo mexem no máximo ±7 p.p. na chance) e memecoins de DEX (histórico curto demais).", ""]
    for nome, (ts, w) in res.items():
        L += [f"## Versão {nome}", "", "| Classe | Operações | Acerto | Chance prevista | Taxa base | Retorno médio | t | Comprar sempre | Tocou alvo | Tocou stop |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for cls in list(M.CLASSES) + ["total"]:
            g = ts if cls == "total" else [t for t in ts if t["cls"] == cls]
            s = estat(g)
            if not s:
                continue
            L.append(f"| {'**Total**' if cls == 'total' else M.CLASSES[cls]['label']} | {s['n']} | {pc(s['acerto'])} | {pc(s['chance_prevista'])} | "
                     f"{pc(s['taxa_base'])} | {pct(s['ret_medio'], 2)} | {num(s['t'])} | {pct(s['comprar_sempre'], 2)} | {pc(s['alvo'], 0)} | {pc(s['stop'], 0)} |")
        # estabilidade no tempo
        if len(ts) >= 20:
            o = sorted(ts, key=lambda t: t["exit_day"]); m = len(o) // 2
            s1, s2 = estat(o[:m]), estat(o[m:])
            L += ["", f"Estabilidade: primeira metade das operações (até {o[m-1]['date']}) teve retorno médio {pct(s1['ret_medio'], 2)} e acerto {pc(s1['acerto'])}; "
                  f"a segunda metade, {pct(s2['ret_medio'], 2)} e {pc(s2['acerto'])}. Se as duas metades discordam muito, o resultado depende do período."]
        # calibração
        L += ["", "**Calibração** (quando o modelo disse X%, acertou quanto?)", "", "| Faixa prevista | Casos | Acerto real |", "|---|---:|---:|"]
        for lo, hi in [(0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.8), (0.8, 1.01)]:
            g = [t for t in ts if lo <= t["p"] < hi]
            if g:
                L.append(f"| {lo*100:.0f}–{min(hi,1)*100:.0f}% | {len(g)} | {pc(np.mean([t['hit'] for t in g]), 0)} |")
        # bancas
        L += ["", "**Bancas** (mesma regra do programa: 1/4 de Kelly, máximo 10% por operação, mínimo por ordem)", "",
              "| Banca | Final | Retorno | Operou | Maior queda |", "|---:|---:|---:|---:|---:|"]
        for b in bancas(ts):
            L.append(f"| {brl(b['bank'])} | {brl(b['final'])} | {pct(b['retorno'])} | {b['ops']} | {pc(b['mdd'])} |")
        if nome == "aprendendo":
            L += ["", "Pesos no fim do teste (começam em: " + ", ".join(f"{k} {M.DEFAULT_W[k]:.2f}" for k in M.BT_KEYS) + "):",
                  "", ", ".join(f"{k} **{w[k]:.2f}**" for k in M.BT_KEYS)]
        L.append("")
    L += ["## Poder de previsão da pontuação, sem filtro nenhum", "",
          "Correlação de postos entre a pontuação técnica de cada dia e o retorno dos dias seguintes (todos os dias, não só as oportunidades). "
          "Zero = nenhuma relação. Em mercados, 0,02 a 0,05 já é útil; abaixo de 0,01 é ruído. Sinal negativo = os fatores apontam ao contrário.", "",
          "| Classe | 3 dias | 7 dias | 14 dias |", "|---|---:|---:|---:|"]
    for cls, d in ic.items():
        L.append(f"| {M.CLASSES[cls]['label']} | " + " | ".join("—" if d.get(h) is None else f"{d[h]:+.3f}".replace(".", ",") for h in M.HORIZONTES) + " |")
    L += ["", "## Limites deste teste (autocrítica)", "",
          "- Os fatores e seus pesos iniciais foram escolhidos por mim antes do teste, mas conhecendo a literatura; não houve ajuste fino aos dados, o que reduz (não elimina) o risco de excesso de ajuste.",
          "- Custos são aproximações fixas por classe; não incluem impostos, spread em momentos de pânico nem atraso de execução.",
          "- Preço de entrada e saída = fechamento do dia. Ao vivo o registro acontece durante o dia, a preço corrente.",
          "- Ações de memecoins sobreviventes (DOGE, SHIB, PEPE, BONK, WIF) têm viés de sobrevivência: as que morreram não estão aqui.",
          "- Notícias, contexto global e psicologia além do Medo & Ganância não entram.",
          f"", f"Tempo de execução: {time.time() - ini:.0f} s. Operações completas em `coletor/dados/backtest/`.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=2012, help="ano inicial para B3, EUA e dólar")
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--sem-vault", action="store_true", help="não grava a nota no vault")
    args = ap.parse_args()
    ini = time.time()
    if args.sintetico:
        ativos, fng = sinteticos(), None
    else:
        ativos, fng = carregar(args.desde), carregar_fng()
    ativos = [a for a in ativos if len(a["df"]) >= M.WARM + 60]
    if not ativos:
        log("nenhum ativo com histórico suficiente"); sys.exit(1)
    log("calculando fatores...")
    preparar(ativos, fng)
    res = {}
    for nome, aprender in (("fixo", False), ("aprendendo", True)):
        log(f"simulando versão {nome}...")
        ts, w = simular(ativos, aprender, None)
        res[nome] = (ts, w)
        s = estat(ts)
        if s:
            log(f"  {s['n']} operações | acerto {s['acerto']*100:.1f}% (base {s['taxa_base']*100:.1f}%) | retorno médio {s['ret_medio']*100:+.2f}% | t {s['t']:.2f}")
    ic = ic_por_classe(ativos)
    os.makedirs(SAIDA, exist_ok=True)
    for nome, (ts, _) in res.items():
        with open(os.path.join(SAIDA, f"operacoes-{nome}.csv"), "w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            cw.writerow(["data", "ativo", "classe", "direcao", "prazo_dias", "chance", "taxa_base", "analogos", "pontuacao", "retorno_liquido", "acerto"])
            for t in ts:
                cw.writerow([t["date"], t["sym"], t["cls"], t["dir"], t["h"], round(t["p"], 4), round(t["base"], 4), t["n_analogos"],
                             round(t["score"], 4), round(t["retNet"], 5), int(t["hit"])])
    md = relatorio(ativos, res, ic, args.sintetico, ini)
    open(os.path.join(SAIDA, "relatorio.md"), "w", encoding="utf-8").write(md)
    if not args.sem_vault and not args.sintetico:
        try:
            vault = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))["vault"]
            pasta = os.path.join(vault, "01 Projeto")
            if os.path.isdir(pasta):
                nome = f"Backtest {datetime.now(BRT):%Y-%m-%d}.md"
                open(os.path.join(pasta, nome), "w", encoding="utf-8").write(md)
                log(f"relatório gravado no vault: 01 Projeto/{nome}")
        except Exception as e:
            log(f"não gravei no vault: {e}")
    log(f"pronto em {time.time() - ini:.0f} s. Relatório: {os.path.join(SAIDA, 'relatorio.md')}")


if __name__ == "__main__":
    main()
