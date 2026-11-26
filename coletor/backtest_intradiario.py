"""
MF Tracker — backtest para trading de curto prazo (candles de 1 hora).

Pergunta: em prazos de 1, 4, 12 e 24 horas, o método do MF Tracker (fatores técnicos + casos parecidos,
com a chance corrigida) acerta mais do que o acaso, depois dos custos?

Igual ao backtest diário, sem olhar o futuro:
  * cada hora, para cada ativo sem posição aberta, o motor analisa com os dados até aquela hora;
  * casos parecidos só do último ano (8.760 horas) e só com resultado já conhecido;
  * versão "todos os fatores" (pesos iniciais) no histórico todo;
  * versão "validada": fatores escolhidos na 1ª metade do histórico de cada classe e testados só na 2ª.

Grava, além do relatório, 99 Meta/validacao.json no vault: é dali que a aba Trading tira o selo de cada
combinação de mercado e prazo (verde só se passou fora da amostra).

Uso: python backtest_intradiario.py            (baixa 1h da Binance via ccxt, desde 2021)
     python backtest_intradiario.py --sintetico
"""
import argparse, json, math, os, sys, time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

import motor as M
import backtest as B

BRT = timezone(timedelta(hours=-3))
TF = 3600
HORAS = [1, 4, 12, 24]
MAX_HIST = 24 * 365
ATIVOS_1H = [("BTC", "crypto", "BTC/USDT"), ("ETH", "crypto", "ETH/USDT"), ("SOL", "crypto", "SOL/USDT"), ("XRP", "crypto", "XRP/USDT"),
             ("DOGE", "meme", "DOGE/USDT"), ("SHIB", "meme", "SHIB/USDT"), ("PEPE", "meme", "PEPE/USDT"), ("BONK", "meme", "BONK/USDT"),
             ("WIF", "meme", "WIF/USDT")]


def baixar_1h(par, desde_ms):
    import ccxt
    ultimo = None
    for ex_id in ("binance", "kraken", "coinbaseexchange"):
        try:
            ex = getattr(ccxt, ex_id)({"enableRateLimit": True, "timeout": 30000})
            ex.load_markets()
            sym = par if par in ex.markets else par.replace("/USDT", "/USD")
            if sym not in ex.markets:
                continue
            rows, since = [], desde_ms
            while since < time.time() * 1000:
                lote = ex.fetch_ohlcv(sym, "1h", since=since, limit=1000)
                if not lote:
                    break
                rows += lote
                nxt = lote[-1][0] + 3600000
                if nxt <= since:
                    break
                since = nxt
            if rows:
                df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).drop_duplicates("t")
                df["t"] = df["t"] // 1000
                return df, ex_id
        except Exception as e:
            ultimo = e
    raise RuntimeError(f"nenhuma exchange respondeu para {par}: {ultimo}")


def carregar():
    os.makedirs(B.HIST, exist_ok=True)
    out = []
    for sym, cls, par in ATIVOS_1H:
        path = os.path.join(B.HIST, f"{sym}-1h.csv")
        try:
            if B.cache_ok(path, 12):
                df, origem = pd.read_csv(path), "cache"
            else:
                df, origem = baixar_1h(par, int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000))
            df = df.sort_values("t").reset_index(drop=True)
            df = df[df["t"] // TF < int(time.time()) // TF].reset_index(drop=True)  # tira a hora em formação
            df.to_csv(path, index=False)
            B.log(f"{sym:6s} {len(df):6d} horas  {datetime.fromtimestamp(df['t'].iloc[0], timezone.utc):%Y-%m-%d} a "
                  f"{datetime.fromtimestamp(df['t'].iloc[-1], timezone.utc):%Y-%m-%d}  ({origem})")
            out.append({"sym": sym, "cls": cls, "df": df})
        except Exception as e:
            B.log(f"{sym:6s} FALHOU: {e}")
    return out


def sinteticos():
    rng = np.random.default_rng(7)
    out = []
    agora = int(time.time()) // TF * TF
    for sym, cls, _ in ATIVOS_1H:
        n = 20000
        r = rng.normal(0.00002, 0.008 if cls == "crypto" else 0.014, n)
        c = 100 * np.exp(np.cumsum(r)); o = np.r_[100, c[:-1]]
        out.append({"sym": sym, "cls": cls, "df": pd.DataFrame({"t": agora - (n - np.arange(n)) * TF, "o": o,
                    "h": np.maximum(o, c) * (1 + rng.uniform(0, 0.003, n)), "l": np.minimum(o, c) * (1 - rng.uniform(0, 0.003, n)),
                    "c": c, "v": rng.lognormal(10, 0.5, n)})})
    return out


def preparar(ativos):
    for a in ativos:
        a["I"] = M.indicators(a["df"])
        a["F"] = M.factor_matrix(a["cls"], a["df"], a["I"], None)
        N = len(a["df"]); a["mid"] = M.WARM + (N - M.WARM) // 2
        C = a["I"]["C"]
        a["mu"] = {h: float(np.mean(C[h:] / C[:-h] - 1)) for h in HORAS}
    for cls in ("crypto", "meme"):
        g = [a for a in ativos if a["cls"] == cls]
        if g:
            M.compute_niche(g, TF)


def ic_metades(ativos, h):
    """Como backtest.ic_fatores, mas para candles de 1h e prazo h horas."""
    out = {}
    for cls in ("crypto", "meme"):
        g = [a for a in ativos if a["cls"] == cls]
        if not g:
            continue
        out[cls] = {}
        for fi, k in enumerate(M.BT_KEYS):
            halves, neff = [[], []], [0.0, 0.0]
            for a in g:
                C = a["I"]["C"]; j = np.arange(M.WARM, len(C) - h); x = a["F"][j, fi]; ok = np.isfinite(x); j, x = j[ok], x[ok]
                if len(j) < 500:
                    continue
                y = C[j + h] / C[j] - 1; m = len(j) // 2
                for hi_, sl in enumerate((slice(0, m), slice(m, None))):
                    xs, ys = pd.Series(x[sl]), pd.Series(y[sl])
                    if xs.std() > 0:
                        halves[hi_].append(xs.rank().corr(ys.rank())); neff[hi_] += len(xs) / h
            out[cls][k] = tuple((float(np.nanmean(v)), 1 / math.sqrt(ne)) if v and ne > 0 else (None, None) for v, ne in zip(halves, neff))
    return out


def simular_ativo(a, h, F, w, so_segunda):
    C = a["I"]["C"]; N = len(C)
    sHist = M.combine(F, w)
    trades, i = [], max(M.WARM + 30, a["mid"] if so_segunda else 0)
    while i < N - h:
        an = M.analyze_at(a["cls"], a["I"], F, i, w, horizons=[h], max_hist=MAX_HIST, sHist=sHist)
        if an and an["isOpp"]:
            ret = C[i + h] / C[i] - 1
            d = an["dir"]
            hi, lo = a["I"]["H"][i:i + h + 1].max(), a["I"]["L"][i:i + h + 1].min()
            trades.append({"sym": a["sym"], "cls": a["cls"], "i": i, "h": h, "date": datetime.fromtimestamp(int(a["df"]["t"].iloc[i]), timezone.utc).strftime("%Y-%m-%d %H:%M"),
                           "exit_day": int(a["df"]["t"].iloc[i + h]), "dir": d, "p": an["p"], "base": an["best"]["base"], "cost": an["cost"],
                           "retNet": d * ret - an["cost"], "hit": d * ret - an["cost"] > 0, "long_net": ret - an["cost"],
                           "excesso": d * (ret - a["mu"][h]), "targetHit": (hi >= an["target"]) if d > 0 else (lo <= an["target"]),
                           "stopHit": (lo <= an["stop"]) if d > 0 else (hi >= an["stop"])})
            i += h  # uma posição por ativo: só volta a analisar quando esta fecha
        else:
            i += 1
    return trades


def selo(st):
    if not st or st["n"] < 30:
        return "cinza"
    if st["t_excesso"] >= 2 and st["z_acerto"] >= 2 and st["ret_medio"] > 0:
        return "verde"
    if st["t_excesso"] >= 1 and st["ret_medio"] > 0:
        return "amarelo"
    return "vermelho"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sintetico", action="store_true"); args = ap.parse_args()
    ini = time.time()
    ativos = sinteticos() if args.sintetico else carregar()
    ativos = [a for a in ativos if len(a["df"]) >= 3000]
    if not ativos:
        B.log("sem histórico suficiente"); sys.exit(1)
    B.log("calculando fatores..."); preparar(ativos)
    res, escolhas = {}, {}
    for h in HORAS:
        icf = ic_metades(ativos, h)
        esc = {}
        for cls, d in icf.items():
            esc[cls] = {k: (1 if a_ > 0 else -1) for k, ((a_, sa), _) in d.items() if a_ is not None and np.isfinite(a_) and abs(a_) >= 2 * sa}
        escolhas[h] = esc
        for nome, val in (("todos os fatores", False), ("validada", True)):
            ts = []
            for a in ativos:
                if val:
                    sinais = esc.get(a["cls"], {})
                    if not sinais:
                        continue
                    w = {k: (M.DEFAULT_W[k] if k in sinais else 0.0) for k in M.DEFAULT_W}
                    F = a["F"].copy()
                    for fi, k in enumerate(M.BT_KEYS):
                        F[:, fi] *= sinais.get(k, 1)
                else:
                    w, F = dict(M.DEFAULT_W), a["F"]
                ts += simular_ativo(a, h, F, w, val)
            res[(h, nome)] = ts
            st = B.estat(ts)
            B.log(f"{h:2d}h {nome:17s} " + (f"{st['n']:5d} op | acerto {st['acerto']*100:.1f}% (base {st['taxa_base']*100:.1f}%) | ret {st['ret_medio']*100:+.3f}% | "
                                            f"acima da tendência {st['excesso']*100:+.3f}% t {st['t_excesso']:.2f}" if st else "sem operações"))
    # relatório
    hoje = datetime.now(BRT)
    L = ["---", "tipo: backtest", f"gerado_por: MF Tracker backtest intradiário (motor {M.VERSAO_MOTOR})", f"data: {hoje:%Y-%m-%dT%H:%M}-03:00",
         f"dados: {'SINTÉTICOS (teste do código)' if args.sintetico else 'reais'}", "tags: [backtest, trading]", "---", "",
         f"# Backtest de trading (candles de 1 hora) — {hoje:%d/%m/%Y}", ""]
    if args.sintetico:
        L += ["> **Dados sintéticos.** O resultado certo é nenhum selo verde.", ""]
    L += ["Prazos de 1, 4, 12 e 24 horas, um de cada vez (sem escolher o melhor depois). Custos de ida e volta: cripto 0,3%, memecoins 0,6%. "
          "**Acima da tendência** desconta o quanto o ativo subia em média por hora no período; é o que mede habilidade. "
          "**Selo**: verde = passou fora da amostra (t ≥ 2 e acerto acima da taxa base com folga); amarelo = indício fraco; vermelho = sem vantagem; cinza = poucas operações.", "",
          "## Dados", "", "| Ativo | Horas | De | Até |", "|---|---:|---|---|"]
    for a in ativos:
        t = a["df"]["t"]
        L.append(f"| {a['sym']} | {len(t)} | {datetime.fromtimestamp(t.iloc[0], timezone.utc):%Y-%m-%d} | {datetime.fromtimestamp(t.iloc[-1], timezone.utc):%Y-%m-%d} |")
    L += ["", "## Resultado por mercado e prazo", "",
          "| Prazo | Versão | Mercado | Operações | Acerto | Taxa base | Chance prevista | Retorno médio | Acima da tendência | t | Selo |",
          "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    validacao = {"gerado": hoje.strftime("%Y-%m-%dT%H:%M-03:00"), "dados": "sinteticos" if args.sintetico else "reais", "tf": "1h", "itens": {}}
    for h in HORAS:
        for nome in ("todos os fatores", "validada"):
            ts = res[(h, nome)]
            for cls in ("crypto", "meme", "total"):
                g = ts if cls == "total" else [t for t in ts if t["cls"] == cls]
                st = B.estat(g)
                sl = selo(st)
                if nome == "validada" and cls != "total":
                    validacao["itens"].setdefault(cls, {})[f"{h}h"] = {"selo": sl, "n": st["n"] if st else 0,
                        "acerto": st["acerto"] if st else None, "base": st["taxa_base"] if st else None,
                        "excesso": st["excesso"] if st else None, "t": st["t_excesso"] if st else None,
                        "fatores": escolhas[h].get(cls, {})}
                if not st:
                    L.append(f"| {h}h | {nome} | {'Total' if cls == 'total' else M.CLASSES[cls]['label']} | 0 | — | — | — | — | — | — | cinza |")
                    continue
                L.append(f"| {h}h | {nome} | {'**Total**' if cls == 'total' else M.CLASSES[cls]['label']} | {st['n']} | {B.pc(st['acerto'])} | {B.pc(st['taxa_base'])} | "
                         f"{B.pc(st['chance_prevista'])} | {B.pct(st['ret_medio'], 3)} | {B.pct(st['excesso'], 3)} | {B.num(st['t_excesso'])} | {sl} |")
    nomes = {"tendencia": "tendência", "momento": "momento", "rsi": "RSI", "bollinger": "desvio da média", "volume": "volume", "psicologia": "psicologia", "nicho": "nicho"}
    L += ["", "## Fatores escolhidos na 1ª metade (por prazo)", ""]
    for h in HORAS:
        for cls, sinais in escolhas[h].items():
            L.append(f"- {h}h, {M.CLASSES[cls]['label']}: " + (", ".join(f"{nomes[k]}{' (invertido)' if v < 0 else ''}" for k, v in sinais.items()) or "nenhum aprovado"))
    L += ["", "## Limites (autocrítica)", "",
          "- Só cripto e memecoins: B3 e EUA não têm candles de 1 hora grátis com anos de histórico.",
          "- Custos fixos; em horas de pânico o spread é maior. Sem taxa de financiamento de posições vendidas.",
          "- Muitas operações se sobrepõem no tempo entre ativos que andam juntos: o t real é menor que o mostrado.",
          "- Memecoins listadas na Binance são sobreviventes.",
          f"", f"Tempo de execução: {time.time() - ini:.0f} s.", ""]
    md = "\n".join(L)
    os.makedirs(B.SAIDA, exist_ok=True)
    open(os.path.join(B.SAIDA, "intradiario.md"), "w", encoding="utf-8").write(md)
    json.dump(validacao, open(os.path.join(B.SAIDA, "validacao.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not args.sintetico:
        try:
            vault = json.load(open(os.path.join(B.BASE, "config.json"), encoding="utf-8"))["vault"]
            if os.path.isdir(os.path.join(vault, "01 Projeto")):
                nome = f"Backtest trading 1h {hoje:%Y-%m-%d %Hh%M}.md"
                open(os.path.join(vault, "01 Projeto", nome), "w", encoding="utf-8").write(md)
                os.makedirs(os.path.join(vault, "99 Meta"), exist_ok=True)
                json.dump(validacao, open(os.path.join(vault, "99 Meta", "validacao.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                B.log(f"gravado no vault: 01 Projeto/{nome} e 99 Meta/validacao.json")
        except Exception as e:
            B.log(f"não gravei no vault: {e}")
    B.log(f"pronto em {time.time() - ini:.0f} s")


if __name__ == "__main__":
    main()
