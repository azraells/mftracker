"""
MF Tracker — a taxa de financiamento dos futuros prevê a direção do preço?

Nos contratos perpétuos, quem está comprado paga quem está vendido a cada 8 horas quando a taxa é positiva.
Taxa muito alta = muita gente apostando na alta com alavancagem; a hipótese clássica é que isso antecede quedas
(e taxa muito negativa, altas). É um dado de fora do preço, que os testes anteriores não tinham.

Teste (mesma régua dos outros):
  * a cada pagamento de financiamento (8 h), z = (taxa - média dos últimos 30 dias) / desvio dos últimos 30 dias;
  * retorno do preço à vista nas h horas seguintes (h em 8, 24, 72), a partir dos candles de 1 h já baixados;
  * na 1ª metade escolhe o sentido (seguir ou contrariar) e o limiar k em {1, 1,5, 2, 2,5};
  * só a 2ª metade conta; t agrupado por horário (ativos que andam juntos contam como uma observação).

Contratos em aberto e liquidações: a Binance só dá 30 dias de histórico grátis, então não dá para testar agora.
O coletor teria que gravar esses dados daqui para a frente.

Uso: python funding.py               (baixa o histórico de financiamento da Binance Futures via ccxt)
     python funding.py --sintetico   (taxas aleatórias: o resultado certo é t abaixo de 2)
"""
import argparse, json, math, os, sys, time
from datetime import datetime, timezone, timedelta
from itertools import product

import numpy as np
import pandas as pd

import backtest as B
import backtest_intradiario as BI

BRT = timezone(timedelta(hours=-3))
PERP = {"BTC": "BTC/USDT:USDT", "ETH": "ETH/USDT:USDT", "SOL": "SOL/USDT:USDT", "XRP": "XRP/USDT:USDT", "DOGE": "DOGE/USDT:USDT",
        "SHIB": "1000SHIB/USDT:USDT", "PEPE": "1000PEPE/USDT:USDT", "BONK": "1000BONK/USDT:USDT", "WIF": "WIF/USDT:USDT"}
HS = [8, 24, 72]
KS = [1.0, 1.5, 2.0, 2.5]
JAN = 90  # 30 dias de pagamentos de 8 h
CUSTOS = [0.0, 0.0008, 0.0015, 0.003]
MIN_OP = 60


def baixar_funding(sym):
    import ccxt
    ex = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 30000})
    since = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    while since < time.time() * 1000:
        lote = ex.fetch_funding_rate_history(PERP[sym], since=since, limit=1000)
        if not lote:
            break
        rows += [(int(x["timestamp"]) // 1000, float(x["fundingRate"])) for x in lote]
        nxt = lote[-1]["timestamp"] + 1
        if nxt <= since:
            break
        since = nxt
    return pd.DataFrame(rows, columns=["t", "f"]).drop_duplicates("t").sort_values("t").reset_index(drop=True)


def carregar_funding(ativos, sintetico):
    rng = np.random.default_rng(11)
    for a in ativos:
        if sintetico:
            t = a["df"]["t"].to_numpy(); t = t[t % 28800 == 0]
            a["fund"] = pd.DataFrame({"t": t, "f": 0.0001 + rng.normal(0, 0.0002, len(t))})
            continue
        path = os.path.join(B.HIST, f"{a['sym']}-funding.csv")
        try:
            if B.cache_ok(path, 12):
                df = pd.read_csv(path)
            else:
                df = baixar_funding(a["sym"]); df.to_csv(path, index=False)
            a["fund"] = df
            B.log(f"{a['sym']:6s} {len(df):5d} pagamentos de financiamento  {datetime.fromtimestamp(df['t'].iloc[0], timezone.utc):%Y-%m-%d} a "
                  f"{datetime.fromtimestamp(df['t'].iloc[-1], timezone.utc):%Y-%m-%d}  média {df['f'].mean()*100:.4f}% por 8 h")
        except Exception as e:
            a["fund"] = None
            B.log(f"{a['sym']:6s} FALHOU financiamento: {e}")


def eventos(a):
    """Tabela por pagamento: t, z da taxa, retornos futuros, metade."""
    f = a["fund"].copy()
    f["t"] = (f["t"] // 3600) * 3600
    f = f.drop_duplicates("t")
    m = f["f"].rolling(JAN, min_periods=JAN // 2).mean().shift(1)
    s = f["f"].rolling(JAN, min_periods=JAN // 2).std().shift(1)
    f["z"] = (f["f"] - m) / s
    px = a["df"].set_index("t")["c"]
    pos = px.index.get_indexer(f["t"])
    C = px.to_numpy()
    for h in HS:
        r = np.full(len(f), np.nan)
        ok = (pos >= 0) & (pos + h < len(C))
        r[ok] = C[pos[ok] + h] / C[pos[ok]] - 1
        f[f"r{h}"] = r
    f = f[np.isfinite(f["z"]) & (pos >= 0)].reset_index(drop=True)
    meio = a["df"]["t"].iloc[len(a["df"]) // 2]
    f["metade"] = (f["t"] >= meio).astype(int)
    f["sym"], f["cls"] = a["sym"], a["cls"]
    for h in HS:
        mu = {m_: float(f.loc[f["metade"] == m_, f"r{h}"].mean()) for m_ in (0, 1)}
        f[f"mu{h}"] = f["metade"].map(mu)
    return f


def regra(ev, h, k, sentido):
    """Operações: z >= k → direção -sentido·(+1)... sentido=+1 segue a multidão, -1 contraria. Sem sobreposição por ativo."""
    out = []
    for sym, g in ev.groupby("sym"):
        livre = -1
        for row in g.itertuples(index=False):
            if row.t < livre or not np.isfinite(getattr(row, f"r{h}")) or abs(row.z) < k:
                continue
            d = sentido * (1 if row.z > 0 else -1)
            r = getattr(row, f"r{h}")
            out.append((row.t, row.metade, row.cls, d * r, d * (r - getattr(row, f"mu{h}"))))
            livre = row.t + h * 3600
    return pd.DataFrame(out, columns=["t", "metade", "cls", "bruto", "exc"])


def resumo(df, custo=0.0):
    if len(df) < 2:
        return None
    g = df.groupby("t")[["bruto", "exc"]].mean()
    def tt(x):
        sd = x.std(ddof=1)
        return float(x.mean() / (sd / math.sqrt(len(x)))) if sd > 0 and len(x) > 1 else 0.0
    return {"n": len(df), "bruto": float(df["bruto"].mean()), "exc": float(df["exc"].mean()), "t": tt(g["exc"]),
            "liq": float(df["bruto"].mean() - custo), "acerto": float((df["bruto"] - custo > 0).mean()), "t_liq": tt(g["bruto"] - custo)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sintetico", action="store_true"); args = ap.parse_args()
    ini = time.time()
    ativos = BI.sinteticos() if args.sintetico else BI.carregar()
    ativos = [a for a in ativos if len(a["df"]) >= 3000]
    carregar_funding(ativos, args.sintetico)
    ativos = [a for a in ativos if a.get("fund") is not None and len(a["fund"]) > 2 * JAN]
    if not ativos:
        B.log("sem histórico de financiamento"); sys.exit(1)
    ev = pd.concat([eventos(a) for a in ativos], ignore_index=True)
    # correlação simples z x retorno futuro por metade (informativo)
    ic = {}
    for cls in ("crypto", "meme"):
        for h in HS:
            for m_ in (0, 1):
                e = ev[(ev["cls"] == cls) & (ev["metade"] == m_)].dropna(subset=[f"r{h}"])
                if len(e) > 30:
                    ic[(cls, h, m_)] = float(e["z"].rank().corr((e[f"r{h}"] - e[f"mu{h}"]).rank()))
    escolha = {}
    for cls in ("crypto", "meme"):
        e = ev[ev["cls"] == cls]
        if e.empty:
            continue
        melhor = None
        for h, k, sen in product(HS, KS, (1, -1)):
            ops = regra(e, h, k, sen)
            s1 = resumo(ops[ops["metade"] == 0])
            if not s1 or s1["n"] < MIN_OP:
                continue
            if melhor is None or s1["t"] > melhor[1]["t"]:
                melhor = ((h, k, sen), s1, ops[ops["metade"] == 1])
        escolha[cls] = melhor
        if melhor:
            (h, k, sen), s1, o2 = melhor
            s2 = resumo(o2)
            B.log(f"{cls:6s} escolhida: {'segue' if sen > 0 else 'contraria'} a multidão, |z|>={k}, sai em {h} h | 1ª: {s1['n']} op exc {s1['exc']*100:+.3f}% t {s1['t']:.2f} | "
                  f"2ª: {s2['n'] if s2 else 0} op exc {(s2['exc'] if s2 else 0)*100:+.3f}% t {(s2['t'] if s2 else 0):.2f}")
    hoje = datetime.now(BRT)
    L = ["---", "tipo: backtest", "gerado_por: MF Tracker funding.py", f"data: {hoje:%Y-%m-%dT%H:%M}-03:00",
         f"dados: {'SINTÉTICOS (teste do código)' if args.sintetico else 'reais'}", "tags: [backtest, trading, futuros]", "---", "",
         f"# Taxa de financiamento dos futuros e direção do preço — {hoje:%d/%m/%Y}", ""]
    if args.sintetico:
        L += ["> **Dados sintéticos** (taxas aleatórias). O resultado certo é t abaixo de 2.", ""]
    L += ["Pergunta: quando a taxa de financiamento dos contratos perpétuos foge muito do normal dos últimos 30 dias (muita alavancagem de um lado), "
          "o preço à vista anda em seguida? Na 1ª metade do histórico escolhe-se o sentido (seguir ou contrariar a multidão), o limiar e o prazo; **só a 2ª metade conta**.", "",
          "## Correlação entre a taxa (em desvios) e o retorno seguinte acima da tendência", "",
          "| Mercado | Prazo | 1ª metade | 2ª metade |", "|---|---:|---:|---:|"]
    for cls, lab in (("crypto", "Cripto"), ("meme", "Memecoins")):
        for h in HS:
            if (cls, h, 0) in ic:
                L.append(f"| {lab} | {h} h | {ic[(cls, h, 0)]:+.3f} | {ic.get((cls, h, 1), float('nan')):+.3f} |")
    L += ["", "Negativo = taxa alta antecede queda (a hipótese de contrariar a multidão).", "", "## Regra escolhida e resultado fora da amostra", ""]
    for cls, lab in (("crypto", "Cripto"), ("meme", "Memecoins")):
        if not escolha.get(cls):
            L += [f"**{lab}:** sem operações suficientes.", ""]; continue
        (h, k, sen), s1, o2 = escolha[cls]
        s2 = resumo(o2)
        L += [f"**{lab}** — {'seguir' if sen > 0 else 'contrariar'} a multidão quando a taxa passa de {k} desvios, saindo em {h} h "
              f"(1ª metade: {s1['n']} operações, acima da tendência {B.pct(s1['exc'], 3)}, t {s1['t']:.2f}).", ""]
        if not s2:
            L += ["Sem operações na 2ª metade.", ""]; continue
        ok = s2["t"] >= 2 and s2["exc"] > 0
        L += [f"2ª metade: {s2['n']} operações · bruto {B.pct(s2['bruto'], 3)} · acima da tendência {B.pct(s2['exc'], 3)} · t {s2['t']:.2f}.", "",
              "**Veredito:** " + ("confirmada fora da amostra; vale conferir com custo real abaixo." if ok else "não confirmada fora da amostra (t abaixo de 2)."), "",
              "| Custo de ida e volta | Líquido por operação | Acerto | t |", "|---:|---:|---:|---:|"]
        for c in CUSTOS:
            r = resumo(o2, c)
            L.append(f"| {B.pct(c, 2)} | {B.pct(r['liq'], 3)} | {B.pc(r['acerto'])} | {r['t_liq']:.2f} |")
        L.append("")
    L += ["## Limites (autocrítica)", "",
          "- Só a taxa de financiamento: contratos em aberto e liquidações têm só 30 dias de histórico grátis na Binance. Para testá-los, o coletor precisa gravar daqui para a frente.",
          "- 16 combinações por mercado (prazo × limiar × sentido): o t da 1ª metade é inflado; só o da 2ª vale.",
          "- Retorno medido no preço à vista; operar no perpétuo também paga ou recebe a própria taxa, que não entrou.",
          "- Memecoins listadas na Binance são sobreviventes.", "", f"Tempo de execução: {time.time() - ini:.0f} s.", ""]
    md = "\n".join(L)
    os.makedirs(B.SAIDA, exist_ok=True)
    open(os.path.join(B.SAIDA, "funding.md"), "w", encoding="utf-8").write(md)
    if not args.sintetico:
        try:
            vault = json.load(open(os.path.join(B.BASE, "config.json"), encoding="utf-8"))["vault"]
            if os.path.isdir(os.path.join(vault, "01 Projeto")):
                nome = f"Financiamento dos futuros {hoje:%Y-%m-%d %Hh%M}.md"
                open(os.path.join(vault, "01 Projeto", nome), "w", encoding="utf-8").write(md)
                B.log(f"gravado no vault: 01 Projeto/{nome}")
        except Exception as e:
            B.log(f"não gravei no vault: {e}")
    B.log(f"pronto em {time.time() - ini:.0f} s")


if __name__ == "__main__":
    main()
