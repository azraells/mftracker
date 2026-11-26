"""
MF Tracker — coletor local de dados ao vivo.

Roda no PC do Luis. A cada N segundos busca preços reais e grava:
  1. dados/snapshots/AAAA-MM-DD.csv  (arquivo bruto, todas as leituras, para backtests futuros)
  2. dados/ultimo.json               (última leitura de cada ativo)
  3. <vault>/02 Mercados/Cotações ao vivo.md   (regravada a cada ciclo)
  4. <vault>/03 Ativos/<ativo>.md, seção "## Snapshots" (uma linha por hora, configurável)

Fontes: ccxt (Binance / Coinbase), DexScreener (memecoins de DEX), yfinance (B3, EUA, índices),
AwesomeAPI (dólar), alternative.me (Medo & Ganância), CoinGecko (mercado cripto), Banco Central (Selic).
Nenhuma fonte exige chave. Se uma falhar, as outras continuam e a falha vai para o log e para a nota.

Uso:  python coletor.py           (fica rodando)
      python coletor.py --once    (uma leitura e sai)
"""
import csv, json, os, sys, time, traceback, urllib.request, re
from datetime import datetime, timezone, timedelta

VERSAO = "0.1.0"
BASE = os.path.dirname(os.path.abspath(__file__))
BRT = timezone(timedelta(hours=-3))
UA = {"User-Agent": "mf-tracker-coletor/0.1", "Accept": "application/json"}

def log(msg):
    linha = f"{datetime.now(BRT):%Y-%m-%d %H:%M:%S} {msg}"
    print(linha, flush=True)
    os.makedirs(os.path.join(BASE, "dados"), exist_ok=True)
    with open(os.path.join(BASE, "dados", "coletor.log"), "a", encoding="utf-8") as f:
        f.write(linha + "\n")

def get_json(url, timeout=20):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None

# ---------------------------------------------------------------- fontes
def coletar_cripto(cfg, falhas):
    """ccxt: tenta cada exchange na ordem; usa a primeira que responder."""
    out = []
    try:
        import ccxt
    except ImportError:
        falhas.append("ccxt não instalado")
        return out
    grupos = [("cripto", cfg["cripto"]["ativos"]), ("memecoins", cfg.get("memecoins", {}))]
    for ex_id in cfg["cripto"]["exchanges"]:
        try:
            ex = getattr(ccxt, ex_id)({"enableRateLimit": True, "timeout": 20000})
            ex.load_markets()
            quote = "USDT" if ex_id == "binance" else "USD"
            pedidos = {}
            for classe, ativos in grupos:
                for ativo, base in ativos.items():
                    sym = f"{base}/{quote}"
                    if sym in ex.markets:
                        pedidos[sym] = (ativo, classe)
            if not pedidos:
                continue
            tks = ex.fetch_tickers(list(pedidos.keys()))
            for sym, (ativo, classe) in pedidos.items():
                t = tks.get(sym)
                if not t or t.get("last") is None:
                    continue
                ts = t.get("timestamp") or int(time.time() * 1000)
                out.append(dict(ativo=ativo, classe=classe, preco=num(t["last"]), moeda="USD" if quote in ("USD", "USDT") else quote,
                                var_24h_pct=num(t.get("percentage")), volume_24h=num(t.get("quoteVolume")), liquidez_usd=None, mcap_usd=None,
                                fonte=f"{ex.name} via ccxt ({sym})", ts_fonte=ts, atraso="tempo real"))
            faltando = [a for g in grupos for a in g[1] if a not in {o["ativo"] for o in out}]
            if not faltando:
                return out
            log(f"{ex_id}: sem dados para {faltando}, tentando próxima exchange")
        except Exception as e:
            falhas.append(f"{ex_id}: {type(e).__name__}: {str(e)[:120]}")
    return out

def coletar_dex(cfg, falhas):
    out = []
    por_rede = {}
    for d in cfg.get("dex", []):
        por_rede.setdefault(d["rede"], []).append(d)
    for rede, itens in por_rede.items():
        try:
            pares = get_json(f"https://api.dexscreener.com/tokens/v1/{rede}/" + ",".join(i["contrato"] for i in itens))
        except Exception as e:
            falhas.append(f"DexScreener {rede}: {e}")
            continue
        for d in itens:
            cand = [p for p in pares if p.get("baseToken", {}).get("address") == d["contrato"]]
            p = next((x for x in cand if x.get("pairAddress") == d.get("pool")), None) or \
                max(cand, key=lambda x: (x.get("liquidity") or {}).get("usd") or 0, default=None)
            if not p:
                falhas.append(f"DexScreener: {d['ativo']} sem par")
                continue
            pc, tx = p.get("priceChange") or {}, (p.get("txns") or {}).get("h24") or {}
            out.append(dict(ativo=d["ativo"], classe="memecoins", preco=num(p.get("priceUsd")), moeda="USD",
                            var_24h_pct=num(pc.get("h24")), volume_24h=num((p.get("volume") or {}).get("h24")),
                            liquidez_usd=num((p.get("liquidity") or {}).get("usd")), mcap_usd=num(p.get("marketCap") or p.get("fdv")),
                            fonte=f"DexScreener ({p.get('dexId')} {p.get('pairAddress','')[:6]}…)", ts_fonte=int(time.time() * 1000), atraso="tempo real",
                            extra=dict(var_1h=num(pc.get("h1")), var_6h=num(pc.get("h6")), compras=tx.get("buys"), vendas=tx.get("sells"))))
    return out

def coletar_yahoo(cfg, falhas):
    out = []
    try:
        import yfinance as yf
    except ImportError:
        falhas.append("yfinance não instalado")
        return out
    for ativo, info in cfg.get("yahoo", {}).items():
        try:
            h = yf.Ticker(info["ticker"]).history(period="5d", interval="1d", auto_adjust=False)
            if h is None or h.empty:
                falhas.append(f"Yahoo {info['ticker']}: sem dados")
                continue
            ultimo = float(h["Close"].iloc[-1]); ant = float(h["Close"].iloc[-2]) if len(h) > 1 else None
            ts = int(h.index[-1].timestamp() * 1000)
            try:  # preço intradiário mais recente, quando o pregão está aberto
                m = yf.Ticker(info["ticker"]).history(period="1d", interval="1m")
                if m is not None and not m.empty:
                    ultimo = float(m["Close"].iloc[-1]); ts = int(m.index[-1].timestamp() * 1000)
            except Exception:
                pass
            out.append(dict(ativo=ativo, classe=info["classe"], preco=ultimo, moeda=info["moeda"],
                            var_24h_pct=(ultimo / ant - 1) * 100 if ant else None, volume_24h=num(h["Volume"].iloc[-1]),
                            liquidez_usd=None, mcap_usd=None, fonte=f"Yahoo Finance via yfinance ({info['ticker']})", ts_fonte=ts,
                            atraso="Yahoo: B3 com ~15 min de atraso" if info["classe"] == "b3" else "Yahoo: EUA quase em tempo real"))
        except Exception as e:
            falhas.append(f"Yahoo {info['ticker']}: {type(e).__name__}: {str(e)[:100]}")
    return out

def coletar_cambio(falhas):
    try:
        r = get_json("https://economia.awesomeapi.com.br/json/last/USD-BRL")["USDBRL"]
        return [dict(ativo="USD-BRL", classe="dolar", preco=num(r["bid"]), moeda="BRL", var_24h_pct=num(r["pctChange"]), volume_24h=None,
                     liquidez_usd=None, mcap_usd=None, fonte="AwesomeAPI", ts_fonte=int(r["timestamp"]) * 1000, atraso="tempo real (comercial)")]
    except Exception as e:
        falhas.append(f"AwesomeAPI: {e}")
        return []

def coletar_contexto(falhas):
    ctx = {}
    try:
        d = get_json("https://api.alternative.me/fng/?limit=1")["data"][0]
        ctx["medo_ganancia"] = f"{d['value']} ({d['value_classification']})"
    except Exception as e:
        falhas.append(f"Medo & Ganância: {e}")
    try:
        g = get_json("https://api.coingecko.com/api/v3/global")["data"]
        ctx["mercado_cripto_24h"] = f"{g['market_cap_change_percentage_24h_usd']:+.2f}%"
        ctx["dominancia_btc"] = f"{g['market_cap_percentage']['btc']:.1f}%"
    except Exception as e:
        falhas.append(f"CoinGecko: {e}")
    try:
        s = get_json("https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json")[0]
        # a série 432 vem preenchida até a próxima reunião do Copom, então a data é o fim da série, não o início da vigência
        ctx["selic"] = f"{s['valor']}% ao ano (BCB, série 432)"
    except Exception as e:
        falhas.append(f"BCB Selic: {e}")
    return ctx

# ---------------------------------------------------------------- formatação
def br_num(v, casas=None):
    if v is None:
        return "—"
    if casas is None:
        casas = 2 if abs(v) >= 1 else 4 if abs(v) >= 0.01 else 8
    s = f"{v:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def usd_curto(v):
    if not v:
        return "—"
    for lim, suf in ((1e9, " bi"), (1e6, " mi"), (1e3, " mil")):
        if v >= lim:
            return f"US$ {br_num(v / lim, 1)}{suf}"
    return f"US$ {br_num(v, 0)}"

def pct(v):
    return "—" if v is None else (("+" if v >= 0 else "") + br_num(v, 2) + "%")

def preco_fmt(r):
    sim = {"USD": "US$ ", "BRL": "R$ "}.get(r["moeda"], "")
    return sim + br_num(r["preco"])

# ---------------------------------------------------------------- gravação
CAMPOS = ["ts_utc", "data_hora_brt", "ativo", "classe", "preco", "moeda", "var_24h_pct", "volume_24h", "liquidez_usd", "mcap_usd", "fonte", "ts_fonte_utc", "atraso"]

def gravar_csv(linhas, agora):
    pasta = os.path.join(BASE, "dados", "snapshots")
    os.makedirs(pasta, exist_ok=True)
    arq = os.path.join(pasta, f"{agora.astimezone(BRT):%Y-%m-%d}.csv")
    novo = not os.path.exists(arq)
    with open(arq, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS, extrasaction="ignore")
        if novo:
            w.writeheader()
        for r in linhas:
            w.writerow({**r, "ts_utc": agora.isoformat(timespec="seconds"), "data_hora_brt": f"{agora.astimezone(BRT):%Y-%m-%d %H:%M:%S}",
                        "ts_fonte_utc": datetime.fromtimestamp(r["ts_fonte"] / 1000, timezone.utc).isoformat(timespec="seconds") if r.get("ts_fonte") else ""})
    with open(os.path.join(BASE, "dados", "ultimo.json"), "w", encoding="utf-8") as f:
        json.dump({"gerado": agora.isoformat(), "linhas": linhas}, f, ensure_ascii=False, indent=1)

def escrever_seguro(caminho, texto):
    tmp = caminho + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(texto)
    os.replace(tmp, caminho)

def nota_cotacoes(cfg, linhas, ctx, falhas, agora):
    ordem = ["cripto", "memecoins", "b3", "bolsa-eua", "dolar"]
    nomes = {"cripto": "Cripto", "memecoins": "Memecoins", "b3": "B3", "bolsa-eua": "Bolsa americana", "dolar": "Dólar"}
    notas = cfg.get("notas", {})
    partes = [f"""---
tipo: painel
gerado_por: MF Tracker coletor {VERSAO}
atualizado: {agora.astimezone(BRT):%Y-%m-%dT%H:%M}-03:00
tags: [painel, cotacoes]
---
# Cotações ao vivo
_Regravada pelo coletor do PC a cada {cfg['intervalo_segundos'] // 60} min enquanto ele está rodando. Leitura de {agora.astimezone(BRT):%d/%m/%Y %H:%M} (Brasília). Não editar à mão. Histórico completo em `MF Tracker Programa/coletor/dados/snapshots/`. Leituras do modelo em [[Sinais do modelo]]._
"""]
    for cl in ordem:
        rs = [r for r in linhas if r["classe"] == cl]
        if not rs:
            continue
        partes.append(f"\n## {nomes[cl]}\n| Ativo | Preço | Var. 24h | Volume 24h | Liquidez | Valor de mercado | Hora da fonte | Fonte |\n|---|---|---|---|---|---|---|---|")
        for r in rs:
            nome = f"[[{notas[r['ativo']]}]]" if r["ativo"] in notas else r["ativo"]
            hora = datetime.fromtimestamp(r["ts_fonte"] / 1000, BRT).strftime("%d/%m %H:%M") if r.get("ts_fonte") else "—"
            vol = usd_curto(r["volume_24h"]) if r["moeda"] == "USD" and r["classe"] in ("cripto", "memecoins") else br_num(r["volume_24h"], 0)
            partes.append(f"| {nome} | {preco_fmt(r)} | {pct(r['var_24h_pct'])} | {vol} | {usd_curto(r['liquidez_usd'])} | {usd_curto(r['mcap_usd'])} | {hora} | {r['fonte']} · {r['atraso']} |")
    rot = {"medo_ganancia": "Medo & Ganância (cripto)", "mercado_cripto_24h": "Mercado cripto total em 24h", "dominancia_btc": "Dominância do BTC", "selic": "Selic meta"}
    partes.append("\n## Contexto\n" + "\n".join(f"- {rot.get(k, k)}: {v}" for k, v in ctx.items()))
    if falhas:
        partes.append("\n## Fontes que falharam nesta leitura\n" + "\n".join(f"- {f}" for f in falhas))
    caminho = os.path.join(cfg["vault"], "02 Mercados", "Cotações ao vivo.md")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    escrever_seguro(caminho, "\n".join(partes) + "\n")

def linha_snapshot(r, colunas, agora):
    dh = f"{agora.astimezone(BRT):%d/%m/%Y %H:%M}"
    if colunas >= 8:  # formato DEX da nota (preço, valor de mercado, liquidez, volume, variações, compras/vendas, fonte)
        ex = r.get("extra") or {}
        var = f"{pct(ex.get('var_1h'))} / {pct(ex.get('var_6h'))} / {pct(r['var_24h_pct'])}"
        cv = f"{ex.get('compras', '—')} / {ex.get('vendas', '—')}"
        return f"| {dh} | {br_num(r['preco'])} | {usd_curto(r['mcap_usd'])} | {usd_curto(r['liquidez_usd'])} | {usd_curto(r['volume_24h'])} | {var} | {cv} | {r['fonte']} (coletor) |"
    return f"| {dh} | {preco_fmt(r)} | {pct(r['var_24h_pct'])} | {br_num(r['volume_24h'], 0)} | {r['fonte']} · {r['atraso']} |"

CAB_PADRAO = "| Data/hora (Brasília) | Preço | Var. 24h | Volume 24h | Fonte |\n|---|---|---|---|---|"

def atualizar_notas(cfg, linhas, agora, estado):
    intervalo = cfg.get("nota_intervalo_minutos", 60) * 60
    notas = cfg.get("notas", {})
    for r in linhas:
        nome = notas.get(r["ativo"])
        if not nome or r["preco"] is None:
            continue
        if time.time() - estado.get(r["ativo"], 0) < intervalo:
            continue
        caminho = os.path.join(cfg["vault"], "03 Ativos", nome + ".md")
        if not os.path.exists(caminho):
            continue
        with open(caminho, encoding="utf-8") as f:
            txt = f.read()
        ls = txt.split("\n")
        idx = next((i for i, l in enumerate(ls) if re.match(r"^##\s+Snapshots\s*$", l.strip())), None)
        if idx is None:  # cria a seção antes de "## Operações simuladas" ou no fim
            alvo = next((i for i, l in enumerate(ls) if l.startswith("## Operações")), len(ls))
            bloco = ["## Snapshots", CAB_PADRAO.split("\n")[0], CAB_PADRAO.split("\n")[1], linha_snapshot(r, 5, agora), ""]
            ls[alvo:alvo] = bloco
        else:
            j = idx + 1
            while j < len(ls) and not ls[j].startswith("|"):
                j += 1
            if j >= len(ls) or not ls[j].startswith("|"):
                ls[idx + 1:idx + 1] = CAB_PADRAO.split("\n") + [linha_snapshot(r, 5, agora)]
            else:
                colunas = ls[j].count("|") - 1
                k = j
                while k < len(ls) and ls[k].startswith("|"):
                    k += 1
                ls.insert(k, linha_snapshot(r, colunas, agora))
        escrever_seguro(caminho, "\n".join(ls))
        estado[r["ativo"]] = time.time()

# ---------------------------------------------------------------- ciclo
def ciclo(cfg, estado):
    agora = datetime.now(timezone.utc)
    falhas = []
    linhas = []
    linhas += coletar_cripto(cfg, falhas)
    linhas += coletar_dex(cfg, falhas)
    linhas += coletar_yahoo(cfg, falhas)
    linhas += coletar_cambio(falhas)
    ctx = coletar_contexto(falhas)
    linhas = [r for r in linhas if r.get("preco") is not None]
    gravar_csv(linhas, agora)
    if os.path.isdir(cfg["vault"]):
        nota_cotacoes(cfg, linhas, ctx, falhas, agora)
        atualizar_notas(cfg, linhas, agora, estado)
    else:
        falhas.append(f"vault não encontrado em {cfg['vault']}")
    log(f"{len(linhas)} ativos gravados; {len(falhas)} falhas" + (": " + " | ".join(falhas) if falhas else ""))
    for r in linhas:
        print(f"   {r['ativo']:<11} {preco_fmt(r):>22}  {pct(r['var_24h_pct']):>9}  {r['fonte']}")
    return linhas, falhas

def main():
    with open(os.path.join(BASE, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    once = "--once" in sys.argv
    estado_arq = os.path.join(BASE, "dados", "estado_notas.json")
    estado = {}
    if os.path.exists(estado_arq):
        try:
            estado = json.load(open(estado_arq, encoding="utf-8"))
        except Exception:
            estado = {}
    log(f"coletor {VERSAO} iniciado (intervalo {cfg['intervalo_segundos']} s, vault {cfg['vault']})")
    while True:
        try:
            ciclo(cfg, estado)
            os.makedirs(os.path.dirname(estado_arq), exist_ok=True)
            json.dump(estado, open(estado_arq, "w", encoding="utf-8"))
        except Exception:
            log("erro no ciclo:\n" + traceback.format_exc())
        if once:
            break
        time.sleep(cfg["intervalo_segundos"])

if __name__ == "__main__":
    main()
