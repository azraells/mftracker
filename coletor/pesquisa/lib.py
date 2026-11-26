"""Biblioteca comum da pesquisa autônoma (só leitura dos dados do coletor).

Os dados vêm do cache do coletor (coletor/dados/historico). Para outra pasta, defina MF_HIST.
"""
import json, math, os
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
HIST = os.environ.get("MF_HIST", os.path.join(AQUI, "..", "dados", "historico"))
OUT = os.path.join(AQUI, "saida")
os.makedirs(OUT, exist_ok=True)

CLASSES = {
    "crypto": ["BTC", "ETH", "SOL", "XRP"],
    "meme": ["DOGE", "SHIB", "PEPE", "BONK", "WIF"],
    "b3": ["PETR4", "VALE3", "ITUB4", "MGLU3", "BBAS3", "WEGE3", "BOVA11"],
    "us": ["AAPL", "QQQ", "MSFT", "NVDA", "SPY", "TSLA", "AMZN"],
    "fx": ["USD-BRL"],
}
NOME = {"crypto": "Cripto", "meme": "Memecoins", "b3": "B3", "us": "EUA", "fx": "Dólar"}
# custo de ida e volta (fração): o mesmo do programa, e uma versão barata (ordem limitada / corretora grande)
CUSTO = {"crypto": 0.003, "meme": 0.006, "b3": 0.001, "us": 0.001, "fx": 0.005}
CUSTO_BARATO = {"crypto": 0.0015, "meme": 0.003, "b3": 0.0005, "us": 0.0005, "fx": 0.002}
CLS_DE = {a: c for c, l in CLASSES.items() for a in l}

N_TESTES = 0  # contador global de simulações (combinação × ativo × período)


def conta(n=1):
    global N_TESTES
    N_TESTES += n


def diario(a):
    d = pd.read_csv(os.path.join(HIST, f"{a}.csv"))
    d["dt"] = pd.to_datetime(d["t"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    d = d.drop_duplicates("dt", keep="last").set_index("dt").sort_index()
    return d[["o", "h", "l", "c", "v"]].astype(float)


def horario(a):
    p = os.path.join(HIST, f"{a}-1h.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d["dt"] = pd.to_datetime(d["t"], unit="s", utc=True).dt.tz_convert(None)
    d = d.drop_duplicates("dt", keep="last").set_index("dt").sort_index()
    return d[["o", "h", "l", "c", "v"]].astype(float)


def funding(a):
    p = os.path.join(HIST, f"{a}-funding.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d["dt"] = pd.to_datetime(d["t"], unit="ms" if d["t"].iloc[0] > 1e11 else "s", utc=True).dt.tz_convert(None)
    return d.set_index("dt")["f"].sort_index()


def selic_diaria():
    """Taxa por dia útil (fração), indexada por data."""
    d = json.load(open(os.path.join(HIST, "selic.json"), encoding="utf-8"))
    s = pd.Series({pd.to_datetime(x["data"], dayfirst=True): float(x["valor"]) / 100 for x in d}).sort_index()
    return s[~s.index.duplicated()]


def juro_eua_diario():
    """^IRX em % ao ano -> fração por dia corrido."""
    d = pd.read_csv(os.path.join(HIST, "irx.csv"))
    d["dt"] = pd.to_datetime(d["t"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    s = d.drop_duplicates("dt").set_index("dt")["c"].astype(float).sort_index()
    return (1 + s / 100) ** (1 / 365) - 1


def caixa_para(idx, cls):
    """Rendimento do caixa por período do índice idx (diário): Selic para ativos em real, juro americano para o resto."""
    if cls in ("b3", "fx"):
        s = selic_diaria().reindex(idx, method="ffill").fillna(0.0)
        return s.values
    s = juro_eua_diario().reindex(idx, method="ffill").fillna(0.0)
    gap = np.r_[1, np.diff(idx.values).astype("timedelta64[D]").astype(int)]
    return ((1 + s.values) ** gap - 1)


# ---------------------------------------------------------------- estatística

def t_stat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def t_por_dia(pnl, quando):
    """t agrupando operações que começam no mesmo dia (MÉDIA por dia: cada dia pesa igual).
    ATENÇÃO (26/11/2026): superestima o t quando dias com poucas operações são os melhores. Para o ganho total,
    use t_por_dia_soma (cada operação pesa igual). No modelo de 4 h, margem 0,05: t 8,1 pela média, 3,0 pela soma."""
    s = pd.Series(np.asarray(pnl, float), index=pd.to_datetime(quando)).groupby(lambda x: x.normalize()).mean()
    return t_stat(s.values), len(s)


def t_por_dia_soma(pnl, quando):
    """t agrupando por dia pela SOMA do dia (cada operação pesa igual; robusto à correlação dentro do dia)."""
    s = pd.Series(np.asarray(pnl, float), index=pd.to_datetime(quando)).groupby(lambda x: x.normalize()).sum()
    return t_stat(s.values), len(s)


def p_bilateral(t):
    return math.erfc(abs(t) / math.sqrt(2))


def holm(pvals, alfa=0.05):
    """Devolve lista de booleanos (sobrevive à correção de Holm)."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    ok = [False] * m
    for k, i in enumerate(idx):
        if pvals[i] <= alfa / (m - k):
            ok[i] = True
        else:
            break
    return ok


def metricas_serie(r, per_ano, rf=None):
    """r: retornos por período (fração). Devolve CAGR, vol, Sharpe (acima do juro do caixa rf, se dado), maior queda."""
    r = np.asarray(r, float)
    rf = np.zeros_like(r) if rf is None else np.asarray(rf, float)
    ok = np.isfinite(r)
    r, rf = r[ok], rf[ok]
    if len(r) < 10:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, mdd=np.nan)
    eq = np.cumprod(1 + r)
    anos = len(r) / per_ano
    cagr = eq[-1] ** (1 / anos) - 1 if eq[-1] > 0 else -1.0
    vol = r.std(ddof=1) * math.sqrt(per_ano)
    ex = r - rf
    sh = ex.mean() / ex.std(ddof=1) * math.sqrt(per_ano) if ex.std(ddof=1) > 0 else 0.0
    pico = np.maximum.accumulate(eq)
    mdd = float((eq / pico - 1).min())
    return dict(cagr=float(cagr), vol=float(vol), sharpe=float(sh), mdd=mdd)


def t_dif_sharpe_mensal(r1, r2, idx, rf=None):
    """t da diferença de retorno ajustado ao risco em blocos mensais, sobre o juro do caixa."""
    rf = np.zeros(len(idx)) if rf is None else np.asarray(rf, float)
    r1 = np.asarray(r1, float) - rf; r2 = np.asarray(r2, float) - rf
    a = pd.Series(r1, index=idx).groupby(idx.to_period("M")).apply(lambda x: (1 + x).prod() - 1)
    b = pd.Series(r2, index=idx).groupby(idx.to_period("M")).apply(lambda x: (1 + x).prod() - 1)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    d = a / a.std() - b / b.std()
    return t_stat(d.values)


def pct(x, casas=1):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x*100:+.{casas}f}%".replace(".", ",")


def num(x, casas=2):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{casas}f}".replace(".", ",")


def salva(nome, obj):
    json.dump(obj, open(os.path.join(OUT, nome), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
