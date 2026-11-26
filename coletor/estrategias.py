"""
MF Tracker — teste de estratégias de prazo mais longo, com evidência acadêmica.

O backtest de 26/11/2026 mostrou que os fatores técnicos de 3 a 14 dias não têm vantagem.
Aqui testo famílias que a literatura documenta há décadas, com parâmetros FIXADOS ANTES do teste
(nada é ajustado aos dados):

  1. Tendência (média de 200 dias): comprado quando o preço está acima da média, fora do mercado quando abaixo.
  2. Momento de série temporal (Moskowitz, Ooi e Pedersen, 2012): média do sinal dos retornos de 3, 6 e 12 meses;
     comprado se positivo, fora se negativo. Rebalanceia a cada 21 pregões (30 dias em cripto).
  3. Momento relativo: a cada mês, compra o terço do grupo que mais subiu em 6 meses (pulando o último mês).
  4. Carry do dólar: ficar "vendido em dólar" (aplicado em real) rende Selic e perde se o dólar subir;
     compara com ficar comprado em dólar.

Cada estratégia é comparada com simplesmente comprar e segurar o mesmo ativo. Na média, seguir tendência
costuma render parecido ou menos que segurar, mas com quedas bem menores: é isso que se mede aqui.

Uso: python estrategias.py   (usa o histórico baixado pelo backtest.py em dados/historico)
     python estrategias.py --sintetico
"""
import argparse, json, math, os, sys, time, urllib.request
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

import motor as M
import backtest as B

BRT = timezone(timedelta(hours=-3))


def custo(cls):
    return M.CLASSES[cls]["cost"]


def ano_dias(cls):
    return 365 if cls in ("crypto", "meme") else 252


def mes(cls):
    return 30 if cls in ("crypto", "meme") else 21


def metricas(r, cls):
    """r: retornos diários da estratégia (já com custos)."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 60:
        return None
    A = ano_dias(cls)
    eq = np.cumprod(1 + r)
    anos = len(r) / A
    cagr = eq[-1] ** (1 / anos) - 1 if eq[-1] > 0 else -1.0
    vol = r.std(ddof=1) * math.sqrt(A)
    sharpe = r.mean() / r.std(ddof=1) * math.sqrt(A) if r.std() > 0 else float("nan")
    mdd = float(np.max(1 - eq / np.maximum.accumulate(eq)))
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "mdd": mdd, "anos": anos, "final": eq[-1]}


def ret_diarios(C):
    return np.r_[0.0, C[1:] / C[:-1] - 1]


def aplicar(pos, C, cls):
    """pos[i] = posição (0..1) decidida no fechamento do dia i, vale para o retorno do dia i+1. Custos a cada mudança."""
    r = ret_diarios(C)
    pos = np.nan_to_num(pos)
    strat = np.r_[0.0, pos[:-1] * r[1:]]
    troca = np.abs(np.diff(np.r_[0.0, pos]))
    strat -= np.r_[0.0, troca[:-1]] * custo(cls) / 2  # custo de ida e volta dividido entre entrada e saída
    return strat


def tendencia_200(C):
    sma = pd.Series(C).rolling(200).mean().to_numpy()
    return np.where(np.isfinite(sma), (C > sma).astype(float), np.nan)


def tsmom(C, cls):
    m = mes(cls)
    pos = np.full(len(C), np.nan)
    for i in range(12 * m, len(C)):
        if (i - 12 * m) % m == 0:
            sinais = [np.sign(C[i] / C[i - k * m] - 1) for k in (3, 6, 12)]
            pos[i] = 1.0 if np.mean(sinais) > 0 else 0.0
        else:
            pos[i] = pos[i - 1]
    return pos


def momento_relativo(grupo, cls):
    """grupo: lista de ativos {sym, df} da mesma classe. Retorna retornos diários da carteira (datas em comum) e da carteira igual."""
    if len(grupo) < 3:
        return None
    m = mes(cls)
    px = pd.concat({a["sym"]: pd.Series(a["df"]["c"].to_numpy(float), index=(a["df"]["t"] // 86400).astype(int).to_numpy()) for a in grupo}, axis=1)
    px = px.sort_index().ffill()
    ret = px.pct_change()
    n = len(px)
    pesos = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    atual = None
    for i in range(7 * m, n):
        if (i - 7 * m) % m == 0:
            mom = px.iloc[i - m] / px.iloc[i - 7 * m] - 1  # 6 meses, pulando o último
            mom = mom.dropna()
            if len(mom) >= 3:
                top = mom.sort_values(ascending=False).index[: max(1, len(mom) // 3)]
                atual = pd.Series(0.0, index=px.columns); atual[top] = 1.0 / len(top)
        if atual is not None:
            pesos.iloc[i] = atual
    w_prev = pesos.shift(1).fillna(0)
    strat = (w_prev * ret.fillna(0)).sum(axis=1)
    giro = pesos.diff().abs().sum(axis=1).shift(1).fillna(0)
    strat = strat - giro * custo(cls) / 2
    disp = px.notna() & ret.notna()
    igual = (ret.where(disp).mean(axis=1)).fillna(0)
    ini = 7 * m + 1
    return strat.iloc[ini:].to_numpy(), igual.iloc[ini:].to_numpy(), px.index[ini]


def selic_diaria():
    """Série 11 do BCB (Selic diária, % ao dia), em blocos de 10 anos (limite da API)."""
    path = os.path.join(B.HIST, "selic.json")
    if B.cache_ok(path, 24):
        d = json.load(open(path, encoding="utf-8"))
    else:
        d = []
        for a, b in (("01/01/2012", "31/12/2021"), ("01/01/2022", datetime.now(BRT).strftime("%d/%m/%Y"))):
            d += B.get_json(f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=json&dataInicial={a}&dataFinal={b}")
        json.dump(d, open(path, "w", encoding="utf-8"))
    return {int(datetime.strptime(x["data"], "%d/%m/%Y").replace(tzinfo=timezone.utc).timestamp()) // 86400: float(x["valor"]) / 100 for x in d}


def fed_diaria():
    """Juro americano (EFFR) do FRED, CSV público sem chave."""
    path = os.path.join(B.HIST, "effr.csv")
    if not B.cache_ok(path, 24):
        with urllib.request.urlopen(urllib.request.Request("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF", headers=B.UA), timeout=30) as r:
            open(path, "wb").write(r.read())
    df = pd.read_csv(path)
    col = df.columns[1]
    df = df[pd.to_numeric(df[col], errors="coerce").notna()]
    return {int(pd.Timestamp(d).tz_localize("UTC").timestamp()) // 86400: float(v) / 100 / 360 for d, v in zip(df.iloc[:, 0], df[col])}


def carry(fx, selic, fed):
    """Ficar aplicado em real (vendido em dólar) contra ficar em dólar, dia a dia, para quem pensa em reais."""
    dias = (fx["df"]["t"] // 86400).astype(int).to_numpy()
    C = fx["df"]["c"].to_numpy(float)
    r_fx = ret_diarios(C)  # alta do dólar em reais
    s = pd.Series(selic).reindex(range(dias[0], dias[-1] + 1)).ffill()
    f = pd.Series(fed).reindex(range(dias[0], dias[-1] + 1)).ffill()
    # juros acumulados entre um pregão e o seguinte
    sel = np.array([np.prod([1 + (s.get(d, np.nan) or 0) for d in range(dias[i - 1], dias[i])]) - 1 if i else 0.0 for i in range(len(dias))])
    fd = np.array([np.prod([1 + (f.get(d, np.nan) or 0) for d in range(dias[i - 1], dias[i])]) - 1 if i else 0.0 for i in range(len(dias))])
    em_real = sel                                  # CDI aproximado pela Selic
    em_dolar = (1 + r_fx) * (1 + fd) - 1           # dólar aplicado a juro americano, medido em reais
    return em_real, em_dolar, dias


def fmt(x, pctv=True):
    if x is None or not np.isfinite(x):
        return "—"
    return (f"{x*100:+.1f}%" if pctv else f"{x:.2f}").replace(".", ",")


def fmt0(x):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:.1f}%".replace(".", ",")


def linha(nome, ms, mb):
    if not ms or not mb:
        return None
    return (f"| {nome} | {fmt(ms['cagr'])} | {fmt(mb['cagr'])} | {fmt(ms['sharpe'], False)} | {fmt(mb['sharpe'], False)} | "
            f"{fmt0(ms['mdd'])} | {fmt0(mb['mdd'])} |")


def dif_t(rs, rb, cls):
    """t da diferença de Sharpe aproximada pelo retorno ajustado ao risco: usa a diferença diária padronizada pela vol de cada uma."""
    rs, rb = np.asarray(rs), np.asarray(rb)
    ok = np.isfinite(rs) & np.isfinite(rb)
    rs, rb = rs[ok], rb[ok]
    if len(rs) < 250 or rs.std() == 0 or rb.std() == 0:
        return float("nan")
    d = rs / rs.std() - rb / rb.std()
    # dados diários com sinais que mudam devagar: corrige a sobreposição usando blocos mensais
    m = mes(cls); nb = len(d) // m
    blocos = d[: nb * m].reshape(nb, m).sum(axis=1)
    return blocos.mean() / blocos.std(ddof=1) * math.sqrt(nb) if nb > 5 and blocos.std() > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    args = ap.parse_args()
    ini = time.time()
    ativos = B.sinteticos() if args.sintetico else B.carregar(2012)
    ativos = [a for a in ativos if len(a["df"]) >= 400]
    L = ["---", "tipo: backtest", f"gerado_por: MF Tracker estratégias (motor {M.VERSAO_MOTOR})", f"data: {datetime.now(BRT):%Y-%m-%dT%H:%M}-03:00",
         f"dados: {'SINTÉTICOS (teste do código)' if args.sintetico else 'reais'}", "tags: [backtest, estrategias]", "---", "",
         f"# Estratégias de prazo longo — {datetime.now(BRT):%d/%m/%Y}", ""]
    if args.sintetico:
        L += ["> **Dados sintéticos.** O resultado certo aqui é nenhuma estratégia batendo o comprar e segurar.", ""]
    L += ["Parâmetros fixados antes do teste (literatura), nada ajustado aos dados. **Sharpe** = retorno por unidade de risco "
          "(acima de 0,5 é bom; comprar e segurar a bolsa costuma ficar entre 0,3 e 0,6). **Maior queda** = pior perda do pico ao fundo. "
          "**t (Sharpe)** compara o retorno ajustado ao risco da estratégia com o de segurar, em blocos mensais: abaixo de 2, a diferença pode ser sorte.", ""]
    resumo = {}
    for nome_e, fn in (("Tendência (média de 200 dias)", lambda a: tendencia_200(a["df"]["c"].to_numpy(float))),
                       ("Momento de série temporal (3/6/12 meses)", lambda a: tsmom(a["df"]["c"].to_numpy(float), a["cls"]))):
        L += [f"## {nome_e}", "", "| Ativo | Retorno ao ano | Segurar | Sharpe | Sharpe segurar | Maior queda | Maior queda segurar | t (Sharpe) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        melhores = []
        for a in ativos:
            C = a["df"]["c"].to_numpy(float)
            pos = fn(a)
            i0 = int(np.argmax(np.isfinite(pos)))
            rs = aplicar(pos, C, a["cls"])[i0 + 1:]
            rb = ret_diarios(C)[i0 + 1:]
            ms, mb = metricas(rs, a["cls"]), metricas(rb, a["cls"])
            t = dif_t(rs, rb, a["cls"])
            l = linha(a["sym"], ms, mb)
            if l:
                L.append(l + f" {fmt(t, False)} |")
                melhores.append((a["cls"], ms["sharpe"] - mb["sharpe"], ms["mdd"] - mb["mdd"], t))
        if melhores:
            dS = np.array([m[1] for m in melhores]); dD = np.array([m[2] for m in melhores]); ts = np.array([m[3] for m in melhores], float)
            L += ["", f"Resumo: Sharpe melhor que segurar em {int((dS > 0).sum())} de {len(dS)} ativos (diferença média {fmt(dS.mean(), False)}); "
                  f"maior queda menor em {int((dD < 0).sum())} de {len(dD)} (diferença média {fmt(dD.mean())}); "
                  f"diferença de Sharpe significativa (t ≥ 2) em {int((ts >= 2).sum())}, e significativamente pior (t ≤ −2) em {int((ts <= -2).sum())}.", ""]
            resumo[nome_e] = {"ativos": len(dS), "sharpe_melhor": int((dS > 0).sum()), "queda_menor": int((dD < 0).sum()),
                              "t_pos": int((ts >= 2).sum()), "t_neg": int((ts <= -2).sum())}
    L += ["## Momento relativo (compra o terço que mais subiu em 6 meses, troca todo mês)", "",
          "| Grupo | Retorno ao ano | Carteira igual | Sharpe | Sharpe igual | Maior queda | Maior queda igual | t (Sharpe) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cls in M.CLASSES:
        g = [a for a in ativos if a["cls"] == cls]
        r = momento_relativo(g, cls)
        if not r:
            continue
        rs, rb, _ = r
        l = linha(M.CLASSES[cls]["label"], metricas(rs, cls), metricas(rb, cls))
        if l:
            L.append(l + f" {fmt(dif_t(rs, rb, cls), False)} |")
    L += ["", "Com 4 a 7 ativos por grupo, o terço de cima tem 1 ou 2 ativos: o resultado depende muito de poucos nomes.", ""]
    if not args.sintetico:
        fx = next((a for a in ativos if a["cls"] == "fx"), None)
        try:
            if not fx:
                raise RuntimeError("sem histórico do dólar")
            em_real, em_dolar, dias = carry(fx, selic_diaria(), fed_diaria())
            mr, md = metricas(em_real[1:], "fx"), metricas(em_dolar[1:], "fx")
            dif = em_real[1:] - em_dolar[1:]
            mdif = metricas(dif, "fx")
            L += ["## Carry do dólar (para quem pensa em reais)", "",
                  "| Posição | Retorno ao ano | Sharpe | Maior queda |", "|---|---:|---:|---:|",
                  f"| Aplicado em real (Selic) | {fmt(mr['cagr'])} | {fmt(mr['sharpe'], False)} | {fmt0(mr['mdd'])} |",
                  f"| Em dólar aplicado a juro americano | {fmt(md['cagr'])} | {fmt(md['sharpe'], False)} | {fmt0(md['mdd'])} |",
                  f"| Diferença (real menos dólar) | {fmt(mdif['cagr'])} | {fmt(mdif['sharpe'], False)} | {fmt0(mdif['mdd'])} |", "",
                  "Carry positivo = o juro maior do Brasil compensou a alta do dólar no período. Não inclui imposto nem IOF.", ""]
        except Exception as e:
            L += ["## Carry do dólar", "", f"Não consegui calcular: {e}", ""]
    L += ["## Limites (autocrítica)", "",
          "- Resultados de um único período histórico (2012 em diante para ações e dólar; 2017 em diante para cripto). Cripto teve poucos ciclos completos.",
          "- Ficar \"fora do mercado\" rende zero aqui; na vida real o dinheiro parado renderia CDI, o que favoreceria as estratégias de tendência no Brasil.",
          "- Custos fixos por classe; sem impostos. Rebalanceamento no fechamento.",
          "- O t compara retorno ajustado ao risco em blocos mensais; ativos que andam juntos fazem os resultados não serem independentes entre si.",
          "", f"Tempo de execução: {time.time() - ini:.0f} s.", ""]
    md = "\n".join(L)
    os.makedirs(B.SAIDA, exist_ok=True)
    open(os.path.join(B.SAIDA, "estrategias.md"), "w", encoding="utf-8").write(md)
    json.dump(resumo, open(os.path.join(B.SAIDA, "estrategias-resumo.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not args.sintetico:
        try:
            vault = json.load(open(os.path.join(B.BASE, "config.json"), encoding="utf-8"))["vault"]
            pasta = os.path.join(vault, "01 Projeto")
            if os.path.isdir(pasta):
                nome = f"Estratégias {datetime.now(BRT):%Y-%m-%d %Hh%M}.md"
                open(os.path.join(pasta, nome), "w", encoding="utf-8").write(md)
                B.log(f"relatório gravado no vault: 01 Projeto/{nome}")
        except Exception as e:
            B.log(f"não gravei no vault: {e}")
    B.log(f"pronto em {time.time() - ini:.0f} s. Relatório: {os.path.join(B.SAIDA, 'estrategias.md')}")
    print(json.dumps(resumo, ensure_ascii=False))


if __name__ == "__main__":
    main()
