"""
MF Tracker — a reversão de curtíssimo prazo vira lucro com custo menor?

O backtest de 1 hora (backtest_intradiario.py) mostrou que, em 1 a 4 horas, os fatores funcionam ao contrário
(o preço tende a devolver parte do movimento), mas o efeito não pagava 0,3% a 0,6% de custo de ida e volta.
Aqui a pergunta é direta: quanto essa reversão rende ANTES do custo, e qual custo ela aguenta?

Regra testada (simples de propósito, para caber num teste honesto):
  * z = -(retorno das últimas L horas) / (volatilidade das últimas 168 horas * raiz de L)
  * se |z| >= k, entra no fechamento da hora na direção de z (caiu forte: compra; subiu forte: vende)
  * sai depois de h horas; operações de um mesmo ativo não se sobrepõem
  * L em {1, 2, 4, 8, 24}, h em {1, 2, 4, 8}, k em {1, 1,5, 2, 2,5, 3}: 100 combinações por mercado

Sem olhar o futuro: a combinação é escolhida na 1ª metade do histórico de cada ativo e medida só na 2ª.
O t agrupa as operações que começam na mesma hora (ativos que andam juntos contam como uma observação).

Uso: python custos.py               (usa o cache de candles de 1h do backtest_intradiario, ou baixa)
     python custos.py --sintetico   (passeio aleatório: o resultado certo é bruto perto de zero)
"""
import argparse, json, math, os, sys, time
from datetime import datetime, timezone, timedelta
from itertools import product

import numpy as np
import pandas as pd

import backtest as B
import backtest_intradiario as BI

BRT = timezone(timedelta(hours=-3))
LOOKS = [1, 2, 4, 8, 24]
HOLDS = [1, 2, 4, 8]
KS = [1.0, 1.5, 2.0, 2.5, 3.0]
CUSTOS = [0.0, 0.0008, 0.0015, 0.002, 0.003, 0.006]  # ida e volta
VOLN = 168
MIN_OP = 150


def operacoes(a, L, h, k):
    """Lista de (t_entrada, metade, bruto, excesso) para um ativo."""
    C = a["df"]["c"].to_numpy(float); T = a["df"]["t"].to_numpy()
    lr = np.r_[0.0, np.diff(np.log(C))]
    vol = pd.Series(lr).rolling(VOLN).std().to_numpy()
    N = len(C); mid = N // 2
    out = []
    i = max(VOLN, L)
    while i + h < N:
        v = vol[i]
        if np.isfinite(v) and v > 0:
            z = -math.log(C[i] / C[i - L]) / (v * math.sqrt(L))
            if abs(z) >= k:
                d = 1 if z > 0 else -1
                r = C[i + h] / C[i] - 1
                out.append((int(T[i]), 0 if i < mid else 1, d * r, d))
                i += h
                continue
        i += 1
    return out


def resumo(ops, mu, custo=0.0):
    """ops: lista de (t, cls, bruto, dir). mu: deriva média por h do ativo, já aplicada fora. Devolve n, médias e t agrupado por hora."""
    if len(ops) < 2:
        return None
    df = pd.DataFrame(ops, columns=["t", "bruto", "exc"])
    df["liq"] = df["bruto"] - custo
    g = df.groupby("t")[["liq", "exc"]].mean()
    n_g = len(g)
    def tt(x):
        s = x.std(ddof=1)
        return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 and len(x) > 1 else 0.0
    return {"n": len(df), "horas": n_g, "bruto": float(df["bruto"].mean()), "liq": float(df["liq"].mean()),
            "acerto": float((df["liq"] > 0).mean()), "exc": float(df["exc"].mean()), "t_exc": tt(g["exc"] + custo * 0) if n_g > 1 else 0.0,
            "t_liq": tt(g["liq"])}


def coletar(ativos, cls, L, h, k, metade):
    ops = []
    for a in ativos:
        if a["cls"] != cls:
            continue
        mu = a["mu_h"][h][metade]
        for t, m, b, d in a["ops"][(L, h, k)]:
            if m == metade:
                ops.append((t, b, b - d * mu))  # excesso: tira a parte que vem só da deriva do ativo
    return ops


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sintetico", action="store_true"); args = ap.parse_args()
    ini = time.time()
    ativos = BI.sinteticos() if args.sintetico else BI.carregar()
    ativos = [a for a in ativos if len(a["df"]) >= 3000]
    if not ativos:
        B.log("sem histórico suficiente"); sys.exit(1)
    B.log("calculando operações das 100 combinações...")
    for a in ativos:
        C = a["df"]["c"].to_numpy(float); N = len(C); mid = N // 2
        a["mu_h"] = {}
        for h in HOLDS:
            r = C[h:] / C[:-h] - 1
            a["mu_h"][h] = (float(r[:mid - h].mean()), float(r[mid:].mean()))
        a["ops"] = {(L, h, k): operacoes(a, L, h, k) for L, h, k in product(LOOKS, HOLDS, KS)}
    classes = [c for c in ("crypto", "meme") if any(a["cls"] == c for a in ativos)]
    escolha, grade = {}, {}
    for cls in classes:
        melhor = None
        for L, h, k in product(LOOKS, HOLDS, KS):
            s1 = resumo(coletar(ativos, cls, L, h, k, 0), None)
            if not s1 or s1["n"] < MIN_OP:
                continue
            s2 = resumo(coletar(ativos, cls, L, h, k, 1), None)
            grade[(cls, L, h, k)] = (s1, s2)
            if melhor is None or s1["t_exc"] > melhor[1]["t_exc"]:
                melhor = ((L, h, k), s1)
        escolha[cls] = melhor
        if melhor:
            (L, h, k), s1 = melhor
            s2 = grade[(cls, L, h, k)][1]
            B.log(f"{cls:6s} escolhida na 1ª metade: L={L}h h={h}h k={k} | 1ª: {s1['n']} op bruto {s1['bruto']*100:+.3f}% t {s1['t_exc']:.2f} | "
                  f"2ª: {s2['n'] if s2 else 0} op bruto {(s2['bruto'] if s2 else 0)*100:+.3f}% t {(s2['t_exc'] if s2 else 0):.2f}")
    # relatório
    hoje = datetime.now(BRT)
    L_ = ["---", "tipo: backtest", "gerado_por: MF Tracker custos.py", f"data: {hoje:%Y-%m-%dT%H:%M}-03:00",
          f"dados: {'SINTÉTICOS (teste do código)' if args.sintetico else 'reais'}", "tags: [backtest, trading, custos]", "---", "",
          f"# Reversão de curto prazo e custo de operar — {hoje:%d/%m/%Y}", ""]
    if args.sintetico:
        L_ += ["> **Dados sintéticos** (passeio aleatório). O resultado certo é bruto perto de zero e t abaixo de 2.", ""]
    L_ += ["Pergunta: o preço que andou forte nas últimas horas devolve parte do movimento, e isso paga o custo de entrar e sair? "
           "Regra: se o retorno das últimas L horas passou de k desvios, entra no sentido contrário e sai depois de h horas. "
           "100 combinações por mercado; **a escolhida é a de maior t na 1ª metade do histórico, e só a 2ª metade conta**. "
           "**Bruto** é antes do custo; **acima da tendência** tira o que viria só da deriva do ativo; o t agrupa operações que começam na mesma hora.", ""]
    L_ += ["## Resultado fora da amostra (2ª metade)", ""]
    for cls in classes:
        lab = {"crypto": "Cripto", "meme": "Memecoins"}[cls]
        if not escolha[cls]:
            L_ += [f"**{lab}:** nenhuma combinação com operações suficientes.", ""]; continue
        (L, h, k), s1 = escolha[cls]
        ops2 = coletar(ativos, cls, L, h, k, 1)
        s2 = resumo(ops2, None)
        L_ += [f"**{lab}** — combinação escolhida na 1ª metade: últimas {L} h, sai em {h} h, limiar {k} desvios "
               f"(1ª metade: {s1['n']} operações, bruto {B.pct(s1['bruto'], 3)}, t {s1['t_exc']:.2f}).", ""]
        if not s2:
            L_ += ["Sem operações na 2ª metade.", ""]; continue
        L_ += [f"2ª metade: {s2['n']} operações em {s2['horas']} horas distintas · bruto {B.pct(s2['bruto'], 3)} por operação · "
               f"acima da tendência {B.pct(s2['exc'], 3)} · t {s2['t_exc']:.2f}. **Custo máximo que aguenta (ida e volta): {B.pct(max(0.0, s2['exc']), 3)}.**", "",
               "**Veredito:** " + ("a reversão existe fora da amostra (t ≥ 2); vale se o seu custo real ficar abaixo do limite acima." if s2['t_exc'] >= 2 and s2['exc'] > 0
                                   else "não confirmada fora da amostra (t abaixo de 2): não dá para contar com ela, qualquer que seja o custo."), "",
               "| Custo de ida e volta | Líquido por operação | Acerto | t |", "|---:|---:|---:|---:|"]
        for c in CUSTOS:
            r = resumo([(t, b, e) for t, b, e in ops2], None, c)
            L_.append(f"| {B.pct(c, 2)} | {B.pct(r['liq'], 3)} | {B.pc(r['acerto'])} | {r['t_liq']:.2f} |")
        L_.append("")
    L_ += ["## Bruto por prazo de saída (2ª metade, melhor L e k da 1ª metade para cada prazo)", "",
           "| Mercado | Sai em | L | k | Operações | Bruto | Acima da tendência | t |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cls in classes:
        for h in HOLDS:
            cand = [(key, v) for key, v in grade.items() if key[0] == cls and key[2] == h]
            if not cand:
                continue
            (c_, L, h_, k), (s1, s2) = max(cand, key=lambda x: x[1][0]["t_exc"])
            if s2:
                L_.append(f"| {'Cripto' if cls == 'crypto' else 'Memecoins'} | {h} h | {L} h | {k} | {s2['n']} | {B.pct(s2['bruto'], 3)} | {B.pct(s2['exc'], 3)} | {s2['t_exc']:.2f} |")
    L_ += ["", "## Como ler", "",
           "- Custos de referência (ida e volta): ordem limitada em corretora grande sai por 0,08% a 0,2%; ordem a mercado em corretora brasileira, 0,3% a 1%. "
           "O MF Tracker vinha usando 0,3% (cripto) e 0,6% (memecoins).",
           "- Se o bruto fora da amostra for menor que o seu custo real, a reversão não serve para você, mesmo que exista.",
           "", "## Limites (autocrítica)", "",
           "- Ordem limitada nem sempre é executada, e costuma ser executada justamente quando o preço continua contra (seleção adversa). "
           "O teste supõe execução no fechamento da hora, então o resultado real com ordem limitada tende a ser pior que o mostrado.",
           "- Venda a descoberto em cripto exige margem ou futuros, com taxa de financiamento que não entrou aqui.",
           "- Uma regra só, escolhida entre 100: o t da 1ª metade é inflado de propósito; só o da 2ª vale.",
           "- Memecoins listadas na Binance são sobreviventes.", "", f"Tempo de execução: {time.time() - ini:.0f} s.", ""]
    md = "\n".join(L_)
    os.makedirs(B.SAIDA, exist_ok=True)
    open(os.path.join(B.SAIDA, "custos.md"), "w", encoding="utf-8").write(md)
    if not args.sintetico:
        try:
            vault = json.load(open(os.path.join(B.BASE, "config.json"), encoding="utf-8"))["vault"]
            if os.path.isdir(os.path.join(vault, "01 Projeto")):
                nome = f"Reversão e custos {hoje:%Y-%m-%d %Hh%M}.md"
                open(os.path.join(vault, "01 Projeto", nome), "w", encoding="utf-8").write(md)
                B.log(f"gravado no vault: 01 Projeto/{nome}")
        except Exception as e:
            B.log(f"não gravei no vault: {e}")
    B.log(f"pronto em {time.time() - ini:.0f} s")


if __name__ == "__main__":
    main()
