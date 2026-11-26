"""
MF Tracker — gestão de risco do guia (Murphy): stop, alvo de 3 para 1 e stop móvel.

O acerto sozinho não decide o resultado. O próprio sistema de exemplo do guia acerta 49% e lucra porque ganha
2,9 vezes mais nas vencedoras do que perde nas perdedoras. Aqui cada operação tem:
  * direção pela tendência diária (preço contra a média de 20 dias), como o guia manda;
  * uma de três ENTRADAS do guia:
      - recuo:      em alta, compra quando o RSI cai abaixo de 40 (em baixa, vende acima de 60): "compre as quedas";
      - rompimento: preço passa a máxima (ou mínima) dos últimos 20 candles a favor da tendência;
      - cruzamento: média de 9 cruza a de 21 a favor da tendência;
  * um STOP técnico: 1,5 ou 2,5 vezes a amplitude média (ATR de 14 candles);
  * uma de duas SAÍDAS:
      - alvo:   alvo de 3 vezes o risco (3:1), ou sai no prazo máximo (3 dias);
      - móvel:  sem alvo, stop móvel a 3 ATR do melhor preço ("deixe o lucro correr"), prazo máximo de 7 dias.
  12 combinações por mercado. Stop e alvo no mesmo candle contam como stop (conservador); gap além do stop sai na abertura.

Medidas: acerto, ganho médio ÷ perda média, FATOR DE LUCRO (ganho bruto ÷ perda bruta; o guia pede 2 ou mais),
resultado médio em R (R = o risco da operação) e líquido de custo, e a MAIOR QUEDA arriscando 1% da banca por operação.
Sem olhar o futuro: a combinação é escolhida na 1ª metade do histórico de cada mercado e medida na 2ª.
No fim, mostra as operações da última semana com a combinação escolhida.

Uso: python risco.py               (cache de 1h do backtest_intradiario e do semana.py)
     python risco.py --sintetico
"""
import argparse, json, math, os, sys, time
from datetime import datetime, timezone, timedelta
from itertools import product

import numpy as np
import pandas as pd

import backtest as B
import backtest_intradiario as BI
import semana as S

BRT = timezone(timedelta(hours=-3))
ENTRADAS = {"recuo": "recuo (RSI) a favor da tendência", "rompimento": "rompimento de 20 candles a favor da tendência",
            "cruzamento": "cruzamento das médias 9/21 a favor da tendência"}
STOPS = [1.5, 2.5]
SAIDAS = {"alvo": "alvo 3:1 ou 3 dias", "movel": "stop móvel de 3 ATR, até 7 dias"}
RISCO_BANCA = 0.01
GRUPOS = {"crypto": "Cripto", "meme": "Memecoins", "b3": "B3", "us": "EUA", "fx": "Dólar"}


def atr(df, n=14):
    H, L, C = (df[k].to_numpy(float) for k in ("h", "l", "c"))
    pc = np.r_[C[0], C[:-1]]
    tr = np.maximum.reduce([H - L, np.abs(H - pc), np.abs(L - pc)])
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def operar(a, I, A, entrada, k_stop, saida, ini, fim):
    """Simula operações de um ativo entre os candles com tempo em [ini, fim). Uma posição por vez."""
    df = a["df"]; O, H, L, C, T = (df[x].to_numpy(float) for x in ("o", "h", "l", "c", "t"))
    N = len(C); bpd = I["bpd"]; custo = S.CUSTO[a["cls"]]
    max_bar = 3 * bpd if saida == "alvo" else 7 * bpd
    ops, i = [], max(30 * bpd, 60)
    while i < N - 1:
        if T[i] < ini:
            i += 1; continue
        if T[i] >= fim:
            break
        td = 0 if not np.isfinite(I["dia_sma"][i]) else (1 if C[i] > I["dia_sma"][i] else -1)
        sinal = False
        if td:
            if entrada == "recuo":
                sinal = (td > 0 and I["rsi"][i] < 40 <= I["rsi"][i - 1]) or (td < 0 and I["rsi"][i] > 60 >= I["rsi"][i - 1])
            elif entrada == "rompimento":
                sinal = (td > 0 and C[i] > I["hi20"][i]) or (td < 0 and C[i] < I["lo20"][i])
            else:
                d0 = I["ema9"][i] - I["ema21"][i]; d1 = I["ema9"][i - 1] - I["ema21"][i - 1]
                sinal = (td > 0 and d0 > 0 >= d1) or (td < 0 and d0 < 0 <= d1)
        if not sinal or not np.isfinite(A[i]) or A[i] <= 0:
            i += 1; continue
        d, e = td, C[i]; risco = k_stop * A[i]
        stop = e - d * risco; alvo = e + d * 3 * risco if saida == "alvo" else None
        melhor = e; sai, j = None, i
        for j in range(i + 1, min(N, i + 1 + max_bar)):
            if d > 0:
                if O[j] <= stop: sai = O[j]; break
                if L[j] <= stop: sai = stop; break
                if alvo is not None and H[j] >= alvo: sai = max(alvo, O[j]); break
                if saida == "movel":
                    melhor = max(melhor, H[j]); stop = max(stop, melhor - 3 * A[j])
            else:
                if O[j] >= stop: sai = O[j]; break
                if H[j] >= stop: sai = stop; break
                if alvo is not None and L[j] <= alvo: sai = min(alvo, O[j]); break
                if saida == "movel":
                    melhor = min(melhor, L[j]); stop = min(stop, melhor + 3 * A[j])
        if sai is None:
            sai = C[j]
        ret = d * (sai / e - 1)
        ops.append({"sym": a["sym"], "cls": a["cls"], "t": int(T[i]), "t_saida": int(T[j]), "dir": d, "entrada": e, "saida": sai,
                    "ret": ret, "liq": ret - custo, "R": (ret - custo) / (risco / e), "risco_pct": risco / e, "barras": j - i})
        i = j + 1
    return ops


def metricas(ops):
    if not ops:
        return None
    liq = np.array([o["liq"] for o in ops]); R = np.array([o["R"] for o in ops])
    g, p = liq[liq > 0], -liq[liq <= 0]
    fl = g.sum() / p.sum() if p.sum() > 0 else float("inf")
    # banca arriscando 1% por operação, em ordem de saída
    banca, pico, queda = 1.0, 1.0, 0.0
    for o in sorted(ops, key=lambda x: x["t_saida"]):
        banca *= 1 + RISCO_BANCA * o["R"]; pico = max(pico, banca); queda = max(queda, 1 - banca / pico)
    # t agrupado por dia de entrada (operações do mesmo mercado no mesmo dia andam juntas)
    df = pd.DataFrame({"dia": [o["t"] // 86400 for o in ops], "R": R}).groupby("dia")["R"].mean()
    t = float(df.mean() / (df.std(ddof=1) / math.sqrt(len(df)))) if len(df) > 2 and df.std(ddof=1) > 0 else 0.0
    return {"n": len(ops), "acerto": float((liq > 0).mean()), "ganho_medio": float(g.mean()) if len(g) else 0.0,
            "perda_media": float(p.mean()) if len(p) else 0.0, "fl": float(fl), "R": float(R.mean()), "liq": float(liq.mean()),
            "banca": banca - 1, "queda": queda, "t": t, "barras": float(np.mean([o["barras"] for o in ops]))}


def linha(nome, m):
    if not m:
        return f"| {nome} | 0 | — | — | — | — | — | — | — |"
    razao = m["ganho_medio"] / m["perda_media"] if m["perda_media"] > 0 else float("inf")
    return (f"| {nome} | {m['n']} | {B.pc(m['acerto'])} | {razao:.2f} | {m['fl']:.2f} | {m['R']:+.3f} R | {B.pct(m['liq'], 3)} | "
            f"{B.pct(m['banca'], 1)} / {B.pct(-m['queda'], 1)} | {m['t']:.2f} |")


CAB = ["| Combinação | Operações | Acerto | Ganho médio ÷ perda média | Fator de lucro | Resultado médio | Líquido por operação | Banca (1% de risco) / maior queda | t |",
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sintetico", action="store_true"); args = ap.parse_args()
    ini_exec = time.time()
    ativos = S.sinteticos() if args.sintetico else [a for a in BI.carregar() if len(a["df"]) >= 3000] + S.carregar_yahoo()
    ativos = [a for a in ativos if len(a["df"]) >= 1000]
    if not ativos:
        B.log("sem dados"); sys.exit(1)
    fim = max(int(a["df"]["t"].iloc[-1]) for a in ativos) + 1
    ini_semana = fim - 7 * 86400
    for a in ativos:
        a["I"] = S.indicadores(a["df"]); a["A"] = atr(a["df"])
        T = a["df"]["t"].to_numpy(); a["meio"] = int(T[len(T) // 2])
    combos = list(product(ENTRADAS, STOPS, SAIDAS))
    res = {}
    for cls in GRUPOS:
        g = [a for a in ativos if a["cls"] == cls]
        if not g:
            continue
        for c in combos:
            o1, o2, os_ = [], [], []
            for a in g:
                T0 = int(a["df"]["t"].iloc[0])
                o1 += operar(a, a["I"], a["A"], *c, T0, a["meio"])
                o2 += operar(a, a["I"], a["A"], *c, a["meio"], fim)
            res[(cls, c)] = (metricas(o1), metricas(o2), [o for o in o2 if o["t"] >= ini_semana])
        B.log(f"{cls:6s} {len(g)} ativos, {len(combos)} combinações calculadas")
    escolha = {}
    for cls in GRUPOS:
        cand = [(c, res[(cls, c)]) for c in combos if (cls, c) in res and res[(cls, c)][0] and res[(cls, c)][0]["n"] >= 30]
        if cand:
            c, (m1, m2, sem) = max(cand, key=lambda x: x[1][0]["R"])   # escolha só pela 1ª metade
            escolha[cls] = (c, m1, m2, sem)
            B.log(f"{cls:6s} escolhida: {c} | 1ª: R {m1['R']:+.3f} FL {m1['fl']:.2f} | 2ª: " +
                  (f"{m2['n']} op, acerto {m2['acerto']*100:.1f}%, FL {m2['fl']:.2f}, R {m2['R']:+.3f}, t {m2['t']:.2f}" if m2 else "sem operações"))
    nome_c = lambda c: f"{ENTRADAS[c[0]].split(' (')[0].split(' de')[0].capitalize()} · stop {str(c[1]).replace('.', ',')} ATR · {SAIDAS[c[2]]}"
    hoje = datetime.now(BRT)
    L = ["---", "tipo: backtest", "gerado_por: MF Tracker risco.py", f"data: {hoje:%Y-%m-%dT%H:%M}-03:00",
         f"dados: {'SINTÉTICOS (teste do código)' if args.sintetico else 'reais'}", "tags: [backtest, trading, risco]", "---", "",
         f"# Gestão de risco do guia: stop, alvo 3:1 e stop móvel — {hoje:%d/%m/%Y}", ""]
    if args.sintetico:
        L += ["> **Dados sintéticos** (passeio aleatório). O resultado certo é fator de lucro perto de 1 antes do custo e abaixo de 1 depois.", ""]
    L += ["Cada operação segue o guia: **direção pela tendência diária**, entrada por recuo, rompimento ou cruzamento de médias, "
          "**stop técnico** (1,5 ou 2,5 ATR) e saída por **alvo de 3 vezes o risco** ou por **stop móvel** (deixar o lucro correr). "
          "R = o risco da operação (distância até o stop); 1 R de resultado médio quer dizer ganhar, em média, o que se arriscou. "
          "**Fator de lucro** = ganho bruto ÷ perda bruta (o guia pede 2 ou mais; abaixo de 1 perde dinheiro). "
          "A banca simulada arrisca 1% por operação. Custos já descontados. "
          "**A combinação de cada mercado é escolhida na 1ª metade do histórico; só a 2ª metade conta.**", "",
          "## Resultado fora da amostra (2ª metade), combinação escolhida", "", "| Mercado | " + CAB[0][2:], "|---|" + CAB[1][1:]]
    for cls, (c, m1, m2, sem) in escolha.items():
        L.append(linha(f"{GRUPOS[cls]} | {nome_c(c)}", m2))
    L += ["", "**Veredito por mercado:**", ""]
    for cls, (c, m1, m2, sem) in escolha.items():
        if not m2:
            L.append(f"- {GRUPOS[cls]}: sem operações na 2ª metade."); continue
        if m2["fl"] >= 1.3 and m2["t"] >= 2:
            v = "lucrou fora da amostra com folga estatística: candidata a acompanhar ao vivo."
        elif m2["fl"] > 1 or m2["R"] > 0 or m2["liq"] > 0:
            v = f"resultado misto fora da amostra (fator de lucro {m2['fl']:.2f}, {m2['R']:+.3f} R, líquido {B.pct(m2['liq'], 3)}), sem folga estatística (t {m2['t']:.2f}): sem vantagem confirmada."
        else:
            v = "perdeu fora da amostra depois dos custos."
        L.append(f"- {GRUPOS[cls]} ({nome_c(c)}): {v}")
    L += ["", "## Todas as combinações na 2ª metade (só para estudo: escolher a melhor aqui seria olhar o futuro)", ""]
    for cls in GRUPOS:
        if not any((cls, c) in res for c in combos):
            continue
        L += [f"### {GRUPOS[cls]}", "", CAB[0], CAB[1]]
        for c in combos:
            if (cls, c) in res:
                L.append(linha(nome_c(c), res[(cls, c)][1]))
        L.append("")
    L += ["## Operações da última semana (combinação escolhida de cada mercado)", "",
          "| Entrada (Brasília) | Ativo | Direção | Preço de entrada | Saída (Brasília) | Preço de saída | Risco até o stop | Resultado líquido | Em R |",
          "|---|---|---|---:|---|---:|---:|---:|---:|"]
    sem_all = []
    for cls, (c, m1, m2, sem) in escolha.items():
        sem_all += sem
    for o in sorted(sem_all, key=lambda x: x["t"]):
        L.append(f"| {datetime.fromtimestamp(o['t'], BRT):%d/%m %H:%M} | {o['sym']} | {'compra' if o['dir'] > 0 else 'venda'} | {o['entrada']:.6g} | "
                 f"{datetime.fromtimestamp(o['t_saida'], BRT):%d/%m %H:%M} | {o['saida']:.6g} | {B.pct(o['risco_pct'], 2)} | {B.pct(o['liq'], 2)} | {o['R']:+.2f} |")
    ms = metricas(sem_all)
    if ms:
        L += ["", f"Semana: {ms['n']} operações, acerto {B.pc(ms['acerto'])}, fator de lucro {ms['fl']:.2f}, resultado médio {ms['R']:+.2f} R, "
              f"banca com 1% de risco {B.pct(ms['banca'], 1)}. Uma semana é amostra pequena demais para concluir."]
    L += ["", "## Limites (autocrítica)", "",
          "- Candles de 1 hora: dentro da hora não se sabe se o stop ou o alvo veio primeiro; quando os dois cabem no mesmo candle, conta como stop.",
          "- Custo fixo por mercado (cripto 0,3%, memecoins 0,6%, bolsa 0,1%, dólar 0,5%); derrapagem em gaps só na abertura do candle seguinte.",
          "- Venda a descoberto supõe que dá para vender (futuros, aluguel de ações); sem custo de aluguel ou financiamento.",
          "- 12 combinações por mercado: a melhor da 1ª metade tende a piorar na 2ª; é exatamente o que este teste mede.",
          "- Ativos do mesmo mercado andam juntos: o t agrupa por dia de entrada.", "", f"Tempo de execução: {time.time() - ini_exec:.0f} s.", ""]
    md = "\n".join(L)
    os.makedirs(B.SAIDA, exist_ok=True)
    open(os.path.join(B.SAIDA, "risco.md"), "w", encoding="utf-8").write(md)
    if not args.sintetico:
        try:
            vault = json.load(open(os.path.join(B.BASE, "config.json"), encoding="utf-8"))["vault"]
            if os.path.isdir(os.path.join(vault, "01 Projeto")):
                nome = f"Gestão de risco (stop e alvo) {hoje:%Y-%m-%d %Hh%M}.md"
                open(os.path.join(vault, "01 Projeto", nome), "w", encoding="utf-8").write(md)
                B.log(f"gravado no vault: 01 Projeto/{nome}")
        except Exception as e:
            B.log(f"não gravei no vault: {e}")
    B.log(f"pronto em {time.time() - ini_exec:.0f} s")


if __name__ == "__main__":
    main()
