"""
MF Tracker — a faixa de preço prevista acerta?

Prever a direção não funcionou em nenhum prazo. O tamanho do movimento costuma ser bem mais previsível
(dias agitados vêm em sequência). Aqui comparamos dois jeitos de prever a faixa onde o preço vai estar
daqui a h horas:

  * fixa:      quantis de todos os retornos de h horas das últimas 1.200 janelas (o que a aba Trading fazia);
  * ajustada:  os mesmos quantis, mas medidos em "desvios" e multiplicados pela volatilidade de agora
               (média exponencial dos retornos de 1 hora, fator 0,94, como o RiskMetrics).

Sem olhar o futuro: na hora i só entram janelas que já terminaram (j + h <= i).
Uma faixa boa cobre o que promete (50% e 90%) em qualquer clima de mercado, calmo ou agitado,
e é a mais estreita possível. A nota de pinball resume as duas coisas (menor é melhor).

Uso: python faixa.py               (cache de candles de 1h do backtest_intradiario, ou baixa)
     python faixa.py --sintetico
"""
import argparse, json, math, os, sys, time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

import backtest as B
import backtest_intradiario as BI

BRT = timezone(timedelta(hours=-3))
HORAS = [1, 4, 12, 24]
JANELA = 1200
LAMBDA = 0.94
QS = [0.05, 0.25, 0.75, 0.95]


def ewma_vol(lr, lam=LAMBDA):
    v = np.empty(len(lr)); s2 = np.nanvar(lr[:200]) if len(lr) > 200 else 1e-4
    for i, x in enumerate(lr):
        s2 = lam * s2 + (1 - lam) * x * x
        v[i] = math.sqrt(s2)
    return v


ALFAS = [0.0, 0.25, 0.5, 0.75, 1.0]   # 0 = fixa, 1 = ajustada; no meio, a volatilidade de agora pesa menos


def avaliar(a, h):
    """Previsões da 1ª e da 2ª metade para cada alfa. A faixa usa vol^alfa: alfa escolhido só na 1ª metade."""
    C = a["df"]["c"].to_numpy(float); N = len(C)
    lr = np.r_[0.0, np.diff(np.log(C))]
    sig = ewma_vol(lr)                                   # sig[i] usa retornos até i
    r = np.full(N, np.nan); r[:N - h] = C[h:] / C[:N - h] - 1   # r[j]: retorno de j a j+h (conhecido em j+h)
    qs = {}
    for al in ALFAS:
        esc = (sig * math.sqrt(h)) ** al
        known = pd.Series(r / esc).shift(h)              # na hora i, só janelas com j+h <= i
        qs[al] = ({q: known.rolling(JANELA, min_periods=JANELA // 2).quantile(q).to_numpy() for q in QS}, esc)
    idx = np.arange(max(JANELA, 300), N - h, h)          # uma previsão a cada h horas, sem sobreposição
    linhas = []
    for i in idx:
        real = r[i]
        if not np.isfinite(real):
            continue
        rec = {"t": int(a["df"]["t"].iloc[i]), "sym": a["sym"], "cls": a["cls"], "h": h, "real": real, "sig": sig[i], "metade": 0 if i < N // 2 else 1}
        ok = True
        for al, (qq, esc) in qs.items():
            qv = {q: qq[q][i] * esc[i] for q in QS}
            if not all(np.isfinite(v) for v in qv.values()):
                ok = False; break
            rec[al] = qv
        if ok:
            linhas.append(rec)
    return linhas


def pinball(real, qv):
    return float(np.mean([max(q * (real - v), (q - 1) * (real - v)) for q, v in qv.items()]))


def resumo(ls, nome):
    if not ls:
        return None
    c50 = np.mean([(l[nome][0.25] <= l["real"] <= l[nome][0.75]) for l in ls])
    c90 = np.mean([(l[nome][0.05] <= l["real"] <= l[nome][0.95]) for l in ls])
    w50 = np.mean([l[nome][0.75] - l[nome][0.25] for l in ls])
    pb = np.mean([pinball(l["real"], l[nome]) for l in ls])
    return {"n": len(ls), "c50": c50, "c90": c90, "w50": w50, "pb": pb}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sintetico", action="store_true"); args = ap.parse_args()
    ini = time.time()
    ativos = BI.sinteticos() if args.sintetico else BI.carregar()
    ativos = [a for a in ativos if len(a["df"]) >= 3000]
    if not ativos:
        B.log("sem histórico suficiente"); sys.exit(1)
    todas, alfa = {}, {}
    for h in HORAS:
        ls = []
        for a in ativos:
            ls += avaliar(a, h)
        m1 = [l for l in ls if l["metade"] == 0]
        alfa[h] = min(ALFAS, key=lambda al: resumo(m1, al)["pb"])   # escolhido só na 1ª metade
        for l in ls:
            l["fixa"], l["ajustada"], l["mista"] = l[0.0], l[1.0], l[alfa[h]]
        ls = [l for l in ls if l["metade"] == 1]
        todas[h] = ls
        f, j, m = resumo(ls, "fixa"), resumo(ls, "ajustada"), resumo(ls, "mista")
        B.log(f"{h:2d}h {len(ls):6d} previsões | alfa da 1ª metade {alfa[h]} | cobre 50%: fixa {f['c50']*100:.1f}% ajustada {j['c50']*100:.1f}% mista {m['c50']*100:.1f}% | "
              f"pinball ajustada/fixa {j['pb']/f['pb']:.3f} mista/fixa {m['pb']/f['pb']:.3f}")
    hoje = datetime.now(BRT)
    L = ["---", "tipo: backtest", "gerado_por: MF Tracker faixa.py", f"data: {hoje:%Y-%m-%dT%H:%M}-03:00",
         f"dados: {'SINTÉTICOS (teste do código)' if args.sintetico else 'reais'}", "tags: [backtest, trading, volatilidade]", "---", "",
         f"# A faixa de preço prevista acerta? — {hoje:%d/%m/%Y}", ""]
    if args.sintetico:
        L += ["> **Dados sintéticos** com volatilidade constante: as faixas devem empatar e alfa perto de 0 é o esperado.", ""]
    V = ("fixa", "ajustada", "mista")
    tri = lambda ls, k, fmt: " / ".join(fmt(resumo(ls, v)[k]) for v in V)
    pbr = lambda ls: f"{resumo(ls, 'ajustada')['pb']/resumo(ls, 'fixa')['pb']:.3f} / {resumo(ls, 'mista')['pb']/resumo(ls, 'fixa')['pb']:.3f}"
    L += ["Três formas de prever onde o preço vai estar daqui a h horas. **Fixa**: os quantis dos retornos das últimas 1.200 janelas. "
          "**Ajustada**: os mesmos quantis em desvios, multiplicados pela volatilidade de agora (média exponencial, fator 0,94). "
          "**Mista**: meio-termo, com a volatilidade de agora elevada a um expoente alfa entre 0 e 1, **escolhido só na 1ª metade** pela nota de pinball. "
          "Os números abaixo são só da 2ª metade, uma previsão a cada h horas por ativo, sem olhar o futuro. "
          "Uma faixa honesta cobre 50% e 90% dos casos **em qualquer clima**; a nota de pinball junta acerto e largura (menor é melhor; abaixo de 1 = melhor que a fixa).", "",
          "## Resumo", "",
          "| Prazo | Alfa da mista | Previsões | Cobre 50% (fixa / ajustada / mista) | Cobre 90% | Largura da faixa de 50% | Pinball ÷ fixa (ajustada / mista) |",
          "|---:|---:|---:|---|---|---|---|"]
    for h in HORAS:
        ls = todas[h]
        L.append(f"| {h} h | {alfa[h]} | {len(ls)} | {tri(ls, 'c50', B.pc)} | {tri(ls, 'c90', B.pc)} | {tri(ls, 'w50', lambda x: B.pct(x, 2))} | {pbr(ls)} |")
    L += ["", "## Por clima de mercado (terço da volatilidade de agora, dentro de cada ativo)", "",
          "| Prazo | Clima | Cobre 50% (fixa / ajustada / mista) | Cobre 90% | Pinball ÷ fixa (ajustada / mista) |", "|---:|---|---|---|---|"]
    for h in HORAS:
        df = pd.DataFrame([{"i": k, "sym": l["sym"], "sig": l["sig"]} for k, l in enumerate(todas[h])])
        df["terco"] = df.groupby("sym")["sig"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
        for t, nome in ((0, "calmo"), (1, "normal"), (2, "agitado")):
            ls = [todas[h][k] for k in df.loc[df["terco"] == t, "i"]]
            if ls:
                L.append(f"| {h} h | {nome} | {tri(ls, 'c50', B.pc)} | {tri(ls, 'c90', B.pc)} | {pbr(ls)} |")
    L += ["", "## Por mercado", "", "| Prazo | Mercado | Cobre 50% (fixa / ajustada / mista) | Cobre 90% | Pinball ÷ fixa (ajustada / mista) |", "|---:|---|---|---|---|"]
    for h in HORAS:
        for cls, lab in (("crypto", "Cripto"), ("meme", "Memecoins")):
            ls = [l for l in todas[h] if l["cls"] == cls]
            if ls:
                L.append(f"| {h} h | {lab} | {tri(ls, 'c50', B.pc)} | {tri(ls, 'c90', B.pc)} | {pbr(ls)} |")
    os.makedirs(B.SAIDA, exist_ok=True)
    json.dump({"gerado": hoje.strftime("%Y-%m-%dT%H:%M-03:00"), "lambda": LAMBDA, "janela": JANELA, "alfa": {f"{h}h": alfa[h] for h in HORAS}},
              open(os.path.join(B.SAIDA, "faixa.json"), "w", encoding="utf-8"), indent=1)
    L += ["", "## Limites (autocrítica)", "",
          "- Preço de fechamento da hora; o caminho dentro da hora (stop tocado e voltou) não entra aqui.",
          "- Fator 0,94 fixado antes do teste, sem ajuste fino.",
          "- Faixa prevê o tamanho do movimento, não o sentido: não diz se é para comprar ou vender.", "",
          f"Tempo de execução: {time.time() - ini:.0f} s.", ""]
    md = "\n".join(L)
    os.makedirs(B.SAIDA, exist_ok=True)
    open(os.path.join(B.SAIDA, "faixa.md"), "w", encoding="utf-8").write(md)
    if not args.sintetico:
        try:
            vault = json.load(open(os.path.join(B.BASE, "config.json"), encoding="utf-8"))["vault"]
            if os.path.isdir(os.path.join(vault, "01 Projeto")):
                nome = f"Faixa de preço {hoje:%Y-%m-%d %Hh%M}.md"
                open(os.path.join(vault, "01 Projeto", nome), "w", encoding="utf-8").write(md)
                os.makedirs(os.path.join(vault, "99 Meta"), exist_ok=True)
                open(os.path.join(vault, "99 Meta", "faixa.json"), "w", encoding="utf-8").write(open(os.path.join(B.SAIDA, "faixa.json"), encoding="utf-8").read())
                B.log(f"gravado no vault: 01 Projeto/{nome} e 99 Meta/faixa.json")
        except Exception as e:
            B.log(f"não gravei no vault: {e}")
    B.log(f"pronto em {time.time() - ini:.0f} s")


if __name__ == "__main__":
    main()
