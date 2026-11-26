"""
MF Tracker — 100 semanas simuladas, uma a uma.

Pedido do Luis (26/11/2026): pelo menos 100 simulações das últimas 100 semanas.
Para cada uma das últimas 100 semanas (sem olhar o futuro, só dados até cada hora):
  1. PALPITES (mesma regra do semana.py, guia de Murphy): acerto da semana, VENTO da semana (quanto acertaria quem
     chutasse sempre a mesma direção) e HABILIDADE = acerto − vento. Também cada técnica e cada situação.
  2. OPERAÇÕES com stop e alvo (mesmas 12 combinações do risco.py): resultado da semana em R e na banca.
Como semanas diferentes são quase independentes, 100 semanas dão uma medida honesta: uma regra que funciona
acerta acima do vento na maioria das semanas, e o t (média ÷ erro padrão entre semanas) passa de 2.
Com 12 combinações testadas ao mesmo tempo, uma passa de t 2 por sorte com chance alta; por isso o
limiar de confiança para as combinações é t ≥ 2,9 (correção de Bonferroni para 12 testes).

Uso: python semanas.py [--semanas 100]     (cache de 1h)
     python semanas.py --sintetico
"""
import argparse, json, math, os, sys, time
from datetime import datetime, timezone, timedelta
from itertools import product

import numpy as np
import pandas as pd

import backtest as B
import backtest_intradiario as BI
import semana as S
import risco as R

BRT = timezone(timedelta(hours=-3))
SEM = 7 * 86400


def tstat(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sintetico", action="store_true"); ap.add_argument("--semanas", type=int, default=100)
    args = ap.parse_args(); W = args.semanas
    ini_exec = time.time()
    ativos = S.sinteticos() if args.sintetico else [a for a in BI.carregar() if len(a["df"]) >= 3000] + S.carregar_yahoo()
    ativos = [a for a in ativos if len(a["df"]) >= 1000]
    if not ativos:
        B.log("sem dados"); sys.exit(1)
    fim = max(int(a["df"]["t"].iloc[-1]) for a in ativos) + 1
    ini = fim - W * SEM
    semana_de = lambda t: int((t - ini) // SEM)              # 0 = mais antiga, W-1 = a última
    B.log(f"{W} semanas: {datetime.fromtimestamp(ini, BRT):%d/%m/%Y} a {datetime.fromtimestamp(fim, BRT):%d/%m/%Y}")

    # ---------- 1. palpites ----------
    recs = []
    for a in ativos:
        ls = [l for l in S.simular(a, fim, ini) if l["dir"] != 0]   # inclui o ano anterior (semana=False) para a taxa base
        for l in ls:
            l["w_idx"] = semana_de(l["t"])
        recs += ls
        B.log(f"{a['sym']:7s} {sum(1 for l in ls if l['semana']):6d} palpites em {W} semanas")
    df = pd.DataFrame([{"w": l["w_idx"], "sym": l["sym"], "cls": l["cls"], "dir": l["dir"], "conc": l["conc"], "r": l.get("r_curto"),
                        "rd": l.get("r_dia"), **{f"v_{k}": x for k, x in l["votos"].items()}, "hora": l["hora_brt"], "sig": l["sig"],
                        "adx": l["adx"], "rsi": l["rsi"], "fx": l.get("faixa_curto")} for l in recs])
    df = df[df["r"].notna()].copy()
    # taxa base SEM olhar o futuro: fração das janelas de 4 candles que subiram nas 52 semanas ANTERIORES, por ativo
    # (mede o vento de longo prazo do ativo; usar a própria semana seria olhar o resultado)
    df["u"] = (df["r"] > 0).astype(float)
    ups = df.groupby(["sym", "w"])["u"].agg(["sum", "count"]).reset_index()
    base = []
    for sym, g in ups.groupby("sym"):
        g = g.set_index("w").reindex(range(int(g["w"].min()), W)).fillna(0)
        s_ = g["sum"].rolling(52, min_periods=8).sum().shift(1); c_ = g["count"].rolling(52, min_periods=8).sum().shift(1)
        base.append(pd.DataFrame({"sym": sym, "w": g.index, "up": (s_ / c_).to_numpy()}))
    df = df.merge(pd.concat(base), on=["sym", "w"], how="left")
    df = df[(df["w"] >= 0) & df["up"].notna()].copy()
    df["hit"] = (df["dir"] * df["r"] > 0).astype(float)
    df["base"] = np.where(df["dir"] > 0, df["up"], 1 - df["up"])
    df["liq"] = df["dir"] * df["r"] - df["cls"].map(S.CUSTO)
    df["dentro"] = [ (f[0] <= r <= f[1]) if isinstance(f, tuple) else np.nan for f, r in zip(df["fx"], df["r"])]
    sem = df.groupby("w").agg(n=("hit", "size"), acerto=("hit", "mean"), vento=("base", "mean"), liq=("liq", "mean"),
                              baixa=("dir", lambda d: float((d < 0).mean())), faixa=("dentro", "mean"))
    sem["habil"] = sem["acerto"] - sem["vento"]
    B.log(f"palpites: acerto médio {sem['acerto'].mean()*100:.1f}% | vento {sem['vento'].mean()*100:.1f}% | habilidade {sem['habil'].mean()*100:+.2f} p.p. t {tstat(sem['habil']):.2f}")

    # técnicas: acerto por semana quando a técnica deu sinal; habilidade = acerto − vento na direção dela
    tec = {}
    for nome in S.TECNICAS:
        v = df[df[f"v_{nome}"] != 0]
        if v.empty:
            continue
        h = (v[f"v_{nome}"] * v["r"] > 0).astype(float)
        b = np.where(v[f"v_{nome}"] > 0, v["up"], 1 - v["up"])
        g = pd.DataFrame({"w": v["w"], "h": h, "hb": h - b}).groupby("w").agg(h=("h", "mean"), hb=("hb", "mean"), n=("h", "size"))
        tec[nome] = (len(v), float(h.mean()), float((g["h"] > 0.5).mean()), float(g["hb"].mean()), tstat(g["hb"]))
    # situações
    cortes = df.groupby("cls")["sig"].quantile([1 / 3, 2 / 3]).unstack()
    clima = np.where(df["sig"] < df["cls"].map(cortes[1 / 3]), "calmo", np.where(df["sig"] > df["cls"].map(cortes[2 / 3]), "agitado", "normal"))
    sits = {"Palpite de alta": df["dir"] > 0, "Palpite de baixa": df["dir"] < 0, "Concordância ≤ 0": df["conc"] <= 0,
            "Concordância 1 a 2": df["conc"].between(1, 2), "Concordância 3 ou mais": df["conc"] >= 3,
            "Mercado calmo": clima == "calmo", "Mercado agitado": clima == "agitado",
            "ADX > 25": df["adx"] > 25, "ADX < 20": df["adx"] < 20, "RSI > 70": df["rsi"] > 70, "RSI < 30": df["rsi"] < 30,
            "Madrugada (0h–8h)": df["hora"] < 8, "Dia (8h–18h)": df["hora"].between(8, 17), "Noite (18h–24h)": df["hora"] >= 18}
    for cls, lab in S.MERCADO.items():
        sits[f"Mercado: {lab}"] = df["cls"] == cls
    sit = {}
    for nome, m in sits.items():
        v = df[m]
        if len(v) < 50:
            continue
        g = v.assign(hb=v["hit"] - v["base"]).groupby("w").agg(h=("hit", "mean"), hb=("hb", "mean"))
        sit[nome] = (len(v), float(v["hit"].mean()), float(v["base"].mean()), float(g["hb"].mean()), tstat(g["hb"]), float((g["hb"] > 0).mean()))

    # ---------- 2. operações com stop e alvo ----------
    for a in ativos:
        a["I"] = S.indicadores(a["df"]); a["A"] = R.atr(a["df"])
    combos = list(product(R.ENTRADAS, R.STOPS, R.SAIDAS))
    ops_c = {}
    for c in combos:
        ops = []
        for a in ativos:
            ops += R.operar(a, a["I"], a["A"], *c, ini, fim)
        ops_c[c] = ops
    B.log("operações com stop e alvo calculadas")
    res_c = {}
    for c, ops in ops_c.items():
        if not ops:
            continue
        o = pd.DataFrame(ops); o["w"] = o["t"].map(semana_de)
        gw = o.groupby("w")["R"].sum().reindex(range(W), fill_value=0.0)          # R somado na semana (1% de risco cada)
        m = R.metricas(ops)
        por_cls = {cls: R.metricas([x for x in ops if x["cls"] == cls]) for cls in S.MERCADO}
        res_c[c] = (m, float(gw.mean()), float((gw > 0).mean()), tstat(gw), por_cls, gw)
    melhor = max(res_c, key=lambda c: res_c[c][3]) if res_c else None

    # ---------- relatório ----------
    hoje = datetime.now(BRT)
    nome_c = lambda c: f"{c[0]} · stop {str(c[1]).replace('.', ',')} ATR · {R.SAIDAS[c[2]]}"
    L = ["---", "tipo: simulacao", "gerado_por: MF Tracker semanas.py", f"data: {hoje:%Y-%m-%dT%H:%M}-03:00",
         f"dados: {'SINTÉTICOS (teste do código)' if args.sintetico else 'reais'}", "tags: [simulacao, trading, 100-semanas]", "---", "",
         f"# {W} semanas simuladas, uma a uma — {hoje:%d/%m/%Y}", ""]
    if args.sintetico:
        L += ["> **Dados sintéticos** (passeio aleatório). O resultado certo é habilidade perto de zero e nenhum t acima de 2.", ""]
    L += [f"De **{datetime.fromtimestamp(ini, BRT):%d/%m/%Y}** a **{datetime.fromtimestamp(fim, BRT):%d/%m/%Y}**, {len(ativos)} ativos "
          f"({', '.join(sorted(set(S.MERCADO[a['cls']] for a in ativos)))}). Cada palpite e cada operação usaram só os dados até aquela hora. "
          "**Vento** = chance de acertar só pela tendência de longo prazo do ativo (fração de janelas que subiram nas 52 semanas anteriores, sem olhar a semana); **habilidade** = acerto − vento. "
          "Como as semanas são quase independentes, o **t entre semanas** é a medida honesta. Como muitas técnicas e situações são testadas ao mesmo tempo, "
          "algumas passariam de t 2 por sorte; por isso o limiar de confirmação é **t ≥ 2,9** (técnicas e combinações) e **t ≥ 3** (situações).", "",
          "## 1. Palpites (regra do guia: direção pela tendência diária, prazo de 4 candles)", "",
          f"- Palpites: **{len(df)}** em {W} semanas.",
          f"- Acerto médio por semana: **{B.pc(sem['acerto'].mean())}**; semanas com acerto acima de 50%: **{int((sem['acerto'] > 0.5).sum())} de {len(sem)}**.",
          f"- Vento médio: {B.pc(sem['vento'].mean())}. **Habilidade média: {sem['habil'].mean()*100:+.2f} p.p. por semana, t {tstat(sem['habil']):.2f}**; "
          f"semanas com habilidade positiva: {int((sem['habil'] > 0).sum())} de {len(sem)}.",
          f"- Resultado líquido médio por palpite (depois do custo): {B.pct(df['liq'].mean(), 3)}.",
          f"- Faixa de 50% prevista: caiu dentro em {B.pc(np.nanmean(df['dentro'].astype(float)))} dos casos (o certo é perto de 50%).", "",
          "### Semana a semana", "", "| Semana (início) | Palpites | Acerto | Vento | Habilidade | % de palpites de baixa | Faixa 50% |", "|---|---:|---:|---:|---:|---:|---:|"]
    for w, r in sem.iterrows():
        L.append(f"| {datetime.fromtimestamp(ini + w * SEM, BRT):%d/%m/%Y} | {int(r['n'])} | {B.pc(r['acerto'])} | {B.pc(r['vento'])} | {r['habil']*100:+.1f} p.p. | "
                 f"{B.pc(r['baixa'])} | {B.pc(r['faixa']) if np.isfinite(r['faixa']) else '—'} |")
    L += ["", "### Cada técnica do guia em 100 semanas", "",
          "| Técnica | Sinais | Acerto | Semanas acima de 50% | Habilidade média (acerto − vento) | t entre semanas | Leitura |", "|---|---:|---:|---:|---:|---:|---|"]
    for nome, (n, h, wpos, hb, t) in sorted(tec.items(), key=lambda x: -x[1][4]):
        leit = "ajuda (t ≥ 2,9)" if t >= 2.9 else ("atrapalha (t ≤ −2,9)" if t <= -2.9 else ("indício fraco" if abs(t) >= 2 else "sem efeito confirmado"))
        L.append(f"| {S.TECNICAS[nome]} | {n} | {B.pc(h)} | {B.pc(wpos)} | {hb*100:+.2f} p.p. | {t:.2f} | {leit} |")
    L += ["", "### Situações em que acerta ou erra mais", "",
          "| Situação | Palpites | Acerto | Vento | Habilidade média | t entre semanas | Semanas com habilidade positiva | Leitura |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for nome, (n, h, b, hb, t, wp) in sorted(sit.items(), key=lambda x: -x[1][4]):
        leit = "**acerta mais (t ≥ 3)**" if t >= 3 else ("**erra mais (t ≤ −3)**" if t <= -3 else ("indício fraco (pode ser sorte entre tantas situações)" if abs(t) >= 2 else "sem diferença confirmada"))
        L.append(f"| {nome} | {n} | {B.pc(h)} | {B.pc(b)} | {hb*100:+.2f} p.p. | {t:.2f} | {B.pc(wp)} | {leit} |")
    L += ["", "## 2. Operações com stop e alvo (gestão de risco do guia)", "",
          "Cada combinação foi operada nas 100 semanas, em todos os ativos, arriscando 1% da banca por operação. "
          "**Com 12 combinações, o limiar de confiança é t ≥ 2,9** (uma delas passaria de 2 por sorte com facilidade).", "",
          "| Combinação | Operações | Acerto | Ganho médio ÷ perda média | Fator de lucro | Resultado médio | Semanas positivas | R por semana (média) | t entre semanas |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c, (m, gm, wp, t, pc_, gw) in sorted(res_c.items(), key=lambda x: -x[1][3]):
        razao = m["ganho_medio"] / m["perda_media"] if m["perda_media"] > 0 else float("inf")
        L.append(f"| {nome_c(c)} | {m['n']} | {B.pc(m['acerto'])} | {razao:.2f} | {m['fl']:.2f} | {m['R']:+.3f} R | {B.pc(wp)} | {gm:+.2f} R | {t:.2f} |")
    if melhor:
        m, gm, wp, t, pc_, gw = res_c[melhor]
        L += ["", f"### A de maior t ({nome_c(melhor)}) por mercado", "", "| Mercado | " + R.CAB[0][2:].replace("Combinação | ", ""), "|---|" + R.CAB[1][5:]]
        for cls, mm in pc_.items():
            if mm:
                L.append(R.linha(S.MERCADO[cls], mm))
        banca, pico, queda = 1.0, 1.0, 0.0
        for x in gw:
            banca *= 1 + 0.01 * x; pico = max(pico, banca); queda = max(queda, 1 - banca / pico)
        L += ["", f"Banca simulada com 1% de risco por operação nessa combinação: **{B.pct(banca - 1, 1)}** em {W} semanas, maior queda **{B.pct(-queda, 1)}**. "
              + ("Passou do limiar de 2,9: vale acompanhar ao vivo." if t >= 2.9 else "Não passou do limiar de 2,9: mesmo sendo a melhor das 12, não dá para descartar sorte.")]
    # conclusão
    L += ["", "## Conclusão", "",
          f"- **Palpites de direção:** habilidade média {sem['habil'].mean()*100:+.2f} p.p. por semana (t {tstat(sem['habil']):.2f}). "
          + ("Há sinal de habilidade." if tstat(sem["habil"]) >= 2 else "Em 100 semanas, o palpite não acerta mais do que o vento do mercado."),
          "- **Técnicas que ajudam de forma confirmada:** " + (", ".join(S.TECNICAS[n] for n, x in tec.items() if x[4] >= 2.9) or "nenhuma") + ".",
          "- **Situações confirmadas:** " + (", ".join(f"{n} ({'acerta' if x[4] > 0 else 'erra'} mais)" for n, x in sit.items() if abs(x[4]) >= 3) or "nenhuma") + ".",
          "- **Stop e alvo:** " + (f"a melhor combinação ({nome_c(melhor)}) teve t {res_c[melhor][3]:.2f} entre semanas" if melhor else "sem operações") + ".", "",
          "## Limites (autocrítica)", "",
          "- Palpites conferidos em 4 candles; custo fixo por mercado; operações com stop e alvo em candles de 1 h (stop e alvo no mesmo candle contam como stop).",
          "- B3, EUA e dólar só têm cerca de 150 semanas de candles de 1 h no Yahoo; nas semanas mais antigas, as médias longas têm menos histórico.",
          "- Memecoins novas (PEPE, BONK, WIF) só entram a partir da listagem.", "", f"Tempo de execução: {time.time() - ini_exec:.0f} s.", ""]
    md = "\n".join(L)
    os.makedirs(B.SAIDA, exist_ok=True)
    open(os.path.join(B.SAIDA, "semanas.md"), "w", encoding="utf-8").write(md)
    if not args.sintetico:
        try:
            vault = json.load(open(os.path.join(B.BASE, "config.json"), encoding="utf-8"))["vault"]
            if os.path.isdir(os.path.join(vault, "01 Projeto")):
                nome = f"{W} semanas simuladas {hoje:%Y-%m-%d %Hh%M}.md"
                open(os.path.join(vault, "01 Projeto", nome), "w", encoding="utf-8").write(md)
                B.log(f"gravado no vault: 01 Projeto/{nome}")
        except Exception as e:
            B.log(f"não gravei no vault: {e}")
    B.log(f"pronto em {time.time() - ini_exec:.0f} s")


if __name__ == "__main__":
    main()
