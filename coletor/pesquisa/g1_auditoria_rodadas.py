"""Auditoria, parte 1: rodadas do modelo. Uso: python g1_auditoria_rodadas.py base|placebo|desloca|grupos
base    -> modelo 4 h e 24 h com embargo e funding defasado (previsões em todas as horas)
placebo -> rótulos embaralhados no mês e passeio aleatório com a mesma vol (4 h e 24 h)
desloca -> rótulos deslocados de 45 a 120 dias por ativo (3 sementes)
grupos  -> retreino tirando um grupo de variáveis de cada vez (4 h e 24 h)"""
import sys, json, time
import numpy as np, pandas as pd
import lib as L, ml_core as M

modo = sys.argv[1]
res = {}
for Hh in (4, 24):
    t0 = time.time()
    if modo == "base":
        P = M.painel(Hh)
        Q = M.walk_forward(P, Hh)
        Q.to_pickle(L.OUT + f"/g_pred_{Hh}h.pkl")
        res[f"{Hh}h"] = dict(auc=M.auc(Q.p, (Q.y > 0).astype(int)), n=len(Q))
    elif modo == "placebo":
        P = M.painel(Hh)
        Q = M.walk_forward(P, Hh, placebo="embaralha", seed=11)
        res[f"{Hh}h embaralhado"] = dict(auc=M.auc(Q.p, (Q.y > 0).astype(int)), n=len(Q))
        P2 = M.painel(Hh, placebo="passeio", seed=21)
        Q2 = M.walk_forward(P2, Hh)
        res[f"{Hh}h passeio aleatório"] = dict(auc=M.auc(Q2.p, (Q2.y > 0).astype(int)), n=len(Q2))
        for mg in (0.05, 0.10):
            for nome, QQ in (("embaralhado", Q), ("passeio aleatório", Q2)):
                R = M.operacoes(QQ, mg)
                t, _ = L.t_por_dia(R.bruto.values, R.index) if len(R) > 10 else (0.0, 0)
                res[f"{Hh}h {nome} margem {mg}"] = dict(ops=len(R), bruto=float(R.bruto.mean()) if len(R) else None, t_dia=t)
    elif modo == "desloca":
        P = M.painel(Hh)
        for sd in (31, 32, 33):
            Q = M.walk_forward(P, Hh, placebo="desloca", seed=sd)
            R = M.operacoes(Q, 0.05)
            t, _ = L.t_por_dia(R.bruto.values, R.index) if len(R) > 10 else (0.0, 0)
            res[f"{Hh}h rótulos deslocados (semente {sd})"] = dict(auc=M.auc(Q.p, (Q.y > 0).astype(int)), ops_m005=len(R), bruto_m005=float(R.bruto.mean()) if len(R) else None, t_dia=t)
    elif modo == "grupos":
        P = M.painel(Hh)
        base = M.walk_forward(P, Hh)
        a0 = M.auc(base.p, (base.y > 0).astype(int))
        res[f"{Hh}h completo"] = dict(auc=a0)
        for g, cols in M.GRUPOS.items():
            Q = M.walk_forward(P, Hh, tirar=cols)
            a = M.auc(Q.p, (Q.y > 0).astype(int))
            R = M.operacoes(Q, 0.05)
            res[f"{Hh}h sem {g}"] = dict(auc=a, queda_auc=a0 - a, bruto_m005=float(R.bruto.mean()), ops_m005=len(R))
            print(g, res[f"{Hh}h sem {g}"], flush=True)
    print(Hh, "h pronto em", round(time.time() - t0), "s", flush=True)
L.salva(f"g1_{modo}.json", res)
L.salva(f"g1_{modo}_n.json", {"n": L.N_TESTES})
print(json.dumps(res, indent=1, default=float))
