# Pesquisa autônoma (2026-11-26)

Scripts da rodada de 4.328 simulações feita no PC do Luis. Relatório completo no vault:
`01 Projeto/Pesquisa autônoma 2026-11-26.md`.

Leem o cache do coletor (`coletor/dados/historico`, ou a pasta em `MF_HIST`) e gravam em `pesquisa/saida/`
(fora do Git). Só numpy e pandas.

Ordem: `a_tempo` → `b_calendario` → `c_relacoes` → `d_funding_carry` → `e_extremos_ml` → `f_saidas_mc`
(usa `a_tempo.csv` e `d_funding_carteira.csv`) → `h_bonferroni`.
Diagnósticos: `b1_confere` (noite × pregão no 1 h), `e3_ml_custos`, `e4_ml_robustez` (superado pela auditoria).

Auditoria do modelo (nota `01 Projeto/Auditoria do modelo 2026-11-26.md`): `ml_core` (núcleo com embargo e placebos)
→ `g1_auditoria_rodadas base|placebo|desloca|grupos` → `g2_auditoria_analise` → `g3_mc`.
Custo baixo e entrada no regime (nota `01 Projeto/Custo baixo e entrada no regime 2026-11-26.md`): `i_custo_regime` (depois do `g1 base`).

Mini índice (WIN), aproximado pelo Ibovespa e pelo BOVA11 em candles de 1 h: `j_mini_indice` (baixa o ^BVSP do Yahoo na primeira vez; precisa de `yfinance`). O teste de verdade precisa do histórico M1/M5 do WIN exportado do MetaTrader 5.

## Revisão do código (autoanálise)

- `e_extremos_ml`: o treino só usa linhas cujo alvo terminou antes do mês de teste (embargo mensal);
  média e desvio da normalização vêm só do treino; o funding entra com ffill pela hora de liquidação,
  que já é conhecida. Não achei vazamento.
- O t do modelo agrupa por dia. Em 4 h isso basta; em 24 h as operações de dias vizinhos se sobrepõem,
  então o t de 24 h está otimista.
- `painel.iloc[::4]` amostra uma linha a cada 4 do painel empilhado (todos os ativos juntos), não uma a
  cada 4 h por ativo. Em média dá o mesmo espaçamento, mas não é exatamente "sem sobreposição".
- `a_tempo` divide a história de cada ativo na própria metade. Na carteira de `f_saidas_mc` o corte é o
  maior dos meios, então tudo o que serviu para escolher fica antes do teste.
- O Monte Carlo sorteia semanas soltas da 2ª metade: perde a sequência de semanas ruins (as quedas reais
  vêm em bloco) e depende muito do período 2023–2026. A chance de perder 50% está subestimada.
- O corte de confiança 0,10 do modelo foi escolhido olhando o resultado. A auditoria confirmou: escolhido só em
  2022–23, ele perde em 2024–26.

Vendas a descoberto (nota `01 Projeto/Vendas a descoberto 2026-11-26.md`): `k_vendas` (lado vendido do modelo de 4 h e regime ao contrário no perpétuo). Esse estudo achou que `lib.t_por_dia` (média por dia) superestimava o t do modelo de 4 h: pela soma do dia (`lib.t_por_dia_soma`) o t é 3,0, não 8,1, e o sinal não passa Bonferroni.

Horizontes de 12 a 72 h, compra e venda (nota `01 Projeto/Horizontes 12 a 72 h 2026-11-26.md`): `l_horizontes` (modelo, tendência, reversão e modelo a favor do regime, nos dois lados) e `l_placebo_diario` (placebo do modelo diário com rótulos deslocados para a frente, além da janela de 250 pregões).
