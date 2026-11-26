---
tipo: projeto
atualizado: 2026-11-26
status: v0.1.0 funcional, em teste
tags: [projeto, arquitetura]
---
# Arquitetura do programa

Construído na thread "Construir o MF Tracker". Atualizar a cada versão (ver [[Changelog]]).

## Onde fica e como rodar
- Programa: `C:\Users\luisf\Documents\MF Tracker Programa\MF-Tracker.html` (cópia de trabalho do Claude em `/mnt/project-files/mf-tracker/MF-Tracker.html`).
- Rodar: dar dois cliques no arquivo e abrir no **Chrome ou Edge**. Precisa de internet. Deixar a aba aberta para acompanhar em tempo real; se fechar, as previsões vencidas são avaliadas na próxima abertura, com o preço histórico do dia do vencimento.
- Não há servidor nem instalação. Um único arquivo com HTML, CSS e JavaScript. Gráficos: TradingView Lightweight Charts 4.1.3 (unpkg).
- Ainda **sem controle de versão (git)**: só existe a cópia no computador e na pasta do projeto.

## Onde os dados ficam
- Diário, pesos aprendidos e configurações: `localStorage` do navegador (chave `mftracker.v1`). Limpar os dados do navegador apaga tudo.
- Proteção: com o vault conectado (aba Oportunidades → Conectar vault do Obsidian → escolher `Documentos\MF Tracker`), o programa grava sozinho:
  - `04 Operações/OP-AAAAMMDD-HHMM-TICKER.md` ao abrir cada operação (frontmatter do [[Modelo — Operação simulada]]);
  - o mesmo arquivo com o frontmatter de encerramento e `05 Análises/AN-OP-….md` no vencimento;
  - `04 Operações/Placar automático.md` (placar e pesos dos fatores);
  - `99 Meta/Backup do programa/mf-tracker-dados.json` (backup completo, pode ser importado em Configurações).
- O Chrome/Edge pode pedir para reautorizar a pasta ao reabrir o arquivo: aparece o botão "Reautorizar vault".

## Fontes de dados (ver [[Fontes de dados]])
| Mercado | Fonte | Atualização |
|---|---|---|
| Cripto e memecoins | Coinbase Exchange (candles diários + ticker) | preço a cada 20 s |
| B3 | brapi.dev (sem token: PETR4, VALE3, ITUB4, MGLU3) | a cada 5 min, pode ter atraso |
| EUA | Twelve Data (sem chave: AAPL e QQQ; limite 8 chamadas/min) | ao abrir e a cada 30 min |
| Dólar | AwesomeAPI | a cada 5 min |
| Contexto | alternative.me (Medo & Ganância), CoinGecko (mercado cripto), BCB (Selic) | a cada 5 min |
| Notícias | rss2json lendo InfoMoney, Money Times, CoinDesk, Cointelegraph | a cada 15 min |

## Motor de análise (ver [[Metodologia de previsão]])
1. **9 fatores** de −1 a +1: tendência (médias 20/50/100), momento (10 dias ÷ volatilidade), exaustão (RSI 14), desvio da média (Bollinger 20,2), fluxo de volume, psicologia (Medo & Ganância em cripto; sequências e distância da máxima nos demais), nicho (momento dos pares da mesma classe), notícias (sentimento por palavras-chave, 72 h, peso decai em 24 h), contexto global.
2. **Pontuação técnica** = média ponderada pelos pesos aprendidos dos 7 fatores que dá para recalcular no passado.
3. **Chance**: para os prazos de 3, 7 e 14 dias (pregões em B3/EUA/câmbio), procura no último ano os dias com pontuação parecida e na mesma direção (pelo menos 12, idealmente 20 casos). Chance = (acertos + 1) / (casos + 2), já descontado o custo da classe (cripto 0,3%, memecoins 0,6%, B3/EUA 0,1%, câmbio 0,5%). Taxa base = mesma conta sem filtro.
4. Ajustes não calibrados: notícias até ±4 p.p., contexto global até ±3 p.p. Correção de calibração pelo próprio histórico após 10 previsões encerradas (até ±15 p.p.).
5. **Prazo escolhido**: maior retorno médio ÷ √prazo.
6. **Oportunidade** se: chance ≥ 55%, chance − taxa base ≥ 2 p.p., retorno médio > 0, |pontuação| ≥ 0,12. Limites editáveis em Configurações.
7. **Valor**: 1/4 de Kelly, teto de 10% da banca virtual (R$ 10.000). Alvo = ganho mediano dos casos vencedores. Stop = maior entre 1,2 × perda média e 1,5 × ATR.

## Avaliação no vencimento
- Preço de saída: fechamento do candle diário do dia do vencimento (ou preço ao vivo, se o vencimento é hoje).
- Calcula retorno líquido, melhor e pior momento, alvo e stop tocados, e qual fator apontou certo.
- **Acerto** = retorno líquido > 0 na direção prevista (ver [[Decisões]]).
- Aprendizado: peso de cada fator × exp(0,08 × leitura × direção real), limitado entre 0,25 e 3.

## Limites conhecidos (autoanálise)
- Notícias e contexto global sem calibração histórica; sentimento por palavras-chave erra ironia e contexto.
- Só um ano de candles diários: as chances têm margem de erro de vários pontos.
- Fatores técnicos correlacionados entre si.
- Sem intradiário: o tracker não vê movimentos dentro do dia (só máxima e mínima).
- Custos estimados; sem impostos nem variação cambial para ativos em dólar.
- Dados no navegador: depende do vault/backup para não perder nada.
