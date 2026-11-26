# MF Tracker

Programa de acompanhamento e estudo do mercado financeiro **em simulação** (sem dinheiro real até provar efetividade).
Identifica oportunidades, registra cada uma com valor simulado em 8 bancas (R$ 50 a R$ 10.000), retorno possível, prazo e chance,
e no vencimento avalia o acerto, ajusta os pesos dos fatores e a calibração.

## Partes
| Arquivo | O que é |
|---|---|
| `MF-Tracker.html` | O programa (painel, motor de análise, diário, radar de memecoins). Abrir no Chrome ou Edge. |
| `chaves.local.js` | Suas chaves de API (brapi, Twelve Data). **Não vai para o Git.** Modelo em `chaves.local.exemplo.js`. |
| `coletor/` | Coletor em Python que grava cotações ao vivo no vault do Obsidian e um histórico bruto em CSV. Rodar `iniciar_coletor.bat`. |

## Dados
- Diário do programa: no navegador (localStorage) e, com o vault conectado, em notas do Obsidian (`04 Operações`, `05 Análises`) e no backup `99 Meta/Backup do programa/mf-tracker-dados.json`.
- Coletor: `coletor/dados/snapshots/AAAA-MM-DD.csv` (fora do Git) e `02 Mercados/Cotações ao vivo.md` no vault.

## Fontes (todas gratuitas)
Coinbase, Binance (ccxt), DexScreener, GeckoTerminal, Yahoo Finance (yfinance), brapi.dev, Twelve Data, AwesomeAPI, alternative.me, CoinGecko, Banco Central.

## Aviso
Previsões são probabilidades estimadas a partir do passado, não garantias. Nada aqui é recomendação de investimento.

## Documentação

A pasta `docs/` guarda cópias da arquitetura, do changelog e das decisões. A versão viva fica no vault do Obsidian, em `01 Projeto/`.

## Chaves de API

Copie `chaves.local.exemplo.js` para `chaves.local.js` (ao lado do `MF-Tracker.html`) e preencha os tokens. Esse arquivo está no `.gitignore` e nunca deve ser enviado ao repositório.

## Backtest

`coletor\.venv\Scripts\python coletor\backtest.py` baixa anos de histórico (Binance via ccxt, Yahoo via yfinance) e testa o método do jeito que ele roda ao vivo, sem olhar o futuro. O relatório vai para `coletor/dados/backtest/relatorio.md` e para o vault em `01 Projeto/Backtest AAAA-MM-DD.md`. Com `--sintetico` roda em preços aleatórios para testar o código (o resultado certo é "sem vantagem").
