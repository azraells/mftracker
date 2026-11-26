---
tipo: log
atualizado: 2026-11-26
tags: [projeto, changelog]
---
# Changelog

Mais recente no topo.

## 2026-11-26 — programa v0.2.0
- Oito bancas simuladas em paralelo (R$ 50, 100, 300, 500, 1.000, 2.500, 5.000, 10.000), com valor mínimo por ordem, custos fixos e deslizamento em DEX. Placar automático mostra cada banca.
- Radar de memecoins novas (GeckoTerminal + DexScreener) com nota de atenção, nota de risco e alertas; guarda os 10 primeiros a cada 6 h e confere o preço em 24 h e 7 dias.
- Moedas de DEX acompanhadas em candles por hora, prazos de 6, 24 e 72 h. DUH (Solana, contrato ngwxpGYdkauMcxLFrNTKp8qBgK7JV53zVvd7F2spump) entra por padrão.
- Novas notas automáticas: `02 Mercados/Cotações ao vivo.md` (a cada 15 min) e `02 Mercados/Radar de memecoins (automático).md`.

## 2026-11-26 — programa v0.1.0
- Primeira versão funcional do programa: `Documentos\MF Tracker Programa\MF-Tracker.html` (ver [[Arquitetura do programa]]).
- 16 ativos: BTC, ETH, SOL, XRP; DOGE, SHIB, PEPE, BONK, WIF; PETR4, VALE3, ITUB4, MGLU3; AAPL, QQQ; USD/BRL.
- 9 fatores; chance calculada por casos semelhantes do último ano, comparada com a taxa base.
- Registro automático de toda oportunidade, avaliação automática no vencimento, ajuste de pesos e de calibração.
- Gravação automática no vault: `04 Operações/OP-*.md`, `05 Análises/AN-OP-*.md`, `04 Operações/Placar automático.md` e backup JSON em `99 Meta/Backup do programa/`.

## 2026-11-26
- Luis define o MF Tracker (ver [[Visão do projeto]]).
- Fontes de dados testadas no navegador do Luis (ver [[Fontes de dados]]).
- Mercados escolhidos (ver [[Decisões]]).
- Início da construção da primeira versão do programa.
- Criado este vault do Obsidian, já preenchido com tudo o que se sabia.
