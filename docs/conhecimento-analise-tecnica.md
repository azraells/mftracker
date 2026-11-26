---
tipo: conhecimento
fonte: "John J. Murphy, Technical Analysis of the Financial Markets (NYIF, 1999), enviado pelo Luis em 26/11/2026"
data: 2026-11-26T08:10-03:00
tags: [conhecimento, analise-tecnica, trading]
---

# Base de conhecimento: análise técnica (guia de Murphy)

Resumo em palavras nossas do guia que o Luis enviou, organizado para uso no MF Tracker. Cada parte diz **o que o guia ensina**, **como o programa usa** e **o que os nossos testes mostraram** com dados reais. O guia é a referência clássica do assunto (usado na certificação CMT); os testes são o que decide se uma técnica entra como sinal.

## 1. As três premissas

1. **O preço já reflete tudo** que se sabe (notícias, fundamentos, psicologia).
2. **Preços andam em tendências**, e uma tendência tende a continuar até dar sinais claros de reversão.
3. **A história se repete**, porque a psicologia dos participantes se repete.

O guia também apresenta a crítica do passeio aleatório (mercado eficiente) e responde que tendências de prazo maior existem. **No MF Tracker:** a premissa 2 é a única com algum apoio nos nossos testes, e só em prazos longos (1 a 12 meses), onde seguir a tendência reduziu a maior queda em 20 de 24 ativos. Em prazos de horas a 2 semanas, nada indicou tendência explorável.

## 2. Teoria de Dow

- O mercado tem **três tendências**: primária (meses a anos), secundária (semanas a meses, correções de 1/3 a 2/3) e menor (dias).
- A tendência primária tem **três fases**: acumulação (os bem informados compram), participação pública (a alta é percebida) e distribuição (euforia, os primeiros vendem).
- **Os índices precisam se confirmar** entre si, e **o volume precisa confirmar** a tendência.
- **A tendência vale até haver sinal claro de reversão.**

**No MF Tracker:** é a base da regra "direção vem da tendência maior". A semana simulada usa a tendência diária como direção.

## 3. Tendência, suporte e resistência

- **Tendência de alta** é uma sequência de topos e fundos mais altos; **de baixa**, de topos e fundos mais baixos. Existe ainda a **lateral**, sem direção, em que métodos de tendência falham.
- **Suporte** é onde a compra segura a queda; **resistência**, onde a venda segura a alta. **Um suporte rompido vira resistência**, e vice-versa. Quanto mais tempo e volume um nível tiver, mais importante ele é. Números redondos funcionam como suporte e resistência psicológicos.
- **Linha de tendência:** são precisos dois pontos para traçá-la e um terceiro para confirmá-la. Linhas muito inclinadas são rompidas logo; a de 45° é a mais sustentável. Pelo princípio do leque, o rompimento da terceira linha confirma a reversão.
- **Canal:** paralela à linha de tendência. Falhar em chegar ao lado oposto do canal é um aviso de enfraquecimento.
- **Retrações** costumam ficar entre 1/3 e 2/3 do movimento, com 50% como a mais comum; 38% e 62% são as de Fibonacci.
- **Dias de reversão:** máxima nova seguida de fechamento abaixo do fechamento anterior, ou o inverso. Ganham peso com volume alto.
- **Gaps:** comum (sem importância), de rompimento (começa um movimento), de continuação (no meio do caminho, serve para medir o resto) e de exaustão (fim do movimento). A **ilha de reversão** é um gap de exaustão seguido de um gap de rompimento no sentido oposto.

**No MF Tracker:** o rompimento da máxima ou mínima de 20 candles e o rompimento do dia anterior (pivô) foram testados na semana simulada. Linhas de tendência e canais são subjetivos e ainda não foram testados.

## 4. Padrões de reversão

- **Pré-requisito:** precisa haver uma tendência anterior para reverter.
- **Ombro-cabeça-ombro (OCO):** três topos, o do meio mais alto. O volume cai a cada topo, e o padrão se confirma com o fechamento abaixo da linha do pescoço, muitas vezes seguido de um teste de volta a ela. O alvo é a altura da cabeça até o pescoço, projetada a partir do rompimento. Existe a versão invertida, para fundos.
- **Topo e fundo triplo ou duplo:** padrões parecidos com o OCO. O duplo só se confirma quando rompe o vale ou o pico do meio.
- **Pires** (virada lenta) e **V** (virada brusca, difícil de antecipar).
- **Regras gerais:**
  - quanto maior o padrão, maior o movimento depois;
  - topos são mais curtos e voláteis que fundos;
  - o volume pesa mais nas altas.

## 5. Padrões de continuação

- **Triângulos:**
  - **Simétrico:** em geral continua a tendência; o rompimento costuma vir entre 1/2 e 3/4 do caminho até o vértice.
  - **Ascendente:** viés de alta.
  - **Descendente:** viés de baixa.
  - **Alargamento:** volatilidade crescendo, comum em topos.
- **Bandeira e flâmula:** pausa curta (1 a 3 semanas) depois de um movimento quase em linha reta, com pouco volume. Aparecem no meio do movimento, o que serve para medir o resto.
- **Cunha:** inclinada contra a tendência. Na cunha de alta, a tendência que vem é de baixa.
- **Retângulo:** faixa lateral de consolidação. O volume indica para que lado vai romper.
- **Movimento medido:** a segunda perna repete o tamanho da primeira.

**No MF Tracker:** padrões gráficos dependem de reconhecimento visual. Para entrar no programa, cada um precisa virar regra objetiva e passar no teste com histórico dividido. Por ora são material de estudo.

## 6. Volume e contratos em aberto

- **Volume subindo** confirma a tendência; **volume caindo** avisa que ela pode mudar.
- **O volume costuma anteceder o preço.** O OBV (saldo de volume) mede a pressão compradora ou vendedora.
- **Contratos em aberto** (futuros):
  - subindo junto com uma tendência, a confirmam;
  - caindo, mostram que a tendência perde força;
  - muito altos no topo, tornam a queda mais violenta;
  - acumulados numa consolidação, fortalecem o rompimento.
- **Exaustão no topo e capitulação no fundo:** movimento brusco com volume enorme e virada rápida.
- **Sentimento:**
  - relatório COT: seguir os comerciais;
  - razão put/call;
  - pesquisas de opinião: quando quase todos estão otimistas, o mercado está perto do topo (opinião contrária).

**No MF Tracker:** o rompimento confirmado por volume foi testado na semana. O Medo e Ganância entra como psicologia. A taxa de financiamento, prima dos contratos em aberto, não se confirmou fora da amostra (26/11). Contratos em aberto e liquidações não têm histórico grátis.

## 7. Gráficos de longo prazo

Gráficos semanais e mensais dão a perspectiva: **a análise vai do longo para o curto prazo**. Sinais semanais pesam mais que os diários. O guia recomenda corrigir preços muito antigos pela inflação.

**No MF Tracker:** coerente com os testes, já que o único efeito útil (menos queda) apareceu em tendências de 200 dias e de 3 a 12 meses.

## 8. Médias móveis

- A média móvel é uma **tendência suavizada, com atraso**. A média curta reage rápido, mas dá falsos sinais; a longa é o oposto.
- **Usos:**
  - preço cruzando a média;
  - duas médias cruzando: 4/9, 9/18, 5/20 e 10/40 em futuros; 50 dias, 30 ou 40 semanas e 200 dias em ações;
  - envelopes percentuais.
- O guia diz que **não há prova de que a média exponencial seja melhor que a simples**.
- **Bandas de Bollinger:**
  - definição: média de 20 períodos ± 2 desvios;
  - uso como alvo: tocar a banda de cima numa faixa lateral é alvo de venda;
  - estreitamento das bandas: antecede um movimento forte.
- **Regra das 4 semanas:** comprar quando o preço passa a máxima das últimas 4 semanas e vender quando passa a mínima. É um sistema de rompimento de canal que funciona bem em mercados com tendência.
- **Otimização:** o guia alerta que parâmetros ajustados demais ao passado perdem valor.
- **Média adaptativa (Kaufman):** acelera quando o mercado tem tendência e desacelera quando está lateral.

**No MF Tracker:** a tendência usa as médias de 20, 50 e 100. Na semana simulada entram o cruzamento de 9 e 21 e o preço contra a média de 20 dias. O alerta de otimização é o motivo de escolhermos parâmetros numa metade do histórico e conferirmos na outra.

## 9. Osciladores e opinião contrária

- **Para que servem:** medem extremos (sobrecompra e sobrevenda) e divergências, isto é, quando o preço faz topo novo e o oscilador não confirma.
- **Momento e taxa de variação:** o cruzamento da linha zero dá sinal na direção da tendência.
- **RSI:**
  - acima de 70 indica sobrecompra e abaixo de 30, sobrevenda (80/20 em mercado forte);
  - a divergência e o "swing de falha" são os sinais mais fortes;
  - a linha de 50 funciona como divisor.
- **Estocástico %K/%D:** o sinal vale quando o %D sai das zonas de 80 ou de 20; a divergência é o sinal principal.
- **Williams %R e CCI:** variantes do mesmo princípio.
- **MACD:** diferença entre as médias de 12 e 26 e a linha de sinal de 9. O histograma vira antes do cruzamento. Funciona melhor no gráfico semanal como filtro e no diário para a entrada.
- **Regra de ouro do guia:**
  - os osciladores servem para escolher o momento de entrar **a favor da tendência maior**;
  - em alta, comprar o sobrevendido; em baixa, vender o sobrecomprado;
  - o guia diz explicitamente que é caro usar divergência para operar contra a tendência.
- **Quando os osciladores funcionam:** em faixas laterais, e em extremos quando a tendência está madura. Num mercado com tendência forte, ficam presos no extremo e dão sinais ruins.
- **Opinião contrária:** extremos de otimismo ou pessimismo na pesquisa de sentimento são sinais contra a maioria.

**No MF Tracker:** RSI, estocástico, MACD e Bollinger estão no motor desde a versão 0.1. Os testes de 26/11 (diário, 1 h e semana) não mostraram vantagem fora da amostra, nem na forma "a favor da tendência" que o guia recomenda (técnica `rsi_na_tendencia` na semana simulada).

## 10. Ponto e figura

Gráfico sem eixo de tempo: só registra movimentos acima de um tamanho mínimo (a caixa) e muda de coluna depois de uma reversão de 3 caixas. Os sinais vêm do rompimento de colunas anteriores. A **contagem horizontal** (largura da congestão) estima o alvo, e as linhas de 45° fazem o papel de linhas de tendência. O formato filtra o ruído e torna os sinais objetivos.

## 11. Candles japoneses

- **Anatomia:** corpo (abertura a fechamento) e sombras.
- **Padrões de reversão:**
  - martelo e enforcado;
  - engolfo de alta e de baixa;
  - nuvem negra e piercing;
  - estrelas da manhã e da noite;
  - doji;
  - harami.
- **Padrões de continuação:** três soldados brancos, três métodos.
- **Contexto:** um padrão só existe dentro de uma tendência de curto prazo.
- **Filtragem de Greg Morris:** só considerar padrões de reversão quando o oscilador (estocástico %D, RSI, %R) está na zona de extremo. Isso elimina muitos sinais falsos.

**No MF Tracker:** a semana simulada testa engolfo e martelo filtrados pelo estocástico (técnica `candle_filtrado`).

## 12. Ondas de Elliott e Fibonacci

- **Estrutura:**
  - 5 ondas a favor da tendência e 3 de correção (A-B-C);
  - a onda 3 nunca é a mais curta;
  - a onda 4 não entra no território da onda 1.
- **Regra da alternância:** se uma correção é simples, a seguinte tende a ser complexa.
- **Projeções e retrações** usam razões de Fibonacci: 0,382, 0,5, 0,618 e 1,618. O fundo da onda 4 costuma servir de suporte depois da onda 5.
- **Avaliação:** o guia reconhece que a contagem é subjetiva e funciona melhor em índices amplos.

**No MF Tracker:** subjetivo demais para virar regra testável agora. Fica como estudo.

## 13. Ciclos de tempo

Ciclos de tempo:
- **Princípios:** somatório, harmonia, sincronia e proporcionalidade.
- **Tendência e translação:** a tendência maior desloca o pico do ciclo para a direita (em alta) ou para a esquerda (em baixa).
- **Ciclos conhecidos:**
  - na bolsa americana, 4 anos e o ciclo presidencial;
  - o barômetro de janeiro;
  - ciclos sazonais em commodities.

**No MF Tracker:** sazonalidade por hora do dia e dia da semana entra como situação na semana simulada, sempre conferida no ano anterior.

## 14. Sistemas e computadores

- **Grupos de ferramentas:** indicadores de tendência para mercados com tendência e osciladores para mercados laterais.
- **Wilder:**
  - o Parabólico é um stop móvel;
  - o Movimento Direcional (+DI, −DI e ADX) mede se há tendência: ADX subindo indica tendência (usar sistema de tendência), ADX caindo de cima de 40 indica tendência perdendo força, e ADX voltando acima de 20 indica tendência nova;
  - o Parabólico deve ser filtrado pelo Movimento Direcional.
- **Prós e contras de sistemas:**
  - prós: disciplina, objetividade, nada de emoção;
  - contras: ajuste excessivo ao passado e perda de eficácia quando todos usam o mesmo.

**No MF Tracker:** o ADX decide na semana simulada se o palpite segue a tendência ou aposta na volta (técnica `adx_regime`).

## 15. Gestão de dinheiro e táticas (o capítulo mais importante para a aba Trading)

- **Três elementos do trading:**
  - **previsão de preço:** para que lado ir;
  - **momento:** quando entrar e sair;
  - **gestão do dinheiro:** quanto colocar.

  Acertar a direção e errar o momento ainda perde dinheiro.
- **Regras de alocação** (escritas para futuros, adaptáveis):
  - no máximo 50% do capital investido; o resto em renda fixa;
  - no máximo 10 a 15% numa posição;
  - **no máximo 5% de risco por operação**, ou seja, a perda até o stop;
  - no máximo 20 a 25% num grupo de mercados que andam juntos.
- **Relação ganho/risco de pelo menos 3 para 1.**
- **Táticas:**
  - **Pirâmide:** só aumentar posição que está ganhando, cada nova parte menor que a anterior, e subir o stop para o ponto de entrada.
  - **Depois de perdas,** reduzir o tamanho das posições.
  - **Stops:** colocar em níveis técnicos (abaixo de suporte, de mínima recente ou de média móvel), não em valores arbitrários.
- **Ordens:**
  - **A mercado:** garante a execução, mas pode sair pior.
  - **Limitada:** melhor preço, mas pode não ser executada.
  - **Stop:** limita a perda.
  - **Se tocar (MIT).**
- **Pivôs intradiários:** máxima, mínima e fechamento do dia anterior, mais a abertura do dia, como gatilhos. Os sinais ganham força quanto mais tarde no dia aparecem.
- **Resumo do guia, em 20 regras.** As centrais:
  - operar a favor da tendência intermediária;
  - em alta, comprar as quedas; em baixa, vender os repiques;
  - deixar o lucro correr e cortar a perda cedo;
  - usar stops;
  - ter um plano e segui-lo;
  - diversificar sem exagero;
  - nunca aumentar posição perdedora;
  - fechar as perdedoras antes das vencedoras;
  - ir do longo para o curto prazo;
  - desconfiar do consenso da mídia;
  - manter simples.

**No MF Tracker:**
- A aba Trading já mostra a faixa provável, o stop e o alvo, com o tamanho pelo critério de Kelly reduzido (no máximo 10% da banca).
- **Lição a aplicar:** o próprio exemplo de sistema do guia (apêndice C) acerta só 49% das operações, mas ganha 2,9 vezes mais nas vencedoras do que perde nas perdedoras. **Taxa de acerto não é tudo**: a relação ganho/risco e o corte rápido das perdas decidem o resultado. O MF Tracker precisa medir também o fator de lucro (ganho bruto ÷ perda bruta) e a maior queda, não só o acerto.

## 16. Análise intermercados

- **Elos clássicos:**
  - dólar contra commodities (inversos);
  - commodities contra juros (andam juntos, por causa da inflação);
  - juros contra ações (inversos: títulos sobem antes das ações).
- **Rotação de setores** segue o ciclo econômico.
- **Força relativa** é a razão entre um ativo e o índice: comprar os mais fortes de setores fortes.

**No MF Tracker:** o momento relativo na B3 deu +33%/ano contra 18,5% da carteira igual, mas foi inconclusivo (t 1,43). Vale aprofundar quando houver mais ações no teste.

## 17. Indicadores de amplitude da bolsa

- **Linha de avanços e declínios:** divergência contra o índice avisa de um topo.
- **Oscilador e índice de soma de McClellan.**
- **Novas máximas contra novas mínimas.**
- **Volume de alta contra volume de baixa.**
- **Índice Arms (TRIN).**
- **Equivolume.**

**No MF Tracker:** precisam de dados de todas as ações da bolsa, que não temos de graça. Ficam para o futuro.

## 18. Apêndices

- **A, indicadores avançados:**
  - Demand Index e Herrick Payoff Index: preço, volume e contratos em aberto;
  - bandas STARC e canais de Keltner: faixas baseadas na média de amplitude (ATR).
- **B, Market Profile:** distribuição do tempo gasto em cada preço; a área de valor tem cerca de 70% da atividade.
- **C, como construir um sistema**, que é exatamente o método que o MF Tracker já segue:
  1. começar de uma ideia;
  2. transformar a ideia em regras objetivas;
  3. olhar nos gráficos;
  4. testar no computador;
  5. avaliar.

  Na avaliação entram o fator de lucro (alvo de pelo menos 2 para longo prazo), o resultado médio por operação (precisa cobrir custos e derrapagem) e a maior queda. O apêndice termina com a ideia de diversificar entre mercados, sistemas, parâmetros e prazos.
- **D, contratos contínuos de futuros:** como emendar vencimentos.

## 19. O checklist do guia (para cada análise)

1. Para onde vai o mercado como um todo e os setores?
2. O que dizem os gráficos semanal e mensal?
3. As tendências maior, intermediária e menor estão para cima, para baixo ou de lado?
4. Onde estão os suportes, as resistências, as linhas de tendência e os canais?
5. O volume confirma o movimento?
6. Onde estão as retrações de 33%, 50% e 66%?
7. Há gaps, padrões de reversão ou de continuação? Quais os alvos?
8. Para onde apontam as médias móveis?
9. Os osciladores estão em extremo? Há divergência?
10. O sentimento está em extremo (opinião contrária)?
11. Elliott, Fibonacci e ciclos mostram algo?
12. Depois de decidir a direção:
    - comprar ou vender?
    - quanto?
    - **quanto arrisco se estiver errado?**
    - qual o alvo?
    - onde entro, com que ordem?
    - **onde fica o stop?**

## O que muda no MF Tracker com este guia

1. **A direção vem da tendência maior; os osciladores só escolhem o momento.** Já aplicado na semana simulada (`coletor/semana.py`).
2. **Medir mais do que o acerto:** fator de lucro, ganho médio contra perda média e maior queda, usando stop e alvo com relação 3:1. É o próximo teste a fazer.
3. **Risco por operação de no máximo 5% da banca,** com stop técnico. A aba Trading deve sugerir o tamanho pela distância até o stop.
4. **Do longo para o curto:** filtro semanal ou diário para sinais horários (como o guia recomenda para MACD e estocástico).
5. **Desconfiar da otimização:** toda regra é escolhida numa metade do histórico e conferida na outra, como o próprio guia alerta.

## Limites honestos

- O guia é de 1999 e foi escrito sobretudo para futuros e ações americanas. Cripto 24 h por dia e memecoins não existiam.
- O guia **afirma** que as técnicas funcionam, mas mostra exemplos escolhidos, não testes estatísticos. Os nossos testes com dados de 2012 a 2026 não confirmaram vantagem dos osciladores e médias em prazos curtos depois dos custos. Isso é comum na pesquisa acadêmica com ativos líquidos: técnicas muito conhecidas tendem a perder o efeito.
- O valor mais sólido do guia, pelos nossos testes, está em **gestão de risco** (stop, tamanho, ganho/risco, seguir tendência longa para evitar grandes quedas), não em prever a direção de curto prazo.

Relacionado: [[Backtest 2026-11-26]], [[Calibração de probabilidades]]
