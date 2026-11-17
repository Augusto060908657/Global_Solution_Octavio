# Relatório de resultados (gerado automaticamente)

Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696  
Pesos: `best.pt` · resolução de inferência: **640 px** · dados: `data/yolo_mix`

## Resumo dos principais resultados

- **mAP@50 = 0.539** e **mAP@50-95 = 0.291** no teste (todas as classes); Precision 0.604, Recall 0.507, F1 0.552.
- Por classe (mAP@50): Buraco 0.678; Rachadura 0.401.
- Como triagem por imagem (há buraco? há rachadura?): buraco P 0.91 / R 0.87; rachadura P 0.96 / R 0.86.
- Latência/FPS: CPU: Intel(R) Xeon(R) CPU @ 2.00GHz 89.0 ms (11.2 FPS) @ 640 px; GPU: Tesla T4 10.4 ms (96.5 FPS) @ 640 px.

## 1. Qualidade da detecção no TESTE (Precision, Recall, F1, mAP)
P/R no ponto de máximo F1 e mAP (IoU 0.5 e 0.5:0.95) calculados pelo Ultralytics; F1 = 2PR/(P+R).

| classe | precision | recall | f1 | mAP50 | mAP50_95 |
|---|---|---|---|---|---|
| Buraco | 0.6246 | 0.6765 | 0.6495 | 0.6777 | 0.4056 |
| Rachadura | 0.5840 | 0.3385 | 0.4286 | 0.4007 | 0.1756 |
| Todas | 0.6043 | 0.5075 | 0.5517 | 0.5392 | 0.2906 |

### Por fonte de dados (dataset combinado)

| fonte | imagens | mAP50 | mAP50_95 | precision | recall | f1 | recall_buraco_por_imagem | recall_rachadura_por_imagem |
|---|---|---|---|---|---|---|---|---|
| br | 335 | 0.3918 | 0.1843 | 0.4850 | 0.4363 | 0.4594 | 0.6905 | 0.8990 |
| rdd2022 | 311 | 0.3427 | 0.1498 | 0.4394 | 0.3511 | 0.3903 | 0.6316 | 0.7991 |
| rf_rua | 238 | 0.8336 | 0.5270 | 0.8521 | 0.7287 | 0.7856 | 0.9824 | 0.0000 |

### No limiar operacional (conf = 0.175, IoU ≥ 0.5)

| classe | TP | FP | FN | precision | recall | f1 |
|---|---|---|---|---|---|---|
| Buraco | 442 | 331 | 195 | 0.5718 | 0.6939 | 0.6270 |
| Rachadura | 511 | 493 | 891 | 0.5090 | 0.3645 | 0.4248 |

Presença por imagem (a imagem tem a classe? — métrica de triagem):

| classe | TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|---|
| Buraco | 305 | 31 | 44 | 504 | 0.9077 | 0.8739 |
| Rachadura | 433 | 19 | 73 | 359 | 0.9580 | 0.8557 |

Matriz de confusão: `reports/confusion_matrix.png` · curvas de treino: `reports/training_curves.png` · curva F1 × limiar (validação): `reports/f1_vs_conf.png`.

## 2. Latência e FPS (equipamento e resolução informados)

| equipamento | resolução (px) | medida | latência média (ms/imagem) | p95 (ms) | FPS |
|---|---|---|---|---|---|
| CPU: Intel(R) Xeon(R) CPU @ 2.00GHz | 640 | ponta a ponta (pré-proc + modelo + NMS) | 89.0 | 112.6 | 11.2 |
| CPU: Intel(R) Xeon(R) CPU @ 2.00GHz | 640 | forward puro | 87.0 | 99.8 | 11.5 |
| GPU: Tesla T4 | 640 | ponta a ponta (pré-proc + modelo + NMS) | 10.4 | 12.6 | 96.5 |
| GPU: Tesla T4 | 640 | forward puro | 10.4 | 12.3 | 96.1 |

10 iterações de warmup + 100 medidas com `time.perf_counter()`; imagem de teste 1024×640, tensor de entrada [1, 3, 416, 640].

## 3. Justificativa do limiar de confiança

O limiar operacional **0.175** foi escolhido **na validação** como o valor que maximiza o F1 médio das classes (IoU ≥ 0.5), sem usar o teste (curva em `reports/f1_vs_conf.png`). Valores menores aumentam o recall mas geram mais falsos positivos (mais vistorias desnecessárias); valores maiores perdem danos reais. No teste, com esse limiar: Buraco P=0.57, R=0.69, F1=0.63; Rachadura P=0.51, R=0.36, F1=0.42.

Efeito do limiar na velocidade (CPU: Intel(R) Xeon(R) CPU @ 2.00GHz, 640 px): limiares mais baixos mantêm mais caixas candidatas na supressão não máxima (NMS) e podem aumentar o tempo de pós-processamento.

| limiar | latência média (ms) | FPS |
|---|---|---|
| 0.0500 | 105.1 | 9.5149 |
| 0.1000 | 96.5 | 10.4 |
| 0.2500 | 74.5 | 13.4 |
| 0.5000 | 71.5 | 14.0 |

Cautela: a tabela usa **uma** imagem de teste e poucas repetições. No mesmo limiar (0,25), a medição ponta a ponta da seção 2 deu 89.0 ms e esta tabela 74.5 ms; essa variação entre medições é da ordem do efeito do limiar, que portanto deve ser lido como tendência, não como valor exato.

Efeito do limiar na velocidade (GPU: Tesla T4, 640 px): limiares mais baixos mantêm mais caixas candidatas na supressão não máxima (NMS) e podem aumentar o tempo de pós-processamento.

| limiar | latência média (ms) | FPS |
|---|---|---|
| 0.0500 | 10.7 | 93.1 |
| 0.1000 | 11.5 | 86.6 |
| 0.2500 | 10.5 | 95.4 |
| 0.5000 | 9.5757 | 104.4 |

Cautela: a tabela usa **uma** imagem de teste e poucas repetições. No mesmo limiar (0,25), a medição ponta a ponta da seção 2 deu 10.4 ms e esta tabela 10.5 ms; essa variação entre medições é da ordem do efeito do limiar, que portanto deve ser lido como tendência, não como valor exato.


A decisão equilibra qualidade e velocidade: o tempo por imagem é dominado pela inferência da rede (ver tabela da seção 2), de modo que escolher o limiar pelo F1 de validação custa pouco em latência.

## 4. Análise de erros (acertos e falhas)

Galeria **real (verde) × previsto** em `reports/error_analysis/` (prefixo `acerto_` = sem erros; `erro_` = piores casos). Abaixo, a influência medida de iluminação, sombra, textura e tamanho do objeto (por tercil das imagens de teste; recall = objetos detectados com a classe certa / objetos reais, IoU ≥ 0.5):

**Iluminação (brilho médio)**

| faixa | imagens | objetos | acertos | fp | faixa_min | faixa_max | recall | fp_por_imagem |
|---|---|---|---|---|---|---|---|---|
| baixo | 295 | 817 | 360 | 355 | 24.5 | 120.7 | 0.4406 | 1.2034 |
| médio | 294 | 635 | 281 | 230 | 120.9 | 143.8 | 0.4425 | 0.7823 |
| alto | 295 | 587 | 312 | 239 | 143.8 | 214.0 | 0.5315 | 0.8102 |

**Sombra/subexposição (fração de pixels escuros)**

| faixa | imagens | objetos | acertos | fp | faixa_min | faixa_max | recall | fp_por_imagem |
|---|---|---|---|---|---|---|---|---|
| baixo | 295 | 765 | 325 | 286 | 0.0000 | 0.0064 | 0.4248 | 0.9695 |
| médio | 294 | 772 | 371 | 314 | 0.0065 | 0.0550 | 0.4806 | 1.0680 |
| alto | 295 | 502 | 257 | 224 | 0.0555 | 0.9272 | 0.5120 | 0.7593 |

**Textura (densidade de bordas)**

| faixa | imagens | objetos | acertos | fp | faixa_min | faixa_max | recall | fp_por_imagem |
|---|---|---|---|---|---|---|---|---|
| baixo | 295 | 941 | 357 | 338 | 0.0039 | 0.0591 | 0.3794 | 1.1458 |
| médio | 294 | 621 | 333 | 264 | 0.0592 | 0.1225 | 0.5362 | 0.8980 |
| alto | 295 | 477 | 263 | 222 | 0.1227 | 0.3430 | 0.5514 | 0.7525 |

**Contraste**

| faixa | imagens | objetos | acertos | fp | faixa_min | faixa_max | recall | fp_por_imagem |
|---|---|---|---|---|---|---|---|---|
| baixo | 295 | 786 | 320 | 329 | 11.6 | 27.9 | 0.4071 | 1.1153 |
| médio | 294 | 818 | 395 | 309 | 27.9 | 45.4 | 0.4829 | 1.0510 |
| alto | 295 | 435 | 238 | 186 | 45.4 | 97.2 | 0.5471 | 0.6305 |

**Tamanho do objeto (área relativa)**

| faixa | objetos | recall | area_min | area_max |
|---|---|---|---|---|
| pequeno | 680 | 0.2618 | 0.0001 | 0.0045 |
| médio | 679 | 0.4065 | 0.0045 | 0.0332 |
| grande | 680 | 0.7338 | 0.0333 | 0.9523 |

### Observações visuais (feitas sobre as imagens geradas nesta execução)

Fontes: `reports/error_analysis/` (real em verde × previsto), `reports/demo/teste_*.jpg`, `reports/demo/externas_*.jpg`, galeria das fotos do autor e a tabela `error_factors` acima. Onde a afirmação vem só da leitura visual de poucos exemplos, está indicado.

**Onde o modelo acerta**
- **Buracos em fotos de rua de outra origem** (`externas_buracos.jpg`): detectou buraco com água (ext_048 e ext_035, 72%), buraco com entulho ao lado de cone (ext_036, 85%), buracos em estrada de terra sob sombra de árvores (ext_015 82%, ext_007 88%/81%, ext_044 87%) e vários buracos pequenos com água em uma rua asfaltada (ext_061, 8 caixas). Nesses exemplos a **sombra projetada** e a **água** não impediram a detecção, e o cone não foi marcado como defeito.
- **Fissuras em close** (CrackForest, `externas_rachaduras.jpg`): 107 de 118 imagens tiveram ao menos uma rachadura detectada. As caixas são largas e as confianças baixas (18% a 60% nos exemplos do mosaico), o que explica o limiar baixo escolhido na validação (0,175).
- **Rodovias brasileiras** (`teste_br.jpg`): remendos craquelados e áreas em "couro de jacaré" detectados com 66% a 69% de confiança.
- **Pavimento íntegro:** nas 4 fotos de pavimento sem danos (pavimento novo, rua molhada com tampa de bueiro, rua arborizada) a execução no Colab não gerou detecção. Amostra pequena (ver limitações).
- **Fotos do autor** (interface Gradio): cratera com bueiro e entulho marcada como buraco (49%) e poça grande com motociclista marcada como buraco (78%); veículos e o motociclista não foram marcados.

**Onde o modelo erra, e por quê**
1. **Granularidade das caixas (principal causa dos "erros" numéricos).** No dataset brasileiro, cada trecho de fissura foi anotado como várias caixas pequenas (conversão das máscaras), e o modelo prevê caixas maiores que agrupam o trecho. Com IoU ≥ 0,5 isso conta como falso negativo **e** falso positivo ao mesmo tempo (ex.: `erro_10err_br_741369_…`: 10 erros em uma imagem em que as regiões rachadas foram localizadas). Em números, no limiar operacional a classe rachadura tem 891 FN e 493 FP por caixa, mas, como triagem por imagem, precisão 0,96 e recall 0,86: **o modelo acha onde há rachadura, mas não a segmenta como foi anotada**.
2. **Objetos pequenos.** Recall por tamanho do objeto (terços da área relativa): 0,26 (pequeno), 0,41 (médio), 0,73 (grande). Buracos e fissuras distantes, típicos de foto de rodovia, são o caso mais difícil.
3. **Textura lisa e baixo contraste.** Recall por terço de densidade de bordas: 0,38 (menor) contra 0,54 e 0,55; por contraste: 0,41 (menor) contra 0,48 e 0,55. Fissuras finas e pouco contrastadas em asfalto liso são mais difíceis. A correlação não prova causa: os terços misturam fontes de dados.
4. **Iluminação.** Recall por brilho médio: 0,44 (baixo), 0,44 (médio), 0,53 (alto); falsos positivos por imagem: 1,20 (baixo) contra 0,78 e 0,81. Imagens escuras tendem a ter mais alarmes falsos.
5. **Sombras:** com o indicador usado (fração de pixels escuros), **não encontramos piora do recall** em imagens com mais área escura (0,42 → 0,48 → 0,51). O indicador é grosseiro; a conclusão vale só para este conjunto.
6. **Marcações viárias e juntas (apenas leitura visual).** Em algumas imagens do RDD2022 há previsões sobre a faixa dupla amarela e juntas do pavimento (ex.: `United_States_003468`: 1 real, 3 previstos). Não medimos isso separadamente.
7. **Caixas duplicadas/sobrepostas.** Visíveis nos mosaicos (ex.: várias caixas sobre a mesma fissura): a supressão não máxima com IoU 0,6 mantém caixas parcialmente sobrepostas, o que aumenta falsos positivos por caixa.
8. **Misses nas imagens externas:** `buraco_ext_013` e 11 imagens do CrackForest (023, 040, 042, 051–054, 056, 080, 086, 087) sem detecção. Não inspecionamos cada uma; a hipótese é fissura muito fina ou de baixo contraste.

**Limitações da leitura**
- **Falso alarme em pavimento íntegro:** amostra de 4 fotos. Na reprodução em CPU local com os mesmos pesos, uma delas (rua molhada com tampa de bueiro) gerou 1 detecção de buraco com confiança 0,18, **logo acima do limiar 0,175**. Isso mostra que detecções de baixa confiança perto do limiar podem mudar entre GPU e CPU (diferenças numéricas): `buraco-itaquera.jpg` também diferiu (Colab: 1 rachadura; CPU local: 1 buraco + 1 rachadura).
- **Teste de buracos "externo":** as imagens do teste do repositório Pothole-Detector não foram usadas no treino, mas o train/valid da mesma fonte foi; mede desempenho nessa fonte, não domínio totalmente novo. O mAP@50 de 0,834 da fonte `rf_rua` (só buracos, mesma origem do treino) não é comparável aos 0,392 do Brasil e 0,343 do RDD2022.


## 5. Teste em fotos externas com rótulo (buracos)

| imagens | TP | FP | FN | precision | recall | f1 | conf | iou_match |
|---|---|---|---|---|---|---|---|---|
| 67 | 121 | 67 | 33 | 0.6436 | 0.7857 | 0.7076 | 0.1750 | 0.5000 |

Fonte: Muhammad-aa/Pothole-Detector (test; estas imagens NÃO foram usadas no treino, mas train/valid da MESMA fonte entraram no treino: mede desempenho nessa fonte, não domínio novo).

## 6. Fotos externas sem rótulo de caixa (esperado pela pasta × detectado)

buracos/rachaduras: acerto = ao menos um defeito da classe esperada; sem_danos: acerto = nenhuma detecção.

| pasta | imagens | acertos | media_defeitos | inferencia_ms |
|---|---|---|---|---|
| buracos | 67 | 66 | 2.8060 | 11.0 |
| rachaduras | 118 | 107 | 1.6695 | 9.1423 |
| sem_danos | 4 | 3 | 0.2500 | 60.5 |

**Nota sobre `sem_danos` (4 fotos de pavimento íntegro, de fontes da internet fornecidas pelo autor):** esta linha vem da **reprodução em CPU local** com os mesmos pesos (`best.pt`): 1 alarme de buraco com confiança 0,18 (rua molhada com tampa de bueiro), logo acima do limiar 0,175. Na execução do **Colab (GPU T4)** pela interface Gradio, as mesmas 4 fotos tiveram **0 detecções** (tabela 6b). A amostra é pequena (4 imagens) e a diferença mostra a sensibilidade de detecções de baixa confiança perto do limiar; não permite estimar uma taxa de falso alarme.


## 6b. Fotos do autor (sem rótulo)

14 fotos de ruas; defeitos detectados no limiar operacional. Sem rótulos de caixa: leitura qualitativa. Se existir `results/minhas_fotos/LEIA-ME.txt`, ele informa a origem da tabela.

| arquivo | defeitos | buracos | rachaduras | inference_ms |
|---|---|---|---|---|
| whatsapp-image-2025-07-03-at-13.44.48.avif | 5 | 5 | 0 | 19.7 |
| images (6).jpg | 2 | 2 | 0 | 34.2 |
| cratera-joao-pessoa-bayeux.webp | 2 | 2 | 0 | 35.3 |
| images (5).jpg | 7 | 7 | 0 | 30.1 |
| Pavimentacao-Travessa-Bezerra-13-e1708093359413.jpeg | 0 | 0 | 0 | 16.7 |
| photo-1546636889-ba9fdd63583e.avif | 0 | 0 | 0 | 39.9 |
| RUA-UMBELINO-DE-OLIVEIRA-scaled.jpg | 0 | 0 | 0 | 30 |
| images (7).jpg | 1 | 1 | 0 | 23.7 |
| pexels-anderson-martinsz-34634610.jpg | 0 | 0 | 0 | 67.8 |
| buraco.webp | 1 | 1 | 0 | 21 |
| buraco-itaquera.jpg | 1 | 0 | 1 | 18.7 |
| 6c692290939d5efe1c7d914d7c173f330db076ae.webp | 1 | 1 | 0 | 38 |
| 7_3.webp | 2 | 2 | 0 | 19.8 |
| rua-racha2.webp | 4 | 0 | 4 | 19.7 |

## 7. Monitoramento por sequência de imagens

40 quadros consecutivos do conjunto de teste · quadros com defeito: 40 · defeitos: 141 · processamento: **73.5 FPS** (14 ms/quadro, dispositivo 0, 640 px, limiar 0.175). Arquivos: `results/sequence/` (quadros anotados, `timeline.png`, `contact_sheet.jpg`, `sequence_log.csv`).
