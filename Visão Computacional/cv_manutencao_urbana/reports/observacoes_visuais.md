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
