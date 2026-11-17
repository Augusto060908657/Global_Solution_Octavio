# Imagens externas

Nenhuma destas imagens pertence ao dataset brasileiro de treino. Mapeamento arquivo -> origem em `origem_arquivos.csv`.

| Pasta | Fonte | Observação |
|---|---|---|
| `buracos/` (67) | Conjunto de **teste** do repositório GitHub `Muhammad-aa/Pothole-Detector` (rótulos YOLO em `../external_labels/buracos`) | Fotos de buracos de ângulos, distâncias e iluminações variados. **Atenção:** quando `pothole_github` entra no treino (train/valid da MESMA fonte), este teste mede desempenho *nessa fonte* (imagens inéditas, mesma origem), não domínio novo. Procedência original não declarada no repositório (o tutorial associado usa um dataset do Roboflow): verificar antes de redistribuir. |
| `rachaduras/` (118) | CrackForest (Shi et al., IEEE T-ITS, 2016), `cuilimeng/CrackForest-dataset`, imagens 001-118 | Close-ups de asfalto urbano (Pequim), 480x320, **domínio bem diferente** do treino: teste real de generalização. Sem caixas (só máscaras .mat, não convertidas): avaliação por presença de detecção. |
| `sem_danos/` (4) | Fotos de pavimento íntegro fornecidas pelo autor, de fontes da internet (procedência/licença não verificadas; não redistribuir) | Amostra pequena: serve como verificação qualitativa de falso alarme, não como taxa. |
| `../../minhas_fotos/` | Suas fotos | Entram automaticamente nas demonstrações. |
