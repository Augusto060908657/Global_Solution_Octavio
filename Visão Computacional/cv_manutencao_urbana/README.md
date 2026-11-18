# CitizenTech "Cidade Para Todos" — Detecção de buracos e rachaduras com YOLO11

**Autor:** Octavio Augusto Ramalho Brandão Pires — **RM 555696** · Disciplina 09 — Visão Computacional · Global Solution FIAP 2026.2 (2TIA) · **versão 3.1.0**  
**Vídeo de demonstração:** https://youtu.be/vfjVLQFblYQ
**Google Colab Executado:** https://colab.research.google.com/drive/1yzuhc_He30FpVZANT2cI89IMOlgrwVzX?usp=sharing
**Google Drive Dataset:** https://drive.google.com/drive/folders/1C5nAYQlW-R6grixC8W0wOffZE4KjeFjW?usp=sharing
## Início rápido

### Google Colab (recomendado, GPU T4)
1. Abra `CitizenTech_CV_YOLO_Colab.ipynb` no Colab e ative **Ambiente de execução → GPU T4**.
2. No Drive, pasta `GS_CV/`: coloque **este projeto em .zip (v3.1.0)** e o zip do dataset brasileiro (`t576ydh9v8-3.zip`).
3. Rode as células em ordem. O notebook **para no primeiro erro**, avisa se o zip do projeto estiver desatualizado, tem um **teste de fumaça** que valida tudo antes do treino longo, e termina com um **checklist automático** dos requisitos.

### Local (CPU)
```bash
pip install -r requirements.txt
# coloque o .zip do dataset brasileiro em dataset/
python run.py all --smoke --no-extra      # (opcional) teste de fumaça: roda todas as etapas em poucos minutos
python run.py clean                       # apaga as saídas do teste
python run.py all                         # completo (CPU é lento; use --quick para ensaiar)
python run.py app                         # interface Gradio
```

## Conformidade com o enunciado
| Requisito | Onde é atendido |
|---|---|
| Base sugerida (RDD2022) e detecção com caixas | `data_extra.py rdd2022` (Hugging Face, CC BY-SA 4.0) + dataset brasileiro + Roboflow; `build_mix.py` |
| Split 70/15/15 sem partilhar a mesma imagem/trecho entre conjuntos | `data_prep.py` (grupos de trechos + zona-tampão + auditoria dHash) e `build_mix.py` (grupos por país/bloco); `assert` de interseção zero |
| Arquitetura pré-treinada de detecção | YOLO11n (COCO) com transfer learning (`train_model.py`); comparação opcional com YOLO11s (`compare_models.py`) |
| Precision, Recall, F1, mAP@50, mAP@50-95, matriz de confusão | `evaluate.py` → `reports/metrics.json`, `reports/confusion_matrix.png` (+ por fonte e por classe) |
| Análise de erros (acertos/falhas; sombras, texturas, iluminação, obstáculos) | `reports/error_analysis/` (real × previsto) + estatística por iluminação/sombra/textura/tamanho em `RESULTADOS.md` + seção de observações visuais |
| Vídeo curto **ou sequência de imagens** | `sequence_demo.py` (quadros consecutivos do teste, FPS, linha do tempo); aceita pasta de fotos e, opcionalmente, vídeo |
| Latência média e FPS, equipamento e resolução | `evaluate.py --bench_devices cpu,0` (T4 e CPU, warmup + 100 iterações, instância nova por dispositivo e verificação de qual dispositivo foi usado) |
| Justificativa do limiar de confiança | limiar escolhido na **validação** (F1 médio), tabela limiar × latência/FPS; texto gerado em `RESULTADOS.md` §3 |
| Front-end: imagem → caixa do dano + confiança | `app_gradio.py` (imagem única e sequência), `cv_service.py`, `api_example.py` |

## 1. Problema
Vistorias manuais de vias são lentas e caras. O módulo recebe **uma fotografia** (ou uma sequência) e devolve **onde** há buracos e rachaduras (bounding boxes), a **categoria**, a **confiança**, a **quantidade** de problemas e o **tempo de inferência**, apoiando a triagem de zeladoria. Ele **sinaliza locais para inspeção**: não estima gravidade nem substitui o técnico (revisão humana). ODS 9, 10 e 11.

## 2. Por que a versão 3
Uma primeira versão treinada **só com o dataset brasileiro** (rodovias fotografadas de dentro de um veículo) funcionou no teste interno (mAP@50 ≈ 0,38 no teste, presença de rachadura por imagem com recall ≈ 0,89), mas **generalizou mal**: buracos em fotos de rua de outras fontes quase não foram detectados e as fotos do próprio autor não geraram detecção. A causa é mudança de domínio (ângulo, escala, resolução). A v3 responde com: (1) **mais dados diversos** (RDD2022 + buracos Roboflow + projeto Roboflow do autor), (2) **augmentation robusta**, (3) mais épocas na GPU, (4) avaliação **por fonte** e demonstrações reais, (5) fluxo de execução blindado (verificação de versão, parada no primeiro erro, smoke test).

## 3. Dados
| Fonte | Conteúdo | Licença / origem |
|---|---|---|
| **Cracks and Potholes in Road Images** (Brasil) | ~2.235 fotos de rodovias (RS/DF/ES) com máscaras de rachadura, buraco e faixa; convertidas em caixas | Kaggle (arquivo do enunciado); conferir licença na página |
| **RDD2022** (Arya et al., arXiv:2209.08538) | subconjunto por grupos de quadros consecutivos de 5 países/dispositivos (sem drone); D00/D10/D20 → rachadura, D40 → buraco | Hugging Face `dronefreak/RDD2022` (reexportação YOLO), CC BY-SA 4.0 |
| Buracos Roboflow (GitHub) | `Muhammad-aa/Pothole-Detector`: train/valid no treino; **test** = teste externo | procedência original a verificar |
| Seu projeto Roboflow (opcional) | `trabaio/analise-de-buracos-na-rua-…` v1, via API (chave nos Secrets) | conforme o projeto |

**Classes finais:** `0 = buraco (pothole)`, `1 = rachadura (crack)`. Classes extras (ex.: faixa de sinalização) são descartadas. O mapeamento é automático por nome (`pothole/buraco/hole`, `crack/rachadura/trinca/fissura`) e pode ser forçado com `--map`. Polígonos de segmentação são convertidos em caixas.

**Formatos verificados** (`reports/dataset_inspection.json`): imagens RAW JPEG; máscaras PNG de 3 canais binárias (0/255) do mesmo tamanho. Conversão máscara → caixa por componentes conectados após dilatação (une fragmentos de uma rachadura), conferida visualmente em `reports/label_check/`.

**Anti-leakage:** (a) Brasil: grupo = trecho + bloco de 200 quadros consecutivos, cada grupo em um só split (seed 42), zona-tampão de 30 quadros e auditoria de quase-duplicatas (dHash); (b) RDD2022: grupo = país + bloco de 50 números consecutivos, split 70/15/15 refeito por grupo; (c) Roboflow: split original (augmentações da mesma foto ficam juntas). `build_mix.py` verifica que **nenhum grupo aparece em dois splits**.

Exemplo real (execução no Colab com `--groups_per_country 8`): Brasil 1497/335/335, Roboflow-GitHub 465/133/0, RDD2022 1531/324/311 imagens em treino/val/teste (`reports/mix_report.json`, ~32% das imagens do RDD2022 sem dano servem de negativos).

## 4. Modelo e treinamento
**YOLO11n** (Ultralytics), ~2,6 M de parâmetros, detector de estágio único; transfer learning dos pesos COCO (`weights/yolo11n.pt`), cabeça de 2 classes reinicializada, rede inteira ajustada. Preset `robust`: escala, perspectiva, rotação leve, HSV (brilho/saturação), mixup, apagamento, *cosine LR*. Na T4: 40 épocas (a validação estabiliza antes), batch 16, 640 px, *early stopping*; seed 42. `python run.py all --compare` treina também o **YOLO11s** e gera `reports/COMPARACAO_MODELOS.md` (mAP, parâmetros, latência) para justificar a escolha da arquitetura.

## 5. Avaliação, latência e limiar
`evaluate.py` mede no **teste** (nunca usado em treino nem para escolher o limiar): Precision/Recall/F1/mAP@50/mAP@50-95 (Ultralytics) por classe, total e **por fonte**; métricas no limiar operacional (IoU ≥ 0.5) e **presença por imagem** (triagem); matriz de confusão; fatores de erro por tercil de brilho, sombra, textura e tamanho do objeto. **Limiar:** escolhido na validação pelo F1 médio; a tabela limiar × latência/FPS mostra o custo em velocidade. **Latência/FPS:** T4 e CPU, ponta a ponta e forward puro, 10 warmups + 100 iterações, resolução informada; cada dispositivo usa uma instância nova do modelo e o código confere em qual dispositivo ele realmente rodou.

## 6. Demonstrações (todas geradas pelo modelo treinado)
`python run.py gallery` / notebook §9: `reports/demo/dashboard.png` (painel-resumo), `teste_<fonte>.jpg` (real verde × previsto), `externas_<pasta>.jpg`, `externas_minhas_fotos.jpg`. `python run.py demo`: por foto externa original/detecção/comparativo/JSON + `results/external/galeria.html`. `python run.py sequence`: `results/sequence/` (quadros anotados, `timeline.png`, `contact_sheet.jpg`, FPS). Interface: `python run.py app` (ou notebook §14, com link público).

## 7. Resultados (execução real: Google Colab, GPU T4, 640 px, 40 épocas)
Todos os números abaixo vêm de `reports/metrics.json` e `reports/RESULTADOS.md` (gerados por `evaluate.py`/`make_report.py` nesta execução). O relatório completo, com a análise de erros, está em **`reports/RESULTADOS.md`**.

**Teste combinado (884 imagens: Brasil 335 + RDD2022 311 + Roboflow-rua 238).** Precision/Recall no ponto de máximo F1 (Ultralytics), mAP com IoU 0,5 e 0,5:0,95:
| Classe | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| Buraco | 0,625 | 0,676 | 0,650 | 0,678 | 0,406 |
| Rachadura | 0,584 | 0,338 | 0,429 | 0,401 | 0,176 |
| **Todas** | **0,604** | **0,507** | **0,552** | **0,539** | **0,291** |

**Por fonte (teste):** Brasil mAP@50 0,392 (mAP@50-95 0,184); RDD2022 0,343 (0,150); Roboflow-rua 0,834 (0,527; só buracos, mesma origem do treino, não comparável às demais).
**Limiar de confiança:** 0,175, escolhido na validação (F1 médio). Como triagem por imagem no teste: buraco P 0,91 / R 0,87; rachadura P 0,96 / R 0,86.
**Latência (ponta a ponta, 640 px, imagem 1024×640):** GPU Tesla T4 **10,4 ms (96,5 FPS)**; CPU Intel Xeon 2,0 GHz com 2 threads **89,0 ms (11,2 FPS)**.
**Imagens externas:** buracos (67 fotos com rótulo de outra origem; train/valid da mesma fonte entraram no treino): precisão 0,64, recall 0,79, F1 0,71 por caixa; ao menos um buraco em 66 de 67 imagens. Rachaduras (CrackForest, 118 close-ups, domínio diferente): ao menos uma rachadura em 107 de 118. Fotos do autor (14, interface Gradio no Colab): 10 com defeitos (26 detecções), 4 de pavimento íntegro sem detecção; na reprodução em CPU local, 1 dessas 4 teve um alarme de 0,18 (ver `reports/RESULTADOS.md`).
**Sequência de imagens:** 40 quadros consecutivos do teste: 141 defeitos, **73,5 FPS** (13,6 ms/quadro) na T4.
**Treino:** 40 épocas; mAP@50 de validação ≈ 0,55 e mAP@50-95 ≈ 0,30 ao final, ainda em leve alta (curvas em `reports/training_curves.png`).

**Comparação com a v1 (só dataset brasileiro):** no teste brasileiro o resultado é parecido (mAP@50 0,38 → 0,39), mas a generalização melhorou: em fotos externas de rachaduras, de 0/118 para 107/118 imagens com detecção.
**Não executado:** a comparação opcional YOLO11n × YOLO11s (`--compare`).

Demonstrações: `reports/demo/` (painel, mosaicos real × previsto, externas), `results/external/`, `results/sequence/`. Análise de erros: `reports/error_analysis/` e `reports/observacoes_visuais.md`.

## 8. Estrutura
| Arquivo | Função |
|---|---|
| `run.py` | ponto de entrada (`check, prep, extra, mix, train, eval, gallery, demo, sequence, report, app, clean, all`) |
| `data_prep.py`, `data_extra.py`, `build_mix.py` | dataset brasileiro, fontes extras (RDD2022/Roboflow), mistura sem leakage |
| `train_model.py`, `evaluate.py`, `compare_models.py` | treino, avaliação/latência, comparação de arquiteturas |
| `demo_gallery.py`, `infer.py`, `sequence_demo.py` | demonstrações, inferência em imagens, sequência |
| `cv_service.py`, `api_example.py`, `app_gradio.py` | integração, API de exemplo, interface |
| `make_report.py` | `reports/RESULTADOS.md` |
| `tests/`, `minhas_fotos/`, `dataset/` | imagens externas, suas fotos, zip do dataset |

## 9. Segurança e ética
- **Nunca coloque a chave da API do Roboflow dentro do notebook/código** (use Secrets do Colab ou variável `ROBOFLOW_API_KEY`). Se uma chave já foi exposta, **revogue e gere outra** no painel do Roboflow.
- LGPD: o serviço processa em memória e não grava imagens; em produção, desfocar rostos e placas; manter revisão humana. Respeitar licenças (RDD2022: CC BY-SA 4.0 com atribuição).

## 10. Limitações e melhorias
- O dataset brasileiro é de rodovia; ruas urbanas, calçadas, noite e chuva exigem mais dados locais (ideal: fotos anotadas de SP). `sem_danos/` tem apenas 4 fotos: não permite estimar taxa de falso alarme.
- Rachaduras longas viram caixas grandes (máscaras convertidas); segmentação (YOLO11n-seg) usaria as máscaras diretamente.
- D20 (couro de jacaré) e fissuras lineares são unificadas em "rachadura"; sem gravidade nem tipo de fissura.
- O teste externo de buracos vem de fonte cujo train/valid entra no treino (mede essa fonte, não domínio novo); CrackForest e suas fotos medem generalização real.

## 11. Sessões longas: Colab × VS Code
O código é o mesmo em qualquer lugar (`run.py`); o notebook é só um invólucro. O que limita é **computação**, não a ferramenta.
| Cenário | O que fazer |
|---|---|
| **Colab (T4)** | Notebook v3.1: checkpoints no **Drive** a cada época, treino em **segundo plano**, saída filtrada (antes o notebook chegava a 3 MB só de barras de progresso) e célula **Retomar** (`train_model.py --resume --project <Drive>`). 40 épocas ≈ 1h30–2h; `FAST=True` (512 px, 30 épocas) ≈ 1h. |
| **VS Code com GPU NVIDIA** | `pip install -r requirements.txt` (PyTorch com CUDA) e `python run.py all --device 0`. Sem limite de sessão. |
| **VS Code só com CPU** | Viável para preparar dados, avaliar, demonstrar e rodar o app; **treino completo não é prático** (≈ 8 mil imagens a 640 px). Treine no Colab e copie `weights/best.pt` para o projeto local. |
| Sessão caiu | Reconecte, rode as células 1–3 e a Parte A, depois **Retomar**. Os pesos já treinados ficam em `GS_CV/resultados/`. |
Dica: antes de salvar ou compartilhar o notebook, use **Editar → Limpar todas as saídas**. A mensagem "arquivo atualizado remotamente" do Colab indica o mesmo arquivo aberto em outra aba: feche as cópias.

## 12. Solução de problemas
| Sintoma | Causa / ação |
|---|---|
| `ZIP DESATUALIZADO` / `unrecognized arguments` | o zip do Drive é antigo: substitua pelo zip v3.1.0 |
| `GPU: NÃO disponível` | ative a GPU T4 no Colab |
| Falha no RDD2022/Hugging Face | rode novamente; sem internet o pipeline segue só com o dataset brasileiro |
| Roboflow não baixa | confira workspace/projeto/versão e a chave (Secrets); o resto roda sem ele |
