# Entrega — Visão Computacional (Disciplina 09) · GS FIAP 2026.2

**Autor:** Octavio Augusto Ramalho Brandão Pires — RM 555696
**Vídeo:** https://youtu.be/vfjVLQFblYQ

## O que enviar no portal
| Item | Onde está |
|---|---|
| Projeto completo (código, pesos, resultados) | este zip |
| Relatório de resultados com métricas, justificativa do limiar e análise de erros | `reports/RESULTADOS.md` (e o relatório técnico em documento, se anexado) |
| Notebook executável (Colab) | `CitizenTech_CV_YOLO_Colab.ipynb` |
| Link do vídeo | https://youtu.be/vfjVLQFblYQ |

## Checklist dos requisitos
| Requisito | Evidência |
|---|---|
| Base RDD2022 + detecção por caixas | `reports/mix_report.json`, `data_extra.py`, `build_mix.py` |
| Split 70/15/15 sem *data leakage* | `reports/split_report.json` (grupos compartilhados: 0), `reports/mix_report.json` |
| YOLO11 pré-treinado (transfer learning) | `train_model.py`, `weights/best.pt` |
| Precision, Recall, F1, mAP@50, mAP@50-95 | `reports/RESULTADOS.md` §1, `reports/metrics.json` |
| Matriz de confusão e curvas | `reports/confusion_matrix.png`, `runs/eval_test/`, `reports/training_curves.png` |
| Análise de erros (sombras, texturas, iluminação, obstáculos) | `reports/RESULTADOS.md` §4, `reports/observacoes_visuais.md`, `reports/error_analysis/` |
| Latência média e FPS (equipamento e resolução) | `reports/RESULTADOS.md` §2: T4 10,4 ms / 96,5 FPS; CPU Xeon 2 threads 89,0 ms / 11,2 FPS; 640 px |
| Justificativa do limiar | `reports/RESULTADOS.md` §3 |
| Sequência de imagens (monitoramento) | `results/sequence/`, `reports/RESULTADOS.md` §7 |
| Interface com caixa e confiança | `app_gradio.py` (executada no Colab pelo autor) |
| Imagens independentes e pavimento sem danos | `tests/external_images/`, `results/external/`, `results/sem_danos/` |

## Pendências e limites (declare-os no relatório/vídeo)
- Comparação YOLO11n × YOLO11s **não executada** (opcional).
- Pavimento sem danos: **4 fotos** (amostra pequena); sem taxa de falso alarme confiável.
- Rachadura (mAP@50 0,401; recall 0,338) é bem mais fraca que buraco (0,678; 0,676), por causa das caixas fragmentadas do dataset brasileiro e de fissuras finas.
- Teste de buracos externo vem de fonte cujo train/valid entrou no treino.
- Procedência/licença do dataset de buracos do GitHub e das fotos do autor não verificadas.
- Chave da API do Roboflow exposta em notebook anterior: revogue e gere outra.
