"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
make_report.py - gera reports/RESULTADOS.md (relatório breve) com as tabelas REAIS produzidas por evaluate.py
(reports/metrics.json), sequence_demo.py, infer.py e compare_models.py. Nenhum número é digitado à mão; as frases de
justificativa são montadas a partir dos valores medidos.
Uso: python make_report.py
"""
import json
from pathlib import Path

import pandas as pd

from common import AUTHOR, CLASS_PT, REPORTS_DIR, ROOT


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int,)) or (isinstance(v, float) and v == v and v.is_integer() and abs(v) >= 1):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    m = json.loads((REPORTS_DIR / "metrics.json").read_text())
    u, o = m["teste_ultralytics_conf0.001"], m["teste_limiar_operacional"]
    conf = o["conf"]
    out = [f"# Relatório de resultados (gerado automaticamente)\n\nAutor: {AUTHOR}  \nPesos: `{m['weights']}` · resolução de inferência: **{m['imgsz']} px** · dados: `{m.get('dados', '')}`\n"]

    pf = o["presenca_por_imagem"]
    bm0 = next(iter(m["benchmark"].values()), None) if m.get("benchmark") else None
    bt = m.get("benchmark", {})
    resumo = [f"- **mAP@50 = {u['mAP50']:.3f}** e **mAP@50-95 = {u['mAP50_95']:.3f}** no teste (todas as classes); Precision {u['precision']:.3f}, Recall {u['recall']:.3f}, F1 {u['f1']:.3f}.",
              f"- Por classe (mAP@50): " + "; ".join(f"{CLASS_PT[k]} {v['mAP50']:.3f}" for k, v in u["por_classe"].items()) + ".",
              f"- Como triagem por imagem (há buraco? há rachadura?): buraco P {pf['pothole']['precision']:.2f} / R {pf['pothole']['recall']:.2f}; rachadura P {pf['crack']['precision']:.2f} / R {pf['crack']['recall']:.2f}."]
    if bt:
        resumo.append("- Latência/FPS: " + "; ".join(f"{b['dispositivo']} {b['ponta_a_ponta']['latencia_media_ms']:.1f} ms ({b['ponta_a_ponta']['fps']:.1f} FPS) @ {b['imgsz']} px" for b in bt.values()) + ".")
    out.append("## Resumo dos principais resultados\n\n" + "\n".join(resumo) + "\n")
    out.append("## 1. Qualidade da detecção no TESTE (Precision, Recall, F1, mAP)\n"
               "P/R no ponto de máximo F1 e mAP (IoU 0.5 e 0.5:0.95) calculados pelo Ultralytics; F1 = 2PR/(P+R).\n")
    rows = [{"classe": CLASS_PT[k], **{x: v[x] for x in ("precision", "recall", "f1", "mAP50", "mAP50_95")}} for k, v in u["por_classe"].items()]
    rows.append({"classe": "Todas", **{x: u[x] for x in ("precision", "recall", "f1", "mAP50", "mAP50_95")}})
    out.append(md(pd.DataFrame(rows)))
    if "teste_por_fonte" in m:
        out.append("\n### Por fonte de dados (dataset combinado)\n")
        out.append(md(pd.DataFrame([{"fonte": k, "imagens": v["imagens"], "mAP50": v["mAP50"], "mAP50_95": v["mAP50_95"],
                                     "precision": v["precision_ult"], "recall": v["recall_ult"], "f1": v["f1_ult"],
                                     "recall_buraco_por_imagem": v["presenca_por_imagem"]["pothole"]["recall"],
                                     "recall_rachadura_por_imagem": v["presenca_por_imagem"]["crack"]["recall"]}
                                    for k, v in m["teste_por_fonte"].items()])))
    out.append(f"\n### No limiar operacional (conf = {conf:.3f}, IoU ≥ 0.5)\n")
    out.append(md(pd.DataFrame([{"classe": CLASS_PT[k], **v} for k, v in o["por_classe"].items()])))
    out.append("\nPresença por imagem (a imagem tem a classe? — métrica de triagem):\n")
    out.append(md(pd.DataFrame([{"classe": CLASS_PT[k], **v} for k, v in o["presenca_por_imagem"].items()])))
    out.append("\nMatriz de confusão: `reports/confusion_matrix.png` · curvas de treino: `reports/training_curves.png` · "
               "curva F1 × limiar (validação): `reports/f1_vs_conf.png`.\n")

    if "benchmark" in m:
        out.append("## 2. Latência e FPS (equipamento e resolução informados)\n")
        rows = []
        for key, b in m["benchmark"].items():
            for nome, v in (("ponta a ponta (pré-proc + modelo + NMS)", b["ponta_a_ponta"]), ("forward puro", b["forward_puro"])):
                rows.append({"equipamento": b["dispositivo"], "resolução (px)": b["imgsz"], "medida": nome,
                             "latência média (ms/imagem)": v["latencia_media_ms"], "p95 (ms)": v["p95_ms"], "FPS": v["fps"]})
        out.append(md(pd.DataFrame(rows)))
        b0 = next(iter(m["benchmark"].values()))
        out.append(f"\n{b0['warmup']} iterações de warmup + {b0['iteracoes']} medidas com `time.perf_counter()`; imagem de teste "
                   f"{b0['imagem_original'][0]}×{b0['imagem_original'][1]}, tensor de entrada {b0['tensor_forward']}.\n")

    out.append("## 3. Justificativa do limiar de confiança\n")
    txt = (f"O limiar operacional **{conf:.3f}** foi escolhido **na validação** como o valor que maximiza o F1 médio das classes "
           f"(IoU ≥ 0.5), sem usar o teste (curva em `reports/f1_vs_conf.png`). Valores menores aumentam o recall mas geram mais falsos "
           f"positivos (mais vistorias desnecessárias); valores maiores perdem danos reais. No teste, com esse limiar: ")
    txt += "; ".join(f"{CLASS_PT[k]} P={v['precision']:.2f}, R={v['recall']:.2f}, F1={v['f1']:.2f}" for k, v in o["por_classe"].items()) + "."
    out.append(txt + "\n")
    if "benchmark" in m:
        for key, b in m["benchmark"].items():
            lx = pd.DataFrame(b["limiar_x_latencia"])
            out.append(f"Efeito do limiar na velocidade ({b['dispositivo']}, {b['imgsz']} px): limiares mais baixos mantêm mais caixas "
                       f"candidatas na supressão não máxima (NMS) e podem aumentar o tempo de pós-processamento.\n")
            out.append(md(lx[["conf", "latencia_media_ms", "fps"]].rename(columns={"conf": "limiar", "latencia_media_ms": "latência média (ms)", "fps": "FPS"})))
            r25 = lx[lx.conf == 0.25]
            if len(r25):
                out.append(f"\nCautela: a tabela usa **uma** imagem de teste e poucas repetições. No mesmo limiar (0,25), a medição ponta a ponta da seção 2 deu {b['ponta_a_ponta']['latencia_media_ms']:.1f} ms e esta tabela {float(r25.latencia_media_ms.iloc[0]):.1f} ms; essa variação entre medições é da ordem do efeito do limiar, que portanto deve ser lido como tendência, não como valor exato.\n")
    out.append("\nA decisão equilibra qualidade e velocidade: o tempo por imagem é dominado pela inferência da rede (ver tabela da seção 2), "
               "de modo que escolher o limiar pelo F1 de validação custa pouco em latência.\n")

    ff = m.get("fatores_de_erro")
    out.append("## 4. Análise de erros (acertos e falhas)\n")
    out.append("Galeria **real (verde) × previsto** em `reports/error_analysis/` (prefixo `acerto_` = sem erros; `erro_` = piores casos). "
               "Abaixo, a influência medida de iluminação, sombra, textura e tamanho do objeto (por tercil das imagens de teste; "
               "recall = objetos detectados com a classe certa / objetos reais, IoU ≥ 0.5):\n")
    if ff:
        nomes = {"brilho": "Iluminação (brilho médio)", "area_escura": "Sombra/subexposição (fração de pixels escuros)",
                 "textura": "Textura (densidade de bordas)", "contraste": "Contraste", "tamanho_objeto": "Tamanho do objeto (área relativa)"}
        for fac, d in ff.items():
            df = pd.DataFrame([{"faixa": k, **v} for k, v in d.items()])
            out.append(f"**{nomes.get(fac, fac)}**\n")
            out.append(md(df))
            out.append("")
    obs = REPORTS_DIR / "observacoes_visuais.md"
    if obs.exists():
        out.append(obs.read_text(encoding="utf-8") + "\n")
    else:
      out.append("### Observações visuais (preencha após ver as imagens de `reports/error_analysis/` e dos testes externos)\n"
               "Descreva, para 2–3 exemplos de acerto e 2–3 de erro, o que ocorreu e o possível motivo. Interferências a considerar: "
               "**sombras** (de postes, veículos, vegetação), **texturas** do asfalto (remendos, juntas, desgaste confundidos com rachaduras), "
               "**iluminação** (contraluz, subexposição, reflexo), **obstáculos** (veículos, pessoas, poças, marcações de faixa) e objetos "
               "muito **pequenos ou distantes**. Cite os arquivos observados.\n")

    if "externo_buracos" in m:
        e = m["externo_buracos"]
        out += ["## 5. Teste em fotos externas com rótulo (buracos)\n", md(pd.DataFrame([{k: v for k, v in e.items() if k != "fonte"}])),
                f"\nFonte: {e['fonte']}.\n"]
    evs = sorted((ROOT / "results").glob("*/expected_vs_detected.csv"))
    if evs:
        d = pd.concat([pd.read_csv(p) for p in evs], ignore_index=True)
        g = d.groupby("pasta").agg(imagens=("resultado_ok", "size"), acertos=("resultado_ok", "sum"),
                                   media_defeitos=("quantidade_defeitos", "mean"), inferencia_ms=("inference_ms", "mean")).reset_index()
        out += ["## 6. Fotos externas sem rótulo de caixa (esperado pela pasta × detectado)\n",
                "buracos/rachaduras: acerto = ao menos um defeito da classe esperada; sem_danos: acerto = nenhuma detecção.\n", md(g), ""]
        nota = REPORTS_DIR / "nota_sem_danos.md"
        if nota.exists():
            out += [nota.read_text(encoding="utf-8"), ""]
    mf = ROOT / "results" / "minhas_fotos" / "summary.csv"
    if mf.exists():
        d = pd.read_csv(mf)
        out += ["## 6b. Fotos do autor (sem rótulo)\n", f"{len(d)} fotos de ruas; defeitos detectados no limiar operacional. Sem rótulos de caixa: leitura qualitativa. Se existir `results/minhas_fotos/LEIA-ME.txt`, ele informa a origem da tabela.\n",
                md(d[[c for c in ("arquivo", "quantidade_defeitos", "buracos", "rachaduras", "conf_maxima", "inference_ms") if c in d.columns]].rename(columns={"quantidade_defeitos": "defeitos"})), ""]
    sq = ROOT / "results" / "sequence" / "sequence_summary.json"
    if sq.exists():
        s = json.loads(sq.read_text())
        out += ["## 7. Monitoramento por sequência de imagens\n",
                f"{s['quadros']} quadros consecutivos do conjunto de teste · quadros com defeito: {s['quadros_com_defeito']} · defeitos: {s['total_defeitos']} · "
                f"processamento: **{s['fps_processamento_medio']:.1f} FPS** ({s['latencia_media_ms']:.0f} ms/quadro, dispositivo {s['dispositivo']}, {s['imgsz']} px, limiar {s['conf_usada']:.3f}). "
                "Arquivos: `results/sequence/` (quadros anotados, `timeline.png`, `contact_sheet.jpg`, `sequence_log.csv`).\n"]
    cm = REPORTS_DIR / "comparacao_modelos.json"
    if cm.exists():
        c = json.loads(cm.read_text())
        out += [f"## 8. Comparação de arquiteturas ({c['dispositivo']})\n", md(pd.DataFrame(c["modelos"])), ""]
    (REPORTS_DIR / "RESULTADOS.md").write_text("\n".join(out), encoding="utf-8")
    print("[ok] reports/RESULTADOS.md")


if __name__ == "__main__":
    main()
