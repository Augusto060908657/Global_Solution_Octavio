"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
GS FIAP 2026.2 - CitizenTech - Visão Computacional

app_gradio.py - protótipo de interface CitizenTech (Gradio) integrado ao detector YOLO:
  Aba 1 - UMA imagem: mostra original e imagem processada com bounding boxes + categoria + confiança, quantidade de
          problemas, se requer inspeção e tempo de inferência.
  Aba 2 - SEQUÊNCIA de imagens (várias fotos em ordem): monitoramento quadro a quadro com FPS de processamento e
          tabela por quadro.
UX: cores semânticas (vermelho = buraco, laranja = rachadura), limiar de confiança ajustável, aviso de revisão humana.
Uso local:  python app_gradio.py            (abre em http://127.0.0.1:7860)
Colab:      python app_gradio.py --share    (gera link público temporário)
"""
import argparse
import time

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from common import AUTHOR
from cv_service import default_conf, detect_image, draw_detections, get_device, get_model, model_imgsz

AVISO = ("⚠️ O sistema **sinaliza locais para inspeção**. A presença de um dano na imagem não permite determinar a "
         "gravidade nem a segurança estrutural: a decisão final é de um técnico (revisão humana).")


def analyze(img_rgb, conf, imgsz):
    if img_rgb is None:
        raise gr.Error("Envie uma imagem.")
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    imgsz = int(imgsz) if imgsz else None
    t0 = time.perf_counter()
    dets, timing = detect_image(bgr, conf=conf, imgsz=imgsz)
    total = (time.perf_counter() - t0) * 1000
    ann = cv2.cvtColor(draw_detections(bgr, dets), cv2.COLOR_BGR2RGB)
    nb = sum(d["class_en"] == "pothole" for d in dets); nr = len(dets) - nb
    if dets:
        resumo = (f"### 🔴 {len(dets)} problema(s) identificado(s)\n- Buracos: **{nb}** · Rachaduras: **{nr}**\n"
                  f"- Maior confiança: **{max(d['confidence'] for d in dets) * 100:.0f}%**\n- Requer inspeção: **sim**\n")
    else:
        resumo = "### 🟢 Nenhum problema detectado neste limiar\n- Requer inspeção: **não indicado pelo modelo**\n"
    resumo += (f"- Tempo de inferência: **{timing.get('inference')} ms** (total {total:.0f} ms) · dispositivo: {get_device()} · "
               f"limiar: {conf:.2f}\n\n{AVISO}")
    tab = pd.DataFrame([{"Categoria": d["class"], "Confiança (%)": round(d["confidence"] * 100, 1),
                         "Caixa [x1,y1,x2,y2]": d["bbox_xyxy"]} for d in dets],
                       columns=["Categoria", "Confiança (%)", "Caixa [x1,y1,x2,y2]"])
    return img_rgb, ann, resumo, tab, {"num_defects": len(dets), "detections": dets, "inference_ms": timing.get("inference")}


def analyze_sequence(files, conf, imgsz):
    if not files:
        raise gr.Error("Envie várias imagens em ordem (nome ordenado = ordem dos quadros).")
    paths = sorted(f if isinstance(f, str) else f.name for f in files)
    imgsz = int(imgsz) if imgsz else None
    gallery, rows, t_all = [], [], 0.0
    for i, p in enumerate(paths, 1):
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        t0 = time.perf_counter()
        dets, timing = detect_image(bgr, conf=conf, imgsz=imgsz)
        ann = draw_detections(bgr, dets)
        dt = time.perf_counter() - t0
        t_all += dt
        gallery.append((cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), f"Quadro {i}: {len(dets)} defeito(s)"))
        rows.append({"Quadro": i, "Arquivo": p.split("/")[-1], "Defeitos": len(dets),
                     "Buracos": sum(d["class_en"] == "pothole" for d in dets),
                     "Rachaduras": sum(d["class_en"] == "crack" for d in dets), "ms/quadro": round(dt * 1000, 1)})
    df = pd.DataFrame(rows)
    n = max(len(df), 1)
    resumo = (f"### Sequência processada: {len(df)} quadros\n- Quadros com defeito: **{int((df.Defeitos > 0).sum())}** · "
              f"total de defeitos: **{int(df.Defeitos.sum())}**\n- Processamento: **{n / max(t_all, 1e-9):.1f} FPS** "
              f"(média {t_all / n * 1000:.0f} ms/quadro, dispositivo {get_device()})\n\n{AVISO}")
    return gallery, resumo, df


def build_app():
    conf0 = default_conf()
    with gr.Blocks(title="CitizenTech - Manutenção Urbana") as demo:
        gr.Markdown(f"# CitizenTech · Manutenção Urbana com Visão Computacional\n"
                    f"Detecção de **buracos** e **rachaduras** em fotos (YOLO11). _{AUTHOR}_")
        with gr.Row():
            conf = gr.Slider(0.05, 0.9, value=conf0, step=0.01, label="Limiar de confiança (menor = mais sensível)")
            imgsz = gr.Dropdown([416, 512, 640, 800, 960], value=None, label="Resolução de inferência (vazio = a do treino)")
        with gr.Tabs():
            with gr.Tab("Uma imagem"):
                inp = gr.Image(type="numpy", label="Foto do pavimento")
                btn = gr.Button("Analisar", variant="primary")
                with gr.Row():
                    o1 = gr.Image(label="Original"); o2 = gr.Image(label="Detecções (caixa, categoria e confiança)")
                md = gr.Markdown(); tb = gr.Dataframe(label="Defeitos"); js = gr.JSON(label="Resultado (JSON)")
                btn.click(analyze, [inp, conf, imgsz], [o1, o2, md, tb, js])
            with gr.Tab("Sequência de imagens (monitoramento)"):
                fl = gr.File(file_count="multiple", file_types=["image"], label="Fotos em ordem")
                b2 = gr.Button("Processar sequência", variant="primary")
                gl = gr.Gallery(label="Quadros anotados", columns=3, height=520); md2 = gr.Markdown(); tb2 = gr.Dataframe()
                b2.click(analyze_sequence, [fl, conf, imgsz], [gl, md2, tb2])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--share", action="store_true"); ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    get_model()  # carrega e aquece o modelo antes de abrir a interface
    build_app().launch(share=a.share, server_port=a.port)
