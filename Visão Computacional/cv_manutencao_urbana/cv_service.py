"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
GS FIAP 2026.2 - CitizenTech - Visão Computacional

cv_service.py - módulo de integração (FastAPI / Streamlit / Gradio). Evoluído do classificador MobileNetV3 para o
detector YOLO11n. Funciona com IMAGENS individuais (sem vídeo).

    from cv_service import predict_urban_damage
    res = predict_urban_damage(uploaded.getvalue(), conf=None, return_images=True)

Retorno (dict):
  num_defects, counts{"Buraco","Rachadura"}, detections[{class, class_en, confidence, bbox_xyxy, bbox_norm_xywh}],
  requires_inspection (True se houver >=1 detecção), image_size [w, h], inference_ms (somente o modelo),
  processing_time_ms (decodificação + pré-proc + inferência + NMS), conf_used,
  e, se return_images=True: original_png e annotated_png (bytes PNG).

Observações: o modelo sinaliza locais para TRIAGEM/inspeção; não estima gravidade nem segurança estrutural.
`conf=None` usa o limiar operacional escolhido na validação (weights/threshold.json) ou 0.25 se ausente.
Privacidade: a imagem é processada em memória; este módulo não grava nem registra a imagem.
"""
import json
import os
import threading
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from common import BEST_PT, CLASS_COLORS_BGR, CLASS_NAMES, CLASS_PT, IMGSZ, WEIGHTS_DIR

_LOCK = threading.Lock()
_STATE = {}
_MAX_BYTES = 20 * 1024 * 1024


def get_device() -> str:
    """CV_DEVICE=cpu|0 força o dispositivo; senão usa GPU se houver (ex.: T4 no Colab), senão CPU."""
    if os.environ.get("CV_DEVICE"):
        return os.environ["CV_DEVICE"]
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def default_conf() -> float:
    f = WEIGHTS_DIR / "threshold.json"
    return float(json.loads(f.read_text())["conf"]) if f.exists() else 0.25


def get_model(weights=None):
    path = str(weights or BEST_PT)
    if _STATE.get("path") != path:
        with _LOCK:
            if _STATE.get("path") != path:
                from ultralytics import YOLO
                if not Path(path).exists():
                    raise FileNotFoundError(f"Pesos não encontrados: {path}. Rode train_model.py primeiro.")
                model = YOLO(path)
                model.predict(np.zeros((IMGSZ, IMGSZ, 3), np.uint8), imgsz=IMGSZ, device=get_device(), verbose=False)  # warmup
                _STATE.update(path=path, model=model)
    return _STATE["model"]


def decode_image(src: Union[bytes, str, Path]) -> np.ndarray:
    if isinstance(src, (bytes, bytearray)):
        if len(src) > _MAX_BYTES:
            raise ValueError("Imagem excede 20 MB.")
        img = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            try:
                import io
                from PIL import Image
                img = cv2.cvtColor(np.array(Image.open(io.BytesIO(src)).convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                img = None
    elif isinstance(src, (str, Path)):
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:                       # webp/avif etc. que o OpenCV local não leia: tenta o Pillow
            try:
                from PIL import Image
                try:
                    import pillow_avif  # noqa: F401  (opcional: habilita .avif)
                except ImportError:
                    pass
                img = cv2.cvtColor(np.array(Image.open(src).convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                img = None
    else:
        raise TypeError("Informe bytes ou caminho (str/Path) da imagem.")
    if img is None:
        raise ValueError("Arquivo não é uma imagem válida.")
    return img


def model_imgsz(model) -> int:
    return int((getattr(model, 'ckpt', None) or {}).get('train_args', {}).get('imgsz', IMGSZ))


def detect_image(img_bgr: np.ndarray, conf=None, iou: float = 0.6, imgsz=None, weights=None):
    """Retorna (detections, timing_ms). Boxes em pixels da imagem original."""
    conf = default_conf() if conf is None else conf
    model = get_model(weights)
    imgsz = imgsz or model_imgsz(model)
    r = model.predict(img_bgr, imgsz=imgsz, conf=conf, iou=iou, device=get_device(), verbose=False)[0]
    h, w = img_bgr.shape[:2]
    dets = []
    for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                      r.boxes.cls.cpu().numpy().astype(int)):
        dets.append({"class": CLASS_PT[CLASS_NAMES[k]], "class_en": CLASS_NAMES[k], "confidence": round(float(c), 4),
                     "bbox_xyxy": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                     "bbox_norm_xywh": [round(float(v), 5) for v in ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                                                                     (x2 - x1) / w, (y2 - y1) / h)]})
    dets.sort(key=lambda d: -d["confidence"])
    return dets, {k: round(float(v), 2) for k, v in r.speed.items()}


def draw_detections(img_bgr: np.ndarray, dets: list) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    th = max(2, round(min(h, w) / 300))
    scale = max(0.5, min(h, w) / 900)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["bbox_xyxy"])
        col = CLASS_COLORS_BGR[d["class_en"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
        label = f'{d["class"]} {d["confidence"] * 100:.0f}%'
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        y0 = max(tht + 6, y1)
        cv2.rectangle(out, (x1, y0 - tht - 6), (x1 + tw + 6, y0), col, -1)
        cv2.putText(out, label, (x1 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def predict_urban_damage(image_bytes_or_path: Union[bytes, str], conf=None, imgsz=None,
                         return_images: bool = False) -> dict:
    """Detecta e localiza buracos e rachaduras em UMA imagem."""
    t0 = time.perf_counter()
    img = decode_image(image_bytes_or_path)
    conf = default_conf() if conf is None else conf
    if not 0.0 < conf < 1.0:
        raise ValueError("conf deve estar entre 0 e 1.")
    dets, timing = detect_image(img, conf=conf, imgsz=imgsz)
    res = {"num_defects": len(dets),
           "counts": {"Buraco": sum(d["class_en"] == "pothole" for d in dets),
                      "Rachadura": sum(d["class_en"] == "crack" for d in dets)},
           "detections": dets, "requires_inspection": len(dets) > 0,
           "image_size": [img.shape[1], img.shape[0]], "conf_used": conf, "imgsz": imgsz or model_imgsz(get_model()),
           "device": get_device(),
           "inference_ms": timing.get("inference"), "timing_ms": timing,
           "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2)}
    if return_images:
        res["original_png"] = cv2.imencode(".png", img)[1].tobytes()
        res["annotated_png"] = cv2.imencode(".png", draw_detections(img, dets))[1].tobytes()
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("uso: python cv_service.py <imagem> [conf]")
    print(json.dumps(predict_urban_damage(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None),
                     indent=2, ensure_ascii=False))
