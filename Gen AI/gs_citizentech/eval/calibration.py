"""Calibração do limiar de similaridade SEM vazamento de dados.

Usa apenas as 5 perguntas de CALIBRACAO_* (eval_sets.py), que não fazem parte dos conjuntos de teste.
Regra: se as notas das perguntas com resposta (mín.) ficam acima das notas das perguntas fora da base
(máx.), o limiar fica 30% acima do máximo negativo (inclinado ao recall: a abstenção final ainda é
protegida pela cobertura lexical e pelo gerador). Limitado a [0,6x; 1,5x] do padrão do backend.
O resultado é salvo em resultados/calibracao.json e reutilizado pela API e pela avaliação.
"""
import json
from typing import Dict, Optional

import config
from eval_sets import CALIBRACAO_NEG, CALIBRACAO_POS

CALIBRATION_FILE = config.RESULTS_DIR / "calibracao.json"


def _signature(retriever) -> Dict:
    e = retriever.embedder
    return {"backend": e.backend, "model": e.model_name if e.backend != "hashing" else "hashing",
            "chunking": retriever.chunks[0].metadata.get("chunking", ""), "n_chunks": len(retriever.chunks)}


def calibrate(retriever, save: bool = True) -> Dict:
    pos = [retriever.retrieve(q, k=1, mode="dense", threshold=-1.0)[0].score for q, _, _ in CALIBRACAO_POS]
    neg = [retriever.retrieve(q, k=1, mode="dense", threshold=-1.0)[0].score for q in CALIBRACAO_NEG]
    default = retriever.embedder.default_threshold
    if max(neg) < min(pos):
        thr = max(neg) + 0.3 * (min(pos) - max(neg))
        thr = min(max(thr, 0.6 * default), 1.5 * default)
        metodo = "calibrado (max_neg + 30% da folga até min_pos)"
    else:
        thr, metodo = default, "padrão do backend (classes não separáveis nas 5 perguntas)"
    info = {**_signature(retriever), "threshold": round(float(thr), 3), "metodo": metodo,
            "scores_positivos": [round(float(x), 3) for x in pos],
            "scores_negativos": [round(float(x), 3) for x in neg]}
    if save:
        config.RESULTS_DIR.mkdir(exist_ok=True)
        CALIBRATION_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def load(retriever) -> Optional[Dict]:
    try:
        info = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    sig = _signature(retriever)
    return info if all(info.get(k) == v for k, v in sig.items()) else None


def apply(retriever, force: bool = False) -> Dict:
    """Define retriever.threshold: SIM_THRESHOLD (env) > calibração salva > nova calibração."""
    if config.SIM_THRESHOLD is not None:
        return {"threshold": config.SIM_THRESHOLD, "metodo": "definido por SIM_THRESHOLD"}
    info = None if force else load(retriever)
    if info is None:
        info = calibrate(retriever)
    retriever.threshold = info["threshold"]
    return info
