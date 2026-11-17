"""Persistência simples (JSON Lines) do feedback humano. Sem dados pessoais: só protocolo e decisão."""
import json
import os
import threading
from datetime import datetime
from pathlib import Path

_lock = threading.Lock()


def _path() -> Path:
    return Path(os.getenv("DATA_DIR", "./data")) / "feedback.jsonl"


def save_feedback(record: dict) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "registrado_em": datetime.now().isoformat(timespec="seconds")}
    with _lock, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
