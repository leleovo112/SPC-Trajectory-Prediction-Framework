# src/utils/io_utils.py
import os
import json
from typing import Any

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)
