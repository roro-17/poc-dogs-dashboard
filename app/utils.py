import json, os, numpy as np
from typing import Any, List

def load_metrics(path: str) -> Any:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_models_list(metrics_obj) -> List[str]:
    if isinstance(metrics_obj, list):
        return [d.get("model", f"model_{i}") for i, d in enumerate(metrics_obj)]
    if isinstance(metrics_obj, dict) and "models" in metrics_obj:
        return [d.get("model", f"model_{i}") for i, d in enumerate(metrics_obj["models"])]
    return []

def get_confusion_matrix(metrics_obj, model_name: str):
    import numpy as np
    if isinstance(metrics_obj, list):
        for d in metrics_obj:
            if d.get("model") == model_name and "confusion_matrix" in d:
                return np.array(d["confusion_matrix"])
    if isinstance(metrics_obj, dict) and "models" in metrics_obj:
        for d in metrics_obj["models"]:
            if d.get("model") == model_name and "confusion_matrix" in d:
                return np.array(d["confusion_matrix"])
    if isinstance(metrics_obj, dict) and "confusion_matrix" in metrics_obj:
        return np.array(metrics_obj["confusion_matrix"])
    return None

def metrics_to_dataframe(metrics_obj):
    import pandas as pd
    if isinstance(metrics_obj, list):
        return pd.DataFrame(metrics_obj)
    if isinstance(metrics_obj, dict) and "models" in metrics_obj:
        return pd.DataFrame(metrics_obj["models"])
    return pd.DataFrame([])
