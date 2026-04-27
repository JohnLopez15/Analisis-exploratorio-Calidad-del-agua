import json
from pathlib import Path

import numpy as np
import pandas as pd


def analizar_modelos(ruta_modelos: str | Path = "modelos_sulfato") -> list[dict]:
    ruta_modelos = Path(ruta_modelos)
    if not ruta_modelos.exists():
        raise FileNotFoundError(f"La carpeta {ruta_modelos} no existe")

    resultados = []
    eps = 1e-8

    for sub in sorted(ruta_modelos.iterdir()):
        if not sub.is_dir():
            continue

        res = {"modelo": sub.name, "ruta": str(sub), "applicable": False}

        ruta_pred = sub / "predicciones_test.csv"
        if not ruta_pred.exists():
            res["error"] = "predicciones_test.csv no encontrado"
            resultados.append(res)
            continue

        try:
            df = pd.read_csv(ruta_pred)
        except Exception as e:
            res["error"] = f"error leyendo CSV: {e}"
            resultados.append(res)
            continue

        # Columnas esperadas: y_real, y_predicha (flexible con mayúsculas/minúsculas)
        col_map = {c.lower(): c for c in df.columns}
        if "y_real" not in col_map or "y_predicha" not in col_map:
            res["error"] = f"columnas requeridas no encontradas en {ruta_pred}"
            resultados.append(res)
            continue

        y_real_raw = df[col_map["y_real"]]
        y_pred_raw = df[col_map["y_predicha"]]

        y_real_num = pd.to_numeric(y_real_raw, errors="coerce")
        y_pred_num = pd.to_numeric(y_pred_raw, errors="coerce")

        mask_valid = y_real_num.notna() & y_pred_num.notna()
        n_valid = int(mask_valid.sum())
        n_total = len(df)

        if n_valid == 0:
            res["error"] = "sin valores numéricos válidos para evaluar"
            resultados.append(res)
            continue

        y_real = y_real_num[mask_valid].to_numpy(dtype=float)
        y_pred = y_pred_num[mask_valid].to_numpy(dtype=float)

        abs_error = np.abs(y_pred - y_real)
        pct_error = (abs_error / np.maximum(np.abs(y_real), eps)) * 100.0

        stats = {
            "n_total_rows": int(n_total),
            "n_valid_numeric": int(n_valid),
            "abs_error_mean": float(np.mean(abs_error)),
            "abs_error_median": float(np.median(abs_error)),
            "abs_error_min": float(np.min(abs_error)),
            "abs_error_max": float(np.max(abs_error)),
            "abs_error_p25": float(np.percentile(abs_error, 25)),
            "abs_error_p75": float(np.percentile(abs_error, 75)),
            "pct_error_mean": float(np.mean(pct_error)),
            "pct_error_median": float(np.median(pct_error)),
            "pct_error_min": float(np.min(pct_error)),
            "pct_error_max": float(np.max(pct_error)),
            "pct_error_p25": float(np.percentile(pct_error, 25)),
            "pct_error_p75": float(np.percentile(pct_error, 75)),
        }

        res["applicable"] = True
        res.update(stats)

        # si existe metricas.json, adjuntarla (solo resumen mínimo)
        ruta_metricas = sub / "metricas.json"
        if ruta_metricas.exists():
            try:
                with open(ruta_metricas, "r", encoding="utf-8") as f:
                    met = json.load(f)
                # incluir loss/mae/mse si existen
                for k in ("mae_test", "mse_test", "accuracy_test", "loss_test"):
                    if k in met:
                        res[k] = met[k]
            except Exception:
                pass

        resultados.append(res)

    return resultados


def guardar_resumen(resultados: list[dict], ruta_salida: str | Path = "modelos_sulfato/errores_resumen"):
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    json_path = ruta_salida.with_suffix(".json")
    csv_path = ruta_salida.with_suffix(".csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # Normalizar resultados para CSV
    df = pd.json_normalize(resultados)
    df.to_csv(csv_path, index=False)

    return json_path, csv_path


if __name__ == "__main__":
    resultados = analizar_modelos("modelos_sulfato")
    json_path, csv_path = guardar_resumen(resultados, "modelos_sulfato/errores_resumen")
    print(f"Resumen guardado: {json_path} and {csv_path}")
    # imprimir tabla resumida en consola
    df = pd.DataFrame(resultados)
    display_cols = [
        "modelo",
        "applicable",
        "n_valid_numeric",
        "abs_error_mean",
        "pct_error_mean",
    ]
    print(df[display_cols].fillna("-").to_string(index=False))
