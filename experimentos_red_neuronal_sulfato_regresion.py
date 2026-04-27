import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential


def definir_capas_ocultas(numero_capas_ocultas: int) -> list[int]:
    if numero_capas_ocultas < 1:
        raise ValueError("El número de capas ocultas debe ser mayor o igual a 1.")

    capas = []
    neuronas = 64
    for _ in range(numero_capas_ocultas):
        capas.append(neuronas)
        neuronas = max(neuronas // 2, 8)
    return capas


def preparar_datos_regresion(
    columnas_a_evaluar: list[str],
    columna_objetivo: str = "Sulfato",
    db_path: str = "sensores.db",
    tabla: str = "datos_unificados",
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {tabla}", conn)

    if "FechaHora" in df.columns:
        df["FechaHora"] = pd.to_datetime(df["FechaHora"], errors="coerce")

    for columna in columnas_a_evaluar:
        if columna not in df.columns:
            raise ValueError(f"La columna '{columna}' no existe en la tabla '{tabla}'.")
        df[columna] = df[columna].interpolate(method="linear")

    if columna_objetivo not in df.columns:
        raise ValueError(f"La columna objetivo '{columna_objetivo}' no existe en la tabla '{tabla}'.")

    # Rellenar hacia atrás si hay valores faltantes en la columna objetivo
    df[columna_objetivo] = pd.to_numeric(df[columna_objetivo], errors="coerce")
    df[columna_objetivo] = df[columna_objetivo].bfill()

    columnas_necesarias = columnas_a_evaluar + [columna_objetivo]
    df_limpio = df[columnas_necesarias].dropna()

    if df_limpio.empty:
        raise ValueError("No hay datos disponibles tras la limpieza para entrenar el modelo.")

    x = df_limpio[columnas_a_evaluar]
    y = df_limpio[columna_objetivo].astype(float)

    # clases discretas para evaluación (valores únicos ordenados)
    clases_discretas = np.unique(y.values)

    return x, y, clases_discretas


def entrenar_red_sulfato_regresion(
    columnas_a_evaluar: list[str],
    numero_capas_ocultas: int,
    nombre_experimento: str | None = None,
    columna_objetivo: str = "Sulfato",
    db_path: str = "sensores.db",
    tabla: str = "datos_unificados",
    carpeta_salida: str = "modelos_sulfato_regresion",
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    if not columnas_a_evaluar:
        raise ValueError("Debes indicar al menos una variable en 'columnas_a_evaluar'.")

    x, y, clases_discretas = preparar_datos_regresion(
        columnas_a_evaluar=columnas_a_evaluar,
        columna_objetivo=columna_objetivo,
        db_path=db_path,
        tabla=tabla,
    )

    # Escaladores para X e Y
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x)

    y_values = y.values.reshape(-1, 1)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_values)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled,
        y_scaled,
        test_size=test_size,
        random_state=random_state,
    )

    numero_entradas = x_train.shape[1]

    capas_ocultas = definir_capas_ocultas(numero_capas_ocultas)

    modelo = Sequential()
    modelo.add(Input(shape=(numero_entradas,)))
    for neuronas in capas_ocultas:
        modelo.add(Dense(neuronas, activation="relu"))
    modelo.add(Dense(1, activation="linear"))

    modelo.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"]) 

    inicio_entrenamiento = perf_counter()
    historial = modelo.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0,
    )
    fin_entrenamiento = perf_counter()
    tiempo_entrenamiento_segundos = fin_entrenamiento - inicio_entrenamiento

    loss, mae, mse = modelo.evaluate(x_test, y_test, verbose=0)

    inicio_prediccion = perf_counter()
    y_pred_scaled = modelo.predict(x_test, verbose=0)
    fin_prediccion = perf_counter()
    tiempo_total_prediccion_segundos = fin_prediccion - inicio_prediccion
    latencia_promedio_prediccion_ms = (tiempo_total_prediccion_segundos / len(x_test)) * 1000

    # Invertir escalado de y
    y_pred_inv = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
    y_test_inv = scaler_y.inverse_transform(y_test).reshape(-1)

    # Errores
    abs_error = np.abs(y_pred_inv - y_test_inv)
    eps = 1e-8
    percent_error = (abs_error / np.maximum(np.abs(y_test_inv), eps)) * 100.0

    # Evaluación basada en clases discretas definidas por los valores únicos de la variable
    if len(clases_discretas) > 1:
        # construir bordes (midpoints) para discretizar continuo
        midpoints = (clases_discretas[:-1] + clases_discretas[1:]) / 2.0
        clase_real_idx = np.digitize(y_test_inv, bins=midpoints)
        clase_pred_idx = np.digitize(y_pred_inv, bins=midpoints)
    else:
        clase_real_idx = np.zeros_like(y_test_inv, dtype=int)
        clase_pred_idx = np.zeros_like(y_pred_inv, dtype=int)

    diferencias_categorias = clase_pred_idx - clase_real_idx
    aciertos_exactos = diferencias_categorias == 0
    dentro_rango_1 = np.abs(diferencias_categorias) <= 1
    dentro_rango_2 = np.abs(diferencias_categorias) <= 2

    eficiencia_global = {
        "total_muestras_test": int(len(y_test_inv)),
        "aciertos_exactos_count": int(aciertos_exactos.sum()),
        "aciertos_exactos_porcentaje": float(aciertos_exactos.mean() * 100),
        "dentro_rango_1_count": int(dentro_rango_1.sum()),
        "dentro_rango_1_porcentaje": float(dentro_rango_1.mean() * 100),
        "dentro_rango_2_count": int(dentro_rango_2.sum()),
        "dentro_rango_2_porcentaje": float(dentro_rango_2.mean() * 100),
        "por_encima_hasta_2_count": int(((diferencias_categorias >= 1) & (diferencias_categorias <= 2)).sum()),
        "por_debajo_hasta_2_count": int(((diferencias_categorias <= -1) & (diferencias_categorias >= -2)).sum()),
        "mae_test": float(mae),
        "mse_test": float(mse),
        "error_promedio_abs": float(np.mean(abs_error)),
        "error_promedio_pct": float(np.mean(percent_error)),
    }

    # Estadísticas por clase (según clase real)
    eficiencia_por_categoria = {}
    clases = np.unique(clase_real_idx)
    for idx in clases:
        mask = clase_real_idx == idx
        errores_c = abs_error[mask]
        errores_pct_c = percent_error[mask]
        total_categoria = int(mask.sum())

        if total_categoria == 0:
            eficiencia_por_categoria[str(int(idx))] = {
                "total_reales": 0,
                "error_abs_mean": 0.0,
                "error_abs_min": 0.0,
                "error_abs_max": 0.0,
                "error_abs_p25": 0.0,
                "error_abs_p75": 0.0,
                "error_pct_mean": 0.0,
            }
            continue

        eficiencia_por_categoria[str(int(idx))] = {
            "total_reales": total_categoria,
            "error_abs_mean": float(np.mean(errores_c)),
            "error_abs_min": float(np.min(errores_c)),
            "error_abs_max": float(np.max(errores_c)),
            "error_abs_p25": float(np.percentile(errores_c, 25)),
            "error_abs_p75": float(np.percentile(errores_c, 75)),
            "error_pct_mean": float(np.mean(errores_pct_c)),
        }

    # Preparar directorio de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_base = nombre_experimento or f"exp_reg_{timestamp}"
    ruta_experimento = Path(carpeta_salida) / nombre_base
    ruta_experimento.mkdir(parents=True, exist_ok=True)

    ruta_modelo = ruta_experimento / "modelo.keras"
    ruta_scaler_x = ruta_experimento / "scaler_x.pkl"
    ruta_scaler_y = ruta_experimento / "scaler_y.pkl"
    ruta_predicciones = ruta_experimento / "predicciones_test.csv"
    ruta_metricas = ruta_experimento / "metricas.json"

    modelo.save(ruta_modelo)

    with open(ruta_scaler_x, "wb") as f:
        pickle.dump(scaler_x, f)

    with open(ruta_scaler_y, "wb") as f:
        pickle.dump(scaler_y, f)

    predicciones_df = pd.DataFrame(
        {
            "y_real": y_test_inv,
            "y_predicha": y_pred_inv,
            "error_abs": abs_error,
            "error_pct": percent_error,
            "clase_real_idx": clase_real_idx,
            "clase_pred_idx": clase_pred_idx,
            "diferencia_clases": diferencias_categorias,
        }
    )
    predicciones_df.to_csv(ruta_predicciones, index=False)

    metricas = {
        "fecha_ejecucion": timestamp,
        "columna_objetivo": columna_objetivo,
        "columnas_a_evaluar": columnas_a_evaluar,
        "numero_capas_ocultas": numero_capas_ocultas,
        "neuronas_por_capa": capas_ocultas,
        "clases_discretas": clases_discretas.tolist(),
        "tiempo_entrenamiento_segundos": float(tiempo_entrenamiento_segundos),
        "latencia_prediccion": {
            "tiempo_total_prediccion_segundos": float(tiempo_total_prediccion_segundos),
            "latencia_promedio_prediccion_ms": float(latencia_promedio_prediccion_ms),
            "muestras_test": int(len(x_test)),
        },
        "analisis_eficiencia": {
            "global": eficiencia_global,
            "por_categoria": eficiencia_por_categoria,
        },
        "historial": {k: [float(v) for v in valores] for k, valores in historial.history.items()},
        "rutas_salida": {
            "modelo": str(ruta_modelo),
            "scaler_x": str(ruta_scaler_x),
            "scaler_y": str(ruta_scaler_y),
            "predicciones_test": str(ruta_predicciones),
        },
    }

    with open(ruta_metricas, "w", encoding="utf-8") as f_metricas:
        json.dump(metricas, f_metricas, ensure_ascii=False, indent=2)

    return metricas


if __name__ == "__main__":
    columnas = [
        "Turbiedad_cruda",
        "Color_agua_Natural",
        "pH_cruda",
        "Alcalinidad_total_cruda",
    ]
    resultado = entrenar_red_sulfato_regresion(
        columnas_a_evaluar=columnas,
        numero_capas_ocultas=4,
        nombre_experimento="sulfato_reg_4_capas",
        epochs=100,
        batch_size=32,
    )
    print("MAE test:", round(resultado["analisis_eficiencia"]["global"]["mae_test"], 4))
    print("MSE test:", round(resultado["analisis_eficiencia"]["global"]["mse_test"], 4))
    print("Error promedio (%):", round(resultado["analisis_eficiencia"]["global"]["error_promedio_pct"], 2))
    print("Dentro ±1 clase (%):", round(resultado["analisis_eficiencia"]["global"]["dentro_rango_1_porcentaje"], 2))
    print("Resultados guardados en:", resultado["rutas_salida"]["modelo"]) 
