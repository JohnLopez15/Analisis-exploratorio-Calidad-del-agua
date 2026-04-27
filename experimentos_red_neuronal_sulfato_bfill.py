import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def definir_capas_ocultas(numero_capas_ocultas: int) -> list[int]:
    if numero_capas_ocultas < 1:
        raise ValueError("El número de capas ocultas debe ser mayor o igual a 1.")

    capas = []
    neuronas = 64
    for _ in range(numero_capas_ocultas):
        capas.append(neuronas)
        neuronas = max(neuronas // 2, 8)
    return capas


def preparar_datos(
    columnas_a_evaluar: list[str],
    columna_objetivo: str = "Sulfato",
    db_path: str = "sensores.db",
    tabla: str = "datos_unificados",
) -> tuple[pd.DataFrame, pd.Series]:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {tabla}", conn)

    if "FechaHora" in df.columns:
        df["FechaHora"] = pd.to_datetime(df["FechaHora"], errors="coerce")

    for columna in columnas_a_evaluar:
        if columna not in df.columns:
            raise ValueError(f"La columna '{columna}' no existe en la tabla '{tabla}'.")
        df[columna] = df[columna].bfill()

    if columna_objetivo not in df.columns:
        raise ValueError(f"La columna objetivo '{columna_objetivo}' no existe en la tabla '{tabla}'.")

    df[columna_objetivo] = df[columna_objetivo].bfill()

    columnas_necesarias = columnas_a_evaluar + [columna_objetivo]
    df_limpio = df[columnas_necesarias].dropna()

    if df_limpio.empty:
        raise ValueError("No hay datos disponibles tras la limpieza para entrenar el modelo.")

    x = df_limpio[columnas_a_evaluar]
    y = df_limpio[columna_objetivo].astype(str)
    return x, y


def entrenar_red_sulfato(
    columnas_a_evaluar: list[str],
    numero_capas_ocultas: int,
    nombre_experimento: str | None = None,
    columna_objetivo: str = "Sulfato",
    db_path: str = "sensores.db",
    tabla: str = "datos_unificados",
    carpeta_salida: str = "modelos_sulfato",
    epochs: int = 50,
    batch_size: int = 10,
    validation_split: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    if not columnas_a_evaluar:
        raise ValueError("Debes indicar al menos una variable en 'columnas_a_evaluar'.")

    x, y = preparar_datos(
        columnas_a_evaluar=columnas_a_evaluar,
        columna_objetivo=columna_objetivo,
        db_path=db_path,
        tabla=tabla,
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if len(np.unique(y_encoded)) < 2:
        raise ValueError("La variable objetivo debe tener al menos dos clases para clasificación.")

    y_categorical = to_categorical(y_encoded)
    clases_objetivo = label_encoder.classes_

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled,
        y_categorical,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    numero_entradas = x_train.shape[1]
    numero_salidas = y_categorical.shape[1]
    capas_ocultas = definir_capas_ocultas(numero_capas_ocultas)

    modelo = Sequential()
    modelo.add(Input(shape=(numero_entradas,)))
    for neuronas in capas_ocultas:
        modelo.add(Dense(neuronas, activation="relu"))
    modelo.add(Dense(numero_salidas, activation="softmax"))

    modelo.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

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

    loss, accuracy = modelo.evaluate(x_test, y_test, verbose=0)

    inicio_prediccion = perf_counter()
    y_pred_prob = modelo.predict(x_test, verbose=0)
    fin_prediccion = perf_counter()
    tiempo_total_prediccion_segundos = fin_prediccion - inicio_prediccion
    latencia_promedio_prediccion_ms = (tiempo_total_prediccion_segundos / len(x_test)) * 1000

    y_pred = np.argmax(y_pred_prob, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    diferencias_categorias = y_pred - y_test_labels
    aciertos_exactos = y_pred == y_test_labels
    dentro_rango_pm2 = np.abs(diferencias_categorias) <= 2

    eficiencia_global = {
        "total_muestras_test": int(len(y_test_labels)),
        "aciertos_exactos_count": int(aciertos_exactos.sum()),
        "aciertos_exactos_porcentaje": float(aciertos_exactos.mean() * 100),
        "dentro_rango_pm2_count": int(dentro_rango_pm2.sum()),
        "dentro_rango_pm2_porcentaje": float(dentro_rango_pm2.mean() * 100),
        "por_encima_hasta_2_count": int(((diferencias_categorias >= 1) & (diferencias_categorias <= 2)).sum()),
        "por_debajo_hasta_2_count": int(((diferencias_categorias <= -1) & (diferencias_categorias >= -2)).sum()),
    }

    eficiencia_por_categoria = {}
    for idx_categoria, nombre_categoria in enumerate(clases_objetivo):
        mascara_categoria_real = y_test_labels == idx_categoria
        total_categoria = int(mascara_categoria_real.sum())

        if total_categoria == 0:
            eficiencia_por_categoria[str(nombre_categoria)] = {
                "total_reales": 0,
                "aciertos_exactos": 0,
                "aciertos_exactos_porcentaje": 0.0,
                "dentro_rango_pm2": 0,
                "dentro_rango_pm2_porcentaje": 0.0,
            }
            continue

        aciertos_categoria = int((aciertos_exactos & mascara_categoria_real).sum())
        dentro_pm2_categoria = int((dentro_rango_pm2 & mascara_categoria_real).sum())

        eficiencia_por_categoria[str(nombre_categoria)] = {
            "total_reales": total_categoria,
            "aciertos_exactos": aciertos_categoria,
            "aciertos_exactos_porcentaje": float((aciertos_categoria / total_categoria) * 100),
            "dentro_rango_pm2": dentro_pm2_categoria,
            "dentro_rango_pm2_porcentaje": float((dentro_pm2_categoria / total_categoria) * 100),
        }

    cm = confusion_matrix(y_test_labels, y_pred)
    cm_df = pd.DataFrame(cm, index=clases_objetivo, columns=clases_objetivo)
    reporte = classification_report(
        y_test_labels,
        y_pred,
        target_names=clases_objetivo.astype(str),
        output_dict=True,
        zero_division=0,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_base = nombre_experimento or f"exp_{timestamp}"
    ruta_experimento = Path(carpeta_salida) / nombre_base
    ruta_experimento.mkdir(parents=True, exist_ok=True)

    ruta_modelo = ruta_experimento / "modelo.keras"
    ruta_scaler = ruta_experimento / "scaler.pkl"
    ruta_encoder = ruta_experimento / "label_encoder.pkl"
    ruta_matriz_confusion = ruta_experimento / "matriz_confusion.csv"
    ruta_predicciones = ruta_experimento / "predicciones_test.csv"
    ruta_metricas = ruta_experimento / "metricas.json"

    modelo.save(ruta_modelo)

    with open(ruta_scaler, "wb") as f_scaler:
        pickle.dump(scaler, f_scaler)

    with open(ruta_encoder, "wb") as f_encoder:
        pickle.dump(label_encoder, f_encoder)

    cm_df.to_csv(ruta_matriz_confusion, index=True)

    predicciones_df = pd.DataFrame(
        {
            "y_real": clases_objetivo[y_test_labels],
            "y_predicha": clases_objetivo[y_pred],
        }
    )
    predicciones_df.to_csv(ruta_predicciones, index=False)

    metricas = {
        "fecha_ejecucion": timestamp,
        "columna_objetivo": columna_objetivo,
        "columnas_a_evaluar": columnas_a_evaluar,
        "numero_capas_ocultas": numero_capas_ocultas,
        "neuronas_por_capa": capas_ocultas,
        "clases_objetivo": clases_objetivo.tolist(),
        "loss_test": float(loss),
        "accuracy_test": float(accuracy),
        "tiempo_entrenamiento_segundos": float(tiempo_entrenamiento_segundos),
        "tiempo_entrenamiento_ms": float(tiempo_entrenamiento_segundos * 1000),
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
        "reporte_clasificacion": reporte,
        "rutas_salida": {
            "modelo": str(ruta_modelo),
            "scaler": str(ruta_scaler),
            "label_encoder": str(ruta_encoder),
            "matriz_confusion": str(ruta_matriz_confusion),
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
    resultado = entrenar_red_sulfato(
        columnas_a_evaluar=columnas,
        numero_capas_ocultas=2,
        nombre_experimento="sulfato_base_2_capas_bfill",
    )
    print("Accuracy test:", round(resultado["accuracy_test"], 4))
    print("Tiempo entrenamiento (s):", round(resultado["tiempo_entrenamiento_segundos"], 4))
    print(
        "Latencia promedio predicción (ms):",
        round(resultado["latencia_prediccion"]["latencia_promedio_prediccion_ms"], 4),
    )
    print(
        "Exactitud global (%):",
        round(resultado["analisis_eficiencia"]["global"]["aciertos_exactos_porcentaje"], 2),
    )
    print(
        "Dentro de ±2 categorías (%):",
        round(resultado["analisis_eficiencia"]["global"]["dentro_rango_pm2_porcentaje"], 2),
    )
    print("Resultados guardados en:", resultado["rutas_salida"]["modelo"])
    
    columnas = [
        "Turbiedad_cruda",
        "Color_agua_Natural",
        "pH_cruda",
        "Alcalinidad_total_cruda",
    ]
    resultado = entrenar_red_sulfato(
        columnas_a_evaluar=columnas,
        numero_capas_ocultas=4,
        nombre_experimento="sulfato_base_4_capas_bfill",
    )
    print("Accuracy test:", round(resultado["accuracy_test"], 4))
    print("Tiempo entrenamiento (s):", round(resultado["tiempo_entrenamiento_segundos"], 4))
    print(
        "Latencia promedio predicción (ms):",
        round(resultado["latencia_prediccion"]["latencia_promedio_prediccion_ms"], 4),
    )
    print(
        "Exactitud global (%):",
        round(resultado["analisis_eficiencia"]["global"]["aciertos_exactos_porcentaje"], 2),
    )
    print(
        "Dentro de ±2 categorías (%):",
        round(resultado["analisis_eficiencia"]["global"]["dentro_rango_pm2_porcentaje"], 2),
    )
    print("Resultados guardados en:", resultado["rutas_salida"]["modelo"])
    
    columnas = [
        "Turbiedad_cruda",
        "Color_agua_Natural",
        "pH_cruda",
    ]
    resultado = entrenar_red_sulfato(
        columnas_a_evaluar=columnas,
        numero_capas_ocultas=5,
        nombre_experimento="sulfato_3_datos_5_capas_bfill",
    )
    print("Accuracy test:", round(resultado["accuracy_test"], 4))
    print("Tiempo entrenamiento (s):", round(resultado["tiempo_entrenamiento_segundos"], 4))
    print(
        "Latencia promedio predicción (ms):",
        round(resultado["latencia_prediccion"]["latencia_promedio_prediccion_ms"], 4),
    )
    print(
        "Exactitud global (%):",
        round(resultado["analisis_eficiencia"]["global"]["aciertos_exactos_porcentaje"], 2),
    )
    print(
        "Dentro de ±2 categorías (%):",
        round(resultado["analisis_eficiencia"]["global"]["dentro_rango_pm2_porcentaje"], 2),
    )
    print("Resultados guardados en:", resultado["rutas_salida"]["modelo"])
    columnas = [
        "Turbiedad_cruda",
        "Color_agua_Natural",
        "pH_cruda",
        "Alcalinidad_total_cruda",
    ]
    resultado = entrenar_red_sulfato(
        columnas_a_evaluar=columnas,
        numero_capas_ocultas=2,
        nombre_experimento="sulfato_base_2_capas_bfill",
    )
    print("Accuracy test:", round(resultado["accuracy_test"], 4))
    print("Tiempo entrenamiento (s):", round(resultado["tiempo_entrenamiento_segundos"], 4))
    print(
        "Latencia promedio predicción (ms):",
        round(resultado["latencia_prediccion"]["latencia_promedio_prediccion_ms"], 4),
    )
    print(
        "Exactitud global (%):",
        round(resultado["analisis_eficiencia"]["global"]["aciertos_exactos_porcentaje"], 2),
    )
    print(
        "Dentro de ±2 categorías (%):",
        round(resultado["analisis_eficiencia"]["global"]["dentro_rango_pm2_porcentaje"], 2),
    )
    print("Resultados guardados en:", resultado["rutas_salida"]["modelo"])
    
    