import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

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
        df[columna] = df[columna].interpolate(method="linear")

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

    historial = modelo.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0,
    )

    loss, accuracy = modelo.evaluate(x_test, y_test, verbose=0)

    y_pred_prob = modelo.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

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
        nombre_experimento="sulfato_base_2_capas",
    )
    print("Accuracy test:", round(resultado["accuracy_test"], 4))
    print("Resultados guardados en:", resultado["rutas_salida"]["modelo"])