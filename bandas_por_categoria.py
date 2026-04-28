import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_bandas_por_categoria(ruta_modelo: str | Path = "modelos_sulfato_regresion/sulfato_reg_4_capas") -> None:
    ruta_modelo = Path(ruta_modelo)
    ruta_csv = ruta_modelo / "predicciones_test.csv"
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    # Asegurar columnas numéricas
    if "y_real" in df.columns:
        df["y_real"] = pd.to_numeric(df["y_real"], errors="coerce")
    if "y_predicha" in df.columns:
        df["y_predicha"] = pd.to_numeric(df["y_predicha"], errors="coerce")

    df = df.dropna(subset=["y_real", "y_predicha"]).copy()
    if df.empty:
        raise ValueError("No hay datos válidos en 'y_real' y 'y_predicha' para graficar.")

    # Agrupar por y_real (cada categoría/valor real)
    grupos = df.groupby("y_real")["y_predicha"]
    xs = np.array(sorted(grupos.groups.keys()), dtype=float)

    mediana = []
    band25_low = []  # central 25% -> 37.5% - 62.5%
    band25_high = []
    band50_low = []  # central 50% -> 25% - 75%
    band50_high = []
    band75_low = []  # central 75% -> 12.5% - 87.5%
    band75_high = []
    extremos_low = []  # min
    extremos_high = []  # max

    for x in xs:
        vals = grupos.get_group(x).values
        mediana.append(np.percentile(vals, 50))
        band25_low.append(np.percentile(vals, 37.5))
        band25_high.append(np.percentile(vals, 62.5))
        band50_low.append(np.percentile(vals, 25))
        band50_high.append(np.percentile(vals, 75))
        band75_low.append(np.percentile(vals, 12.5))
        band75_high.append(np.percentile(vals, 87.5))
        extremos_low.append(np.min(vals))
        extremos_high.append(np.max(vals))

    mediana = np.array(mediana)
    band25_low = np.array(band25_low)
    band25_high = np.array(band25_high)
    band50_low = np.array(band50_low)
    band50_high = np.array(band50_high)
    band75_low = np.array(band75_low)
    band75_high = np.array(band75_high)
    extremos_low = np.array(extremos_low)
    extremos_high = np.array(extremos_high)

    plt.figure(figsize=(14, 7))

    # Mediana
    plt.plot(xs, mediana, marker="o", color="black", linestyle="-", label="Mediana")

    # Bandas 25% (verde)
    plt.plot(xs, band25_low, color="green", linestyle="--", linewidth=1.8, label="25% inferior (37.5%)")
    plt.plot(xs, band25_high, color="green", linestyle="--", linewidth=1.8, label="25% superior (62.5%)")

    # Bandas 50% (azul claro)
    plt.plot(xs, band50_low, color="deepskyblue", linestyle="-.", linewidth=1.8, label="50% inferior (25%)")
    plt.plot(xs, band50_high, color="deepskyblue", linestyle="-.", linewidth=1.8, label="50% superior (75%)")

    # Bandas 75% (amarillo) -> percentiles 12.5% / 87.5%
    plt.plot(xs, band75_low, color="yellow", linestyle=":", linewidth=1.8, label="75% inferior (12.5%)")
    plt.plot(xs, band75_high, color="yellow", linestyle=":", linewidth=1.8, label="75% superior (87.5%)")

    # Extremos (naranja) -> min y max
    plt.plot(xs, extremos_low, color="orange", linestyle=(0, (3, 1, 1, 1)), linewidth=1.8, label="Mínimo")
    plt.plot(xs, extremos_high, color="orangered", linestyle=(0, (3, 1, 1, 1)), linewidth=1.8, label="Máximo")

    # Linea 1:1
    plt.plot(xs, xs, color="red", linewidth=2.2, label="1:1 (Perfecto)")

    plt.xlabel("Valor Real de Sulfato", fontsize=12, fontweight="bold")
    plt.ylabel("Valor Predicho de Sulfato", fontsize=12, fontweight="bold")
    plt.title("Mediana y bandas centradas por categoría (25% verde, 50% azul claro)", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")

    # Evitar duplicar leyendas similares: crear handles/labels únicos
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="best")

    ruta_out = ruta_modelo / "bandas_por_categoria.png"
    plt.tight_layout()
    plt.savefig(ruta_out, dpi=300, bbox_inches="tight")
    print(f"Gráfica guardada en: {ruta_out}")
    plt.show()


if __name__ == "__main__":
    plot_bandas_por_categoria()
