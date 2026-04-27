import pandas as pd


def leer_matriz_desde_csv(ruta_csv: str) -> pd.DataFrame:
    """
    Lee una matriz de confusión desde un archivo CSV y la devuelve como un DataFrame de pandas.

    Args:
        ruta_csv (str): La ruta al archivo CSV que contiene la matriz de confusión.

    Returns:
        pd.DataFrame: Un DataFrame que representa la matriz de confusión.
    """
    try:
        matriz_confusion = pd.read_csv(ruta_csv, index_col=0)
        return matriz_confusion
    except Exception as e:
        print(f"Error al leer la matriz de confusión desde {ruta_csv}: {e}")
        return pd.DataFrame()


def graficar_matriz_confusion(matriz_confusion: pd.DataFrame, titulo: str = "Matriz de Confusión") -> None:
    """
    Grafica una matriz de confusión utilizando seaborn.

    Args:
        matriz_confusion (pd.DataFrame): Un DataFrame que representa la matriz de confusión.
        titulo (str): El título del gráfico.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues")
    plt.title(titulo)
    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")
    plt.show()


def construir_df_eficiencia_desde_metricas(modelos: list[dict[str, str]]) -> pd.DataFrame:
    import json
    from pathlib import Path

    filas = []

    for modelo in modelos:
        ruta_metricas = Path(modelo["ruta_metricas"])
        nombre_modelo = modelo["nombre"]

        with open(ruta_metricas, "r", encoding="utf-8") as f_metricas:
            metricas = json.load(f_metricas)

        por_categoria = metricas["analisis_eficiencia"]["por_categoria"]

        for categoria, valores in por_categoria.items():
            filas.append(
                {
                    "modelo": nombre_modelo,
                    "categoria": categoria,
                    "aciertos_exactos_porcentaje": valores["aciertos_exactos_porcentaje"],
                    "dentro_rango_pm2_porcentaje": valores["dentro_rango_pm2_porcentaje"],
                }
            )

    df = pd.DataFrame(filas)
    if df.empty:
        return df

    # Ordena categorias numericamente para facilitar la lectura visual.
    df["categoria_num"] = pd.to_numeric(df["categoria"], errors="coerce")
    df = df.sort_values(["categoria_num", "modelo"]).drop(columns=["categoria_num"])

    return df


def graficar_comparacion_eficiencia(df_eficiencia: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df_eficiencia.empty:
        print("No hay datos de eficiencia para graficar.")
        return

    categorias_ordenadas = (
        pd.to_numeric(df_eficiencia["categoria"], errors="coerce")
        .sort_values()
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    df_plot = df_eficiencia.copy()
    df_plot["categoria"] = pd.Categorical(
        df_plot["categoria"], categories=categorias_ordenadas, ordered=True
    )

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

    sns.lineplot(
        data=df_plot,
        x="categoria",
        y="aciertos_exactos_porcentaje",
        hue="modelo",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Comparacion por categoria: Aciertos exactos (%)")
    axes[0].set_ylabel("Porcentaje")
    axes[0].set_xlabel("")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    sns.lineplot(
        data=df_plot,
        x="categoria",
        y="dentro_rango_pm2_porcentaje",
        hue="modelo",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_title("Comparacion por categoria: Dentro de +/-2 categorias (%)")
    axes[1].set_ylabel("Porcentaje")
    axes[1].set_xlabel("Categoria real")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    # Deja una sola leyenda para toda la figura.
    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()

    fig.legend(handles, labels, title="Modelo", loc="upper center", ncol=3)
    fig.suptitle("Comparacion de eficiencia por categoria para 3 modelos", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


if __name__ == "__main__":
    direccion = "modelos_sulfato/sulfato_base_2_capas/matriz_confusion.csv"
    matriz_confusion = leer_matriz_desde_csv(direccion)
    graficar_matriz_confusion(matriz_confusion, titulo="Matriz de Confusion - Sulfato Base 2 Capas")

    direccion = "modelos_sulfato/sulfato_base_4_capas/matriz_confusion.csv"
    matriz_confusion = leer_matriz_desde_csv(direccion)
    graficar_matriz_confusion(matriz_confusion, titulo="Matriz de Confusion - Sulfato Base 4 Capas")

    direccion = "modelos_sulfato/sulfato_3_datos_5_capas/matriz_confusion.csv"
    matriz_confusion = leer_matriz_desde_csv(direccion)
    graficar_matriz_confusion(matriz_confusion, titulo="Matriz de Confusion - Sulfato 3 Datos 5 Capas")

    modelos_metricas = [
        {
            "nombre": "Sulfato Base 2 Capas",
            "ruta_metricas": "modelos_sulfato/sulfato_base_2_capas/metricas.json",
        },
        {
            "nombre": "Sulfato Base 4 Capas",
            "ruta_metricas": "modelos_sulfato/sulfato_base_4_capas/metricas.json",
        },
        {
            "nombre": "Sulfato 3 Datos 5 Capas",
            "ruta_metricas": "modelos_sulfato/sulfato_3_datos_5_capas/metricas.json",
        },
    ]

    df_eficiencia = construir_df_eficiencia_desde_metricas(modelos_metricas)
    graficar_comparacion_eficiencia(df_eficiencia)
