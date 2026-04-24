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
    
direccion="modelos_sulfato/sulfato_base_2_capas/matriz_confusion.csv"
matriz_confusion = leer_matriz_desde_csv(direccion)
graficar_matriz_confusion(matriz_confusion, titulo="Matriz de Confusión - Sulfato Base 2 Capas")
