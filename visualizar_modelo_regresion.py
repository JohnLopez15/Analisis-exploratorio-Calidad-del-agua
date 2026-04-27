import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def visualizar_modelo_regresion(
    ruta_modelo: str | Path = "modelos_sulfato_regresion/sulfato_reg_4_capas",
) -> None:
    ruta_modelo = Path(ruta_modelo)
    
    # Leer predicciones
    ruta_pred = ruta_modelo / "predicciones_test.csv"
    if not ruta_pred.exists():
        raise FileNotFoundError(f"No se encontró {ruta_pred}")
    
    df_pred = pd.read_csv(ruta_pred)
    
    # Leer métricas para obtener info del modelo
    ruta_metricas = ruta_modelo / "metricas.json"
    metricas = {}
    if ruta_metricas.exists():
        with open(ruta_metricas, "r", encoding="utf-8") as f:
            metricas = json.load(f)
    
    # Convertir columnas a numéricas si existen
    if "y_real" in df_pred.columns:
        df_pred["y_real"] = pd.to_numeric(df_pred["y_real"], errors="coerce")
    if "y_predicha" in df_pred.columns:
        df_pred["y_predicha"] = pd.to_numeric(df_pred["y_predicha"], errors="coerce")
    if "clase_real_idx" in df_pred.columns:
        df_pred["clase_real_idx"] = pd.to_numeric(df_pred["clase_real_idx"], errors="coerce")
    if "error_abs" in df_pred.columns:
        df_pred["error_abs"] = pd.to_numeric(df_pred["error_abs"], errors="coerce")
    
    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ============ GRÁFICO 1: Violín - Distribución de predicciones por categoría real ============
    # Queremos que el eje X muestre la categoría real de Sulfato y el eje Y los valores predichos
    if "y_predicha" in df_pred.columns:
        # Determinar columna de categoría real: preferir 'clase_real', luego 'clase_real_idx', luego construir a partir de 'y_real'
        if "clase_real" in df_pred.columns:
            cat_col = "clase_real"
        elif "clase_real_idx" in df_pred.columns:
            cat_col = "clase_real_idx"
        elif "y_real" in df_pred.columns:
            cat_col = "y_real"
        else:
            cat_col = None

        if cat_col is not None:
            cols_needed = [col for col in [cat_col, "y_predicha"] if col in df_pred.columns]
            df_plot = df_pred[cols_needed].dropna().copy()

            # Intentar obtener mapeo desde metricas['clases_discretas']
            clases_discretas = metricas.get("clases_discretas") if isinstance(metricas, dict) else None

            # Crear etiqueta numérica de la categoría ordenada según 'clases_discretas'
            if clases_discretas:
                clases_arr = np.array(clases_discretas, dtype=float)

                if cat_col in ("clase_real", "clase_real_idx"):
                    # Asumir que la columna contiene índices de clase (0..n-1)
                    try:
                        idxs = df_plot[cat_col].astype(int)
                        df_plot["categoria_num"] = idxs.map(lambda i: float(clases_arr[int(i)]) )
                    except Exception:
                        # Si no son índices, intentar mapear valores directos existentes
                        df_plot["categoria_num"] = pd.to_numeric(df_plot[cat_col], errors="coerce")
                else:
                    # cat_col == 'y_real': asignar la clase discreta más cercana al y_real
                    def nearest_class(val):
                        if pd.isna(val):
                            return np.nan
                        i = int(np.abs(clases_arr - float(val)).argmin())
                        return float(clases_arr[i])

                    df_plot["categoria_num"] = df_plot[cat_col].apply(nearest_class)

                # Preparar etiquetas ordenadas (ascendente)
                unique_nums = np.sort(df_plot["categoria_num"].dropna().unique())
                labels = [f"{v:.2f}" for v in unique_nums]
                # Crear columna de etiqueta para el eje X (cadena) y forzar orden
                num_to_label = {v: f"{v:.2f}" for v in unique_nums}
                df_plot["categoria_label"] = df_plot["categoria_num"].map(num_to_label)

                # Usar violinplot con orden ascendente de las clases numéricas
                sns.violinplot(x="categoria_label", y="y_predicha", data=df_plot, ax=ax1, inner="quartile", palette="Set2", order=labels)
                ax1.set_xlabel("Valor Real (clase) de Sulfato", fontsize=12, fontweight="bold")
                ax1.set_ylabel("Valor Predicho de Sulfato", fontsize=12, fontweight="bold")
                ax1.set_title("Distribución de Predicciones por Clase Real (Violín, clases ordenadas)", fontsize=14, fontweight="bold")
                ax1.grid(True, alpha=0.3, linestyle="--")
                for label in ax1.get_xticklabels():
                    label.set_rotation(30)
            else:
                # Sin 'clases_discretas', caer al comportamiento anterior (mostrar la columna tal cual)
                df_plot["categoria_real"] = df_plot[cat_col].astype(str)
                sns.violinplot(x="categoria_real", y="y_predicha", data=df_plot, ax=ax1, inner="quartile", palette="Set2")
                ax1.set_xlabel("Categoría Real de Sulfato", fontsize=12, fontweight="bold")
                ax1.set_ylabel("Valor Predicho de Sulfato", fontsize=12, fontweight="bold")
                ax1.set_title("Distribución de Predicciones por Categoría Real (Violín)", fontsize=14, fontweight="bold")
                ax1.grid(True, alpha=0.3, linestyle="--")
                for label in ax1.get_xticklabels():
                    label.set_rotation(30)
        else:
            ax1.text(0.5, 0.5, "No hay columna de categoría real disponible.", transform=ax1.transAxes, ha="center")
    
    # ============ GRÁFICO 2: Histograma de error absoluto en unidades + línea acumulada ============
    # Mostrar error absoluto real (unidades), no porcentaje relativo al valor real
    if "y_predicha" in df_pred.columns:
        if "error_abs" in df_pred.columns:
            errores_series = df_pred["error_abs"].dropna().abs()
        elif "y_real" in df_pred.columns:
            errores_series = (df_pred["y_predicha"] - df_pred["y_real"]).abs().dropna()
        else:
            errores_series = pd.Series(dtype=float)

        errores = errores_series.values
        if len(errores) > 0:
            total_muestras = len(errores)

            # Crear histograma en unidades (x = error absoluto)
            counts, bins, patches = ax2.hist(
                errores, bins=30, color="steelblue", edgecolor="black", alpha=0.7, density=False
            )

            # Convertir counts a porcentaje del total de muestras para eje Y principal
            counts_pct = (counts / total_muestras) * 100
            for count, patch in zip(counts_pct, patches):
                patch.set_height(count)

            # Ajustar límites del eje Y principal desde 0 hasta un poco por encima del máximo
            max_pct = counts_pct.max() if len(counts_pct) > 0 else 0
            ax2.set_ylim(0, max_pct * 1.05 if max_pct > 0 else 1)
            # Definir ticks simples desde 0 hasta el máximo
            try:
                yticks = np.linspace(0, ax2.get_ylim()[1], num=6)
                ax2.set_yticks(yticks)
            except Exception:
                pass

            # Segundo eje para la curva acumulada (porcentaje acumulado 0-100)
            ax2_twin = ax2.twinx()
            sorted_errores = np.sort(errores)
            acumulado_pct = np.arange(1, len(sorted_errores) + 1) / len(sorted_errores) * 100
            ax2_twin.plot(sorted_errores, acumulado_pct, color="red", linewidth=2.5, label="Acumulado %", zorder=5)
            ax2_twin.set_ylabel("Porcentaje Acumulado (%)", fontsize=12, fontweight="bold", color="red")
            ax2_twin.tick_params(axis="y", labelcolor="red")
            ax2_twin.set_ylim(0, 100)
            ax2_twin.set_yticks(np.linspace(0, 100, 11))

            mean_error = np.mean(errores)
            median_error = np.median(errores)

            ax2.axvline(mean_error, color="green", linestyle="--", linewidth=2.5, label=f"Media: {mean_error:.4f}", zorder=4)
            ax2.axvline(median_error, color="orange", linestyle="--", linewidth=2.5, label=f"Mediana: {median_error:.4f}", zorder=4)

            ax2.set_xlabel("Error Absoluto (unidades)", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Porcentaje de Muestras (%)", fontsize=12, fontweight="bold")
            ax2.set_title("Distribución del Error Absoluto (unidades) + Acumulado", fontsize=14, fontweight="bold")

            # Combinar leyendas
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

            ax2.grid(True, alpha=0.3, linestyle="--", axis="y")

            stats_text = (
                f"Muestras: {len(errores)}\n"
                f"Media: {mean_error:.6f}\n"
                f"Mediana: {median_error:.6f}\n"
                f"Mín: {np.min(errores):.6f}\n"
                f"Máx: {np.max(errores):.6f}\n"
                f"P25: {np.percentile(errores, 25):.6f}\n"
                f"P75: {np.percentile(errores, 75):.6f}\n"
                f"Desv. Est.: {np.std(errores):.6f}"
            )
            ax2.text(0.98, 0.70, stats_text, transform=ax2.transAxes, fontsize=9, verticalalignment="top", horizontalalignment="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    plt.tight_layout()
    
    # Guardar figura
    ruta_salida = ruta_modelo / "visualizacion.png"
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    print(f"Visualización guardada en: {ruta_salida}")
    
    plt.show()


if __name__ == "__main__":
    visualizar_modelo_regresion("modelos_sulfato_regresion/sulfato_reg_4_capas")
