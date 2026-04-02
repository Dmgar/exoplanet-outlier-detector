"""
plots.py
--------
Funciones de visualización para curvas de luz y resultados de detección.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_lightcurve(
    lc: pd.DataFrame,
    outlier_mask: np.ndarray = None,
    title: str = "Curva de Luz",
    figsize: tuple = (14, 5),
    save_path: str = None,
):
    """
    Grafica una curva de luz y resalta los outliers detectados.

    Parameters
    ----------
    lc : pd.DataFrame
        DataFrame con columnas: time, flux.
    outlier_mask : np.ndarray, optional
        Máscara booleana indicando outliers.
    title : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura.
    save_path : str, optional
        Ruta donde guardar la imagen. Si es None, muestra en pantalla.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # Curva base
    ax.plot(
        lc["time"], lc["flux"],
        color="#4a9eff", linewidth=0.6, alpha=0.8, label="Flujo"
    )

    # Outliers
    if outlier_mask is not None and outlier_mask.any():
        dips = outlier_mask & (lc["flux"] < lc["flux"].median())
        peaks = outlier_mask & (lc["flux"] >= lc["flux"].median())

        ax.scatter(
            lc["time"][dips], lc["flux"][dips],
            color="#ff4757", s=25, zorder=5, label=f"Dips ({dips.sum()})"
        )
        ax.scatter(
            lc["time"][peaks], lc["flux"][peaks],
            color="#ffa502", s=25, zorder=5, label=f"Peaks ({peaks.sum()})"
        )

    ax.set_xlabel("Tiempo (BKJD)", color="#8b949e", fontsize=11)
    ax.set_ylabel("Flujo Normalizado", color="#8b949e", fontsize=11)
    ax.set_title(title, color="#e6edf3", fontsize=13, pad=12)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura guardada en: {save_path}")
    else:
        plt.show()

    return fig, ax


def plot_score_distribution(
    scores: np.ndarray,
    threshold: float = 3.5,
    figsize: tuple = (10, 4),
):
    """
    Histograma de los scores de anomalía con líneas de umbral.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax.hist(scores, bins=80, color="#4a9eff", alpha=0.7, edgecolor="none")
    ax.axvline(-threshold, color="#ff4757", linestyle="--", linewidth=1.5, label=f"Umbral ±{threshold}")
    ax.axvline(threshold, color="#ff4757", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Score de Anomalía", color="#8b949e", fontsize=11)
    ax.set_ylabel("Frecuencia", color="#8b949e", fontsize=11)
    ax.set_title("Distribución de Scores", color="#e6edf3", fontsize=13)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    plt.tight_layout()
    plt.show()
    return fig, ax
