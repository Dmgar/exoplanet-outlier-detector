"""
Métodos estadísticos clásicos para detección de outliers en curvas de luz.

Métodos disponibles:
    - zscore    : Desviación estándar (Z-score)
    - iqr       : Rango intercuartil (IQR)
    - mad       : Mediana de desviaciones absolutas (MAD) — robusto al ruido
"""

import numpy as np
import pandas as pd
from typing import Literal


class OutlierDetector:
    """
    Detecta outliers en el flujo de una curva de luz usando métodos estadísticos.

    Parameters
    ----------
    method : str
        Método de detección: "zscore", "iqr" o "mad".
    threshold : float
        Umbral para clasificar como outlier.
        - zscore: número de sigmas (típico: 3.0 – 5.0)
        - iqr: multiplicador del IQR (típico: 1.5 – 3.0)
        - mad: número de MADs (típico: 3.5 – 5.0)
    direction : str
        "both"  → detecta dips y picos (defecto)
        "down"  → solo dips (caída de flujo = tránsito planetario)
        "up"    → solo picos (flares estelares)
    """

    METHODS = ("zscore", "iqr", "mad")

    def __init__(
        self,
        method: Literal["zscore", "iqr", "mad"] = "mad",
        threshold: float = 3.5,
        direction: Literal["both", "down", "up"] = "both",
    ):
        if method not in self.METHODS:
            raise ValueError(f"Método '{method}' no válido. Opciones: {self.METHODS}")
        self.method = method
        self.threshold = threshold
        self.direction = direction
        self._scores: np.ndarray = None

    def fit_predict(self, lc: pd.DataFrame) -> np.ndarray:
        """
        Calcula scores y retorna máscara booleana de outliers.

        Parameters
        ----------
        lc : pd.DataFrame
            DataFrame con columna "flux".

        Returns
        -------
        np.ndarray
            Array booleano True donde hay outlier.
        """
        flux = lc["flux"].values.astype(float)
        scores = self._compute_scores(flux)
        self._scores = scores

        if self.direction == "both":
            mask = np.abs(scores) > self.threshold
        elif self.direction == "down":
            mask = scores < -self.threshold
        else:  # up
            mask = scores > self.threshold

        return mask

    def _compute_scores(self, flux: np.ndarray) -> np.ndarray:
        if self.method == "zscore":
            return (flux - np.mean(flux)) / np.std(flux)

        elif self.method == "iqr":
            q1, q3 = np.percentile(flux, [25, 75])
            iqr = q3 - q1
            return (flux - np.median(flux)) / (iqr + 1e-10)

        elif self.method == "mad":
            median = np.median(flux)
            mad = np.median(np.abs(flux - median))
            # Factor 0.6745 normaliza MAD para que sea consistente con sigma
            return (flux - median) / (0.6745 * mad + 1e-10)

    @property
    def scores(self) -> np.ndarray:
        """Retorna los scores calculados en el último fit_predict."""
        if self._scores is None:
            raise RuntimeError("Ejecuta fit_predict() primero.")
        return self._scores

    def summary(self, lc: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
        """
        Retorna un DataFrame con los puntos detectados como outliers.

        Parameters
        ----------
        lc : pd.DataFrame
            Curva de luz original.
        mask : np.ndarray
            Máscara booleana de outliers.

        Returns
        -------
        pd.DataFrame
        """
        outliers = lc[mask].copy()
        outliers["score"] = self._scores[mask]
        outliers["direction"] = np.where(outliers["score"] < 0, "dip", "peak")
        return outliers.sort_values("score")
