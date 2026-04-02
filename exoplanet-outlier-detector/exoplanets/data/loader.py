"""
loader.py
---------
Descarga y carga de curvas de luz usando lightkurve.
Soporta misiones: Kepler, K2, TESS.
"""

import numpy as np
import pandas as pd


class LightCurveLoader:
    """
    Descarga curvas de luz de estrellas desde MAST usando lightkurve.

    Parameters
    ----------
    mission : str
        Misión espacial: "Kepler", "K2" o "TESS".
    cadence : str
        Cadencia de observación: "long" (30 min) o "short" (1 min).
    """

    SUPPORTED_MISSIONS = ("Kepler", "K2", "TESS")

    def __init__(self, mission: str = "Kepler", cadence: str = "long"):
        if mission not in self.SUPPORTED_MISSIONS:
            raise ValueError(f"Misión '{mission}' no soportada. Opciones: {self.SUPPORTED_MISSIONS}")
        self.mission = mission
        self.cadence = cadence

    def fetch(self, target: str, quarter: int = None) -> pd.DataFrame:
        """
        Descarga la curva de luz de una estrella y la retorna como DataFrame.

        Parameters
        ----------
        target : str
            Identificador de la estrella, ej: "KIC 757076" o "TIC 261136679".
        quarter : int, optional
            Trimestre específico (solo Kepler). Si es None, descarga todos.

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas: time, flux, flux_err.
        """
        try:
            import lightkurve as lk
        except ImportError:
            raise ImportError("Instala lightkurve: pip install lightkurve")

        search_kwargs = {"author": self.mission, "cadence": self.cadence}
        if quarter is not None and self.mission == "Kepler":
            search_kwargs["quarter"] = quarter

        results = lk.search_lightcurve(target, **search_kwargs)

        if len(results) == 0:
            raise ValueError(f"No se encontraron datos para '{target}' en {self.mission}.")

        lc_collection = results.download_all()
        lc = lc_collection.stitch()
        lc = lc.remove_nans().normalize()

        return pd.DataFrame({
            "time": lc.time.value,
            "flux": lc.flux.value,
            "flux_err": lc.flux_err.value if lc.flux_err is not None else np.zeros(len(lc.flux)),
        })

    def fetch_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Carga una curva de luz desde un archivo CSV local.

        Parameters
        ----------
        filepath : str
            Ruta al archivo CSV con columnas: time, flux, flux_err.

        Returns
        -------
        pd.DataFrame
        """
        df = pd.read_csv(filepath)
        required = {"time", "flux"}
        if not required.issubset(df.columns):
            raise ValueError(f"El CSV debe tener las columnas: {required}")
        if "flux_err" not in df.columns:
            df["flux_err"] = 0.0
        return df[["time", "flux", "flux_err"]].dropna()
