"""
test_statistical.py
-------------------
Tests unitarios para el módulo de detección de outliers estadísticos.
"""

import numpy as np
import pandas as pd
import pytest
from exoplanets.detection.statistical import OutlierDetector


def make_lightcurve(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Genera una curva de luz sintética con outliers artificiales."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 100, n)
    flux = 1.0 + rng.normal(0, 0.005, n)  # ruido Gaussiano pequeño

    # Insertar dips artificiales (simulando tránsitos)
    flux[100:103] -= 0.05   # dip profundo
    flux[250:252] -= 0.03   # dip moderado

    return pd.DataFrame({"time": time, "flux": flux, "flux_err": np.full(n, 0.005)})


class TestOutlierDetector:

    def test_init_valid_method(self):
        det = OutlierDetector(method="zscore")
        assert det.method == "zscore"

    def test_init_invalid_method(self):
        with pytest.raises(ValueError, match="no válido"):
            OutlierDetector(method="unknown")

    def test_fit_predict_returns_bool_array(self):
        lc = make_lightcurve()
        det = OutlierDetector(method="mad", threshold=3.5)
        mask = det.fit_predict(lc)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(lc)

    def test_detects_artificial_dips(self):
        lc = make_lightcurve()
        det = OutlierDetector(method="mad", threshold=3.5, direction="down")
        mask = det.fit_predict(lc)
        # Los índices 100-102 deben ser detectados
        assert mask[100:103].any(), "No detectó el dip en índices 100-102"

    def test_zscore_method(self):
        lc = make_lightcurve()
        det = OutlierDetector(method="zscore", threshold=3.0)
        mask = det.fit_predict(lc)
        assert mask.any()

    def test_iqr_method(self):
        lc = make_lightcurve()
        det = OutlierDetector(method="iqr", threshold=2.0)
        mask = det.fit_predict(lc)
        assert mask.any()

    def test_scores_property_before_fit(self):
        det = OutlierDetector()
        with pytest.raises(RuntimeError):
            _ = det.scores

    def test_scores_property_after_fit(self):
        lc = make_lightcurve()
        det = OutlierDetector()
        det.fit_predict(lc)
        assert len(det.scores) == len(lc)

    def test_summary_returns_dataframe(self):
        lc = make_lightcurve()
        det = OutlierDetector(method="mad", threshold=3.5)
        mask = det.fit_predict(lc)
        summary = det.summary(lc, mask)
        assert isinstance(summary, pd.DataFrame)
        assert "score" in summary.columns
        assert "direction" in summary.columns

    def test_direction_down_no_peaks(self):
        lc = make_lightcurve()
        det = OutlierDetector(method="mad", threshold=3.5, direction="down")
        mask = det.fit_predict(lc)
        # Con direction="down" no deben detectarse picos positivos
        if mask.any():
            assert (lc["flux"][mask] < lc["flux"].median()).all()
