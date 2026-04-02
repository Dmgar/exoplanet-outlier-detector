# 🪐 Exoplanet Outlier Detector

Detección de exoplanetas candidatos mediante análisis de **outliers estadísticos** en series de tiempo fotométricas (curvas de luz) y datos de velocidad radial.

Este proyecto explora métodos no convencionales para identificar señales anómalas que podrían corresponder a tránsitos planetarios, usando datos públicos de misiones como **Kepler**, **K2** y **TESS**.

---

## 🎯 Motivación

Los métodos clásicos de detección (Box Least Squares, BLS) buscan patrones periódicos específicos. Este proyecto se enfoca en un enfoque complementario: **¿qué señales son estadísticamente inusuales?** Algunos exoplanetas con órbitas excéntricas, tránsitos únicos (monotransits) o sistemas multiplanetarios complejos pueden ser difíciles de capturar con métodos periódicos tradicionales.

---

## 🔭 Fuentes de datos

| Dataset | Descripción | Acceso |
|---|---|---|
| [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) | Catálogo oficial de exoplanetas confirmados | API REST |
| [MAST / lightkurve](https://docs.lightkurve.org/) | Curvas de luz de Kepler/TESS | `lightkurve` Python |
| [Exoplanet Archive TAP](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html) | Consultas SQL via ADQL | Requests |

---

## 🗂️ Estructura del proyecto

```
exoplanet-outlier-detector/
│
├── exoplanets/                  # Paquete principal
│   ├── data/                    # Carga y descarga de datos
│   │   ├── loader.py            # Descarga de curvas de luz (lightkurve)
│   │   └── preprocessor.py     # Limpieza y normalización
│   │
│   ├── detection/               # Algoritmos de detección
│   │   ├── statistical.py       # Métodos estadísticos (IQR, Z-score, MAD)
│   │   ├── isolation_forest.py  # Isolation Forest para anomalías
│   │   └── autoencoder.py       # Autoencoder (próximamente)
│   │
│   ├── visualization/           # Gráficas y reportes
│   │   └── plots.py             # Curvas de luz, scatter plots, heatmaps
│   │
│   └── utils/                   # Utilidades compartidas
│       └── metrics.py           # Métricas de evaluación
│
├── notebooks/                   # Exploración y demos
│   ├── 01_data_exploration.ipynb
│   ├── 02_outlier_methods.ipynb
│   └── 03_case_study_kepler.ipynb
│
├── tests/                       # Pruebas unitarias
│   ├── test_loader.py
│   ├── test_statistical.py
│   └── test_preprocessing.py
│
├── data/
│   ├── raw/                     # Datos originales (no subir a git)
│   └── processed/               # Datos transformados
│
├── docs/                        # Documentación adicional
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/TU_USUARIO/exoplanet-outlier-detector.git
cd exoplanet-outlier-detector

# Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## ⚡ Uso rápido

```python
from exoplanets.data.loader import LightCurveLoader
from exoplanets.detection.statistical import OutlierDetector

# Descargar curva de luz de una estrella Kepler
loader = LightCurveLoader(mission="Kepler")
lc = loader.fetch("KIC 757076")

# Detectar outliers con Z-score
detector = OutlierDetector(method="zscore", threshold=3.5)
outliers = detector.fit_predict(lc)

print(f"Outliers encontrados: {outliers.sum()}")
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📚 Referencias

- Aigrain & Irwin (2004) — *Practical planet prospecting*
- Shallue & Vanderburg (2018) — *Identifying Exoplanets with Deep Learning*
- [lightkurve documentation](https://docs.lightkurve.org/)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

---

## 📄 Licencia

MIT License — ver [LICENSE](LICENSE) para detalles.
