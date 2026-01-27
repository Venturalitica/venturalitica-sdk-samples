# 🖼️ Vision Fairness (FairFace)

Este escenario demuestra la integración de Venturalítica en un pipeline de Computer Vision basado en **PyTorch Lightning**. Se centra en el cumplimiento de los artículos 10 y 15 de la **EU AI Act**.

## 🚀 Usage (Sequential Path)

Siguiendo el estilo de "Glass Box", hemos descompuesto el proceso en 3 pasos lógicos:

### 0. Prepare Data
Descarga el dataset FairFace completo (~86k imágenes).
```bash
uv run prepare_data.py
```

### 1. Data Governance (Article 10)
Analiza el sesgo y la representación en los datos *antes* de entrenar.
```bash
# POC Mode (Fast)
VL_DATA_SCALE=POC uv run 01_data_governance.py

# Full Audit (Demo)
VL_DATA_SCALE=FULL uv run 01_data_governance.py
```

### 2. Model Training & Fairness (Article 15)
Entrena un modelo ResNet18 y audita el sesgo del algoritmo durante la validación.
```bash
# POC Mode (~1 min)
VL_DATA_SCALE=POC uv run 02_model_training.py

# Full Training (GPU recommended)
VL_DATA_SCALE=FULL uv run 02_model_training.py
```

### 3. Advanced Metrics Deep Dive
Explora métricas complejas como **Predictive Parity** usando datos simulados.
```bash
uv run 03_advanced_metrics.py
```

## 🛡️ Governance & MLOps
- **Dashboard**: Ejecuta `uv run venturalitica ui` para ver el mapa regulatorio.
- **MLflow**: Snapshots de políticas en `http://localhost:5000`.
- **WandB**: Métricas de gobernanza en tiempo real.
