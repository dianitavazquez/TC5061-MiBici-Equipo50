# Arquitectura propuesta

```mermaid
flowchart LR
    A[Portal MiBici<br/>CSV mensuales] --> B[Google Drive<br/>copia pública]
    B --> C[Ingesta Python<br/>urllib + SHA-256]
    C --> D[(data/raw<br/>CSV inmutables + manifest)]
    D --> E[Pandas<br/>limpieza y eventos]
    E --> F[(Panel estación-hora<br/>features rezagadas)]
    F --> G[scikit-learn Pipeline<br/>split temporal + entrenamiento]
    G --> H[Evaluación futura<br/>F1, precision, recall, PR-AUC]
    G --> I[(MLflow Tracking<br/>parámetros, métricas, modelo)]
    H --> I
    J[Git<br/>código + commit] --> I
```

| Componente | Herramienta | Justificación |
|---|---|---|
| Fuente | Portal MiBici + Drive público | Fuente oficial y copia compartida por el equipo |
| Ingesta | Python estándar | Automatizable, portable y sin credenciales para archivos públicos |
| Almacenamiento | CSV crudo local + manifest SHA-256 | Conserva el original, evita subir datos pesados y permite identificar la versión exacta |
| Procesamiento | pandas | Adecuado al volumen del baseline y compatible con los CSV/tablas del proyecto |
| Modelado | Pipeline de scikit-learn | Encapsula transformaciones y evita ajustarlas con datos de prueba |
| Tracking | MLflow local | Registra parámetros, métricas, artefactos, modelo y commit de cada ejecución |
| Versionado | Git/GitHub | Revisión colaborativa y trazabilidad del código en `feature/classifier-aavm` |

Para escalar, el almacenamiento procesado puede migrar a Parquet y la agregación a
DuckDB/Spark sin cambiar el contrato estación-hora ni el tracking de MLflow.
