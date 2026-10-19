# Arquitectura propuesta

```mermaid
flowchart LR
    A[Portal MiBici<br/>CSV mensuales] --> B[Google Drive<br/>copia pública]
    B --> C[Ingesta Python<br/>urllib + SHA-256]
    C --> D[(data/raw<br/>CSV inmutables)]
    D --> E[Pandas<br/>normalización y limpieza]
    E --> F[(Parquet + manifiesto<br/>filas y SHA-256)]
    F --> K[Features reutilizables<br/>P1, P2 y P3]
    K --> G[scikit-learn Pipeline<br/>split temporal + entrenamiento]
    G --> H[Evaluación futura<br/>F1, precision, recall, PR-AUC]
    G --> I[(MLflow Tracking<br/>parámetros, métricas, modelo)]
    H --> I
    J[Git<br/>código + commit] --> I
```

| Componente | Herramienta | Justificación |
|---|---|---|
| Fuente | Portal MiBici + Drive público | Fuente oficial y copia compartida por el equipo |
| Ingesta | Python estándar | Automatizable, portable y sin credenciales para archivos públicos |
| Almacenamiento | CSV crudo + Parquet procesado + manifiesto SHA-256 | Conserva la fuente, escala a millones de filas e identifica la versión exacta |
| Procesamiento | pandas | Adecuado al volumen del baseline y compatible con los CSV/tablas del proyecto |
| Modelado | Pipeline de scikit-learn | Encapsula transformaciones y evita ajustarlas con datos de prueba |
| Tracking | MLflow local | Registra parámetros, métricas, artefactos, modelo y commit de cada ejecución |
| Versionado | Git/GitHub Flow | Ramas cortas, Conventional Commits, Pull Requests y revisión cruzada antes de `main` |

La ingesta procesada ya usa Parquet. Para escalar más, la agregación puede migrar a
DuckDB/Spark sin cambiar el contrato de features ni el tracking de MLflow.
