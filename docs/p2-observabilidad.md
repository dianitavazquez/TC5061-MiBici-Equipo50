# Observabilidad del champion P2 — octubre de 2026

## Conjuntos comparados

El archivo fuente del modelo cubre del **1-oct-2025 al 30-sep-2026**, usando septiembre de
2025 solo como contexto para construir rezagos. Sin embargo, P2 aplica una partición
cronológica 80/20: la referencia que realmente ajustó el champion comprende del
**1-oct-2025 al 19-jul-2026** (2,637,387 filas estación-hora y 436 estaciones). Del
20-jul-2026 al 30-sep-2026 se reservó para evaluación y no forma parte del ajuste.

El conjunto actual comprende del **1-oct-2026 al 31-oct-2026** (348,360 filas y 469
estaciones). Es el primer mes completo posterior a la ventana original y, por ello, representa
el comportamiento más reciente disponible. Septiembre de 2026 se usa únicamente como
contexto para calcular rezagos de octubre y después se elimina del conjunto actual.

Esta separación evita contaminar la referencia, mantiene intacto el manifiesto del champion y
permite atribuir los cambios a datos posteriores al entrenamiento.

## Método

- Evidently 0.7.23 con `DataDriftPreset` y `DataSummaryPreset`.
- PSI por variable con umbral **0.20**; valores superiores se marcan como drift.
- Comparación de las 22 variables de entrada y los 2 targets de P2.
- Muestra determinista de 100,000 filas por conjunto para el reporte; los controles de calidad
  se calculan sobre la totalidad de ambos conjuntos.
- El drift global de Evidently se activa si al menos 50 % de las columnas presenta drift.
- Evaluación adicional del champion versión 2 sobre todas las filas de octubre, aprovechando
  que las etiquetas demoradas ya pueden reconstruirse desde los viajes observados.

## Calidad de datos

No se encontraron valores faltantes ni llaves `station_id`–`hour` duplicadas en ninguno de los
dos conjuntos. Octubre cubre las 744 horas del mes y sus 348,360 filas corresponden a 469
estaciones. `month_sin` y `month_cos` son constantes en octubre por diseño, no por un error de
calidad: representan un único mes.

## Resultados de drift

Evidently detectó drift en **3 de 25 columnas (12 %)**. Como la proporción es menor al 50 %,
no se activa drift global del dataset, pero las alertas individuales requieren interpretación.

| Variable | PSI | Interpretación |
|---|---:|---|
| `month_sin` | 8.346 | Drift esperado: octubre es un solo mes frente a una referencia que contiene varias estaciones del año. |
| `month_cos` | 8.207 | Mismo efecto estacional; no indica por sí solo degradación. |
| `station_id` | 1.524 | Drift operativo: cambió la mezcla de estaciones y aparecieron categorías no vistas durante el ajuste. |

Las variables de demanda están por debajo del umbral, aunque las medias de 168 horas son las
más cercanas (`departures_mean_168h=0.140`, `arrivals_mean_168h=0.129`). Conviene seguirlas en
los meses siguientes: todavía no justifican una alerta individual, pero sugieren mayor actividad
reciente. Capacidad (`0.027`), rezagos de una hora, hora, día de semana y targets permanecen
estables según PSI.

### Cambio de estaciones

Octubre contiene **34 estaciones no observadas en la referencia de ajuste**, equivalentes al
**7.1 % de las filas actuales**; una estación de la referencia no aparece en octubre. El pipeline
usa `OneHotEncoder(handle_unknown="ignore")`, por lo que no falla técnicamente, pero representa
cada estación desconocida con todos los indicadores de estación en cero. Sus predicciones no
incluyen un efecto específico aprendido para esa estación y pueden estar peor calibradas.

## Implicación para el desempeño

Con las etiquetas de octubre, el F1 macro observado es **0.1806**, frente a **0.2057** en la
evaluación original del champion: una reducción relativa de **12.2 %**.

| Target | F1 evaluación original | F1 octubre | Cambio |
|---|---:|---:|---:|
| Falta de bicicletas | 0.1718 | 0.1219 | −29.1 % |
| Falta de anclajes | 0.2395 | 0.2394 | −0.1 % |

La degradación se concentra en `risk_no_bikes`: la precisión cae a 0.094 y el recall sube a
0.173, consistente con un cambio de calibración y más falsos positivos. La prevalencia también
aumenta respecto a la referencia de ajuste: +43.9 % para falta de bicicletas y +40.9 % para
falta de anclajes. El PSI de los targets sigue bajo porque las clases positivas son muy raras;
por eso no debe usarse como sustituto de las métricas supervisadas.

## Decisión recomendada

No reemplazar automáticamente al champion por un solo mes. Sí debe abrirse una alerta y:

1. segmentar las métricas entre estaciones conocidas y las 34 no vistas;
2. revisar umbrales de decisión/calibración para `risk_no_bikes`;
3. repetir el monitoreo con noviembre y diciembre para separar estacionalidad de cambio
   persistente;
4. iniciar un candidato de reentrenamiento que incluya octubre si el F1 macro continúa por
   debajo del baseline o se mantiene la degradación de falta de bicicletas.

## Artefactos y reproducción

- Reporte interactivo: `reports/evidently/p2/drift_2026_10.html`
- Resultado Evidently: `reports/evidently/p2/drift_2026_10.json`
- Resumen auditable: `references/observabilidad/p2_drift_2026_10.json`
- Ejecución: `make monitor-p2` o `mlflow run . -e monitor-p2`
