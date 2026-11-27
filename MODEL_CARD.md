# Model Card — MiBici P2 Risk Classifier

## Identificación y estado

| Campo | Valor |
|---|---|
| Modelo registrado | `mibici-equipo50-p2-riesgo` |
| Versión en producción | 2 |
| Alias de producción | `champion` |
| Run de origen | `8981c76434b6469eb96bdfc45d261376` |
| Challenger | versión 3, alias `challenger` |
| Equipo responsable | Equipo 50 |
| Última evaluación documentada | octubre de 2026 |

El estado operativo se representa con aliases de MLflow: `champion` identifica la versión
aprobada para producción y `challenger` la candidata. Los aliases reemplazan las etapas
obsoletas del Model Registry. `make bootstrap-registry` reproduce este estado en un Registry
vacío y conserva como tags la versión y el run originales.

## Propósito y uso previsto

Clasificador multisalida que estima, para cada estación y con horizonte de una hora:

- riesgo de falta de bicicletas (`risk_no_bikes`);
- riesgo de falta de anclajes (`risk_no_docks`).

Su salida apoya la priorización del equipo de rebalanceo. Debe combinarse con capacidad
operativa, inventario en tiempo real y criterio humano; no constituye una orden automática.

### Usos fuera de alcance

- decisiones sobre personas, tarifas, sanciones o acceso al servicio;
- predicciones fuera de la red y periodos representados por los datos de MiBici;
- uso como estimador exacto del inventario disponible;
- promoción automática de modelos sin evaluación temporal, A/B y revisión responsable.

## Datos y variables

La referencia de entrenamiento usa datos abiertos de viajes entre octubre de 2025 y julio de
2026, después de aplicar una partición cronológica 80/20. Septiembre de 2025 se utiliza como
contexto para rezagos. Los manifiestos versionados contienen conteos y hashes de las fuentes.
No se usan atributos personales.

Las variables incluyen estación, capacidad, hora, día de semana, estacionalidad, rezagos de
una hora y medias móviles de 3, 24 y 168 horas. Todas las variables históricas se desplazan
antes de calcularse para evitar fuga temporal.

## Modelo y selección

El modelo es una regresión logística multisalida con preprocesamiento de variables numéricas
y categóricas. La selección compara baseline y candidatos en MLflow usando macro-F1 como
métrica primaria, debido al fuerte desbalance de ambas etiquetas.

La versión 2 fue seleccionada y registrada con linaje hacia:

- experimento y run de entrenamiento;
- commit de Git;
- versión de datos;
- configuración, métricas y artefactos de evaluación.

## Evaluación

| Periodo / modelo | Macro-F1 | F1 sin bicicletas | F1 sin anclajes |
|---|---:|---:|---:|
| Evaluación original del champion v2 | 0.2057 | 0.1718 | 0.2395 |
| Octubre de 2026, champion v2 | 0.1806 | 0.1219 | 0.2394 |
| Octubre de 2026, challenger v3 | 0.0294 | 0.0232 | 0.0356 |

El A/B test pareado por estación obtuvo para el challenger una diferencia observada de
macro-F1 de **−0.1512**, IC 95% **[−0.1909, −0.0752]** y probabilidad de mejora de 0.001.
Por ello, no se promovió y la versión 2 continúa como champion.

## Monitoreo y drift

El reporte Evidently de octubre de 2026 detectó drift en 3 de 25 columnas (12%):
`month_sin`, `month_cos` y `station_id`. No se activó drift global, pero aparecieron 34
estaciones no observadas durante el ajuste, equivalentes al 7.1% de las filas actuales.

La reducción relativa del macro-F1 frente a la evaluación original fue aproximadamente
12.2%, concentrada en el riesgo de falta de bicicletas. Esto genera una alerta de revisión,
pero no justifica reemplazar automáticamente el champion con un único mes.

Artefactos:

- `docs/p2-observabilidad.md`;
- `reports/evidently/p2/drift_2026_10.html`;
- `references/observabilidad/p2_drift_2026_10.json`;
- `references/ab_testing/p2_ab_2026_10.json`.

## Limitaciones y riesgos

- Los eventos positivos son muy poco frecuentes; accuracy no es una métrica útil.
- Estaciones nuevas son codificadas sin un efecto específico aprendido.
- La estacionalidad mensual puede producir drift esperado y requiere interpretación.
- Las etiquetas son proxies de presión de demanda, no lecturas directas de inventario.
- Cambios de infraestructura, capacidad o comportamiento pueden degradar la calibración.
- El desempeño local no garantiza el mismo resultado con otra ventana o ciudad.

No se evalúa equidad demográfica porque el modelo no usa datos personales. Como control de
impacto territorial deben revisarse métricas por estación y evitar que errores persistentes
concentren el servicio de rebalanceo en una parte de la red.

## Criterios de operación y reentrenamiento

1. Mantener el alias `champion` mientras un challenger no supere el límite inferior del IC
   95% de macro-F1 y no reduzca recall más de 0.05.
2. Revisar mensualmente drift, estaciones nuevas, macro-F1, precisión y recall cuando las
   etiquetas demoradas estén disponibles.
3. Abrir candidato de reentrenamiento si la degradación persiste dos periodos, aparece drift
   en variables de demanda o aumenta materialmente la cobertura de estaciones no vistas.
4. Toda promoción requiere actualizar el alias en MLflow, el modelo empaquetado, este Model
   Card y las pruebas de API, carga, drift y A/B.

## Reproducción

```bash
docker compose up --build -d --wait
curl http://localhost:8000/model-info
make load-test
make monitor-p2
```

El primer comando inicializa MLflow, registra champion y challenger con sus aliases, y levanta
la API. El monitoreo completo requiere las fuentes abiertas de entrenamiento y octubre.
