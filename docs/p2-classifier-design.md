# P2 — Riesgo de una estación en la hora siguiente

## 1. Problema y decisión operativa

Se generan dos predicciones binarias para cada **estación-hora**, al inicio de la hora y
con horizonte de **una hora**:

1. riesgo de presión por falta de bicicletas; y
2. riesgo de presión por falta de anclajes.

El equipo de rebalanceo usaría las probabilidades para priorizar estaciones y decidir
dónde retirar o entregar bicicletas antes de la siguiente hora. No se usan como una
afirmación de disponibilidad real: los datos no incluyen el inventario inicial ni
snapshots de bicicletas/anclajes disponibles.

## 2. Datos

### Fuente, rango y tamaño real

- Publicador: [portal oficial de datos abiertos de MiBici](https://mibici.net/es/datos-abiertos).
- Copia de trabajo: [carpeta pública de Google Drive](https://drive.google.com/drive/folders/1XdbZGmBX2BnosPNioK7jD1kwKoSEvwG4).
- Archivos: `datos_abiertos_YYYY_MM.csv` de
  [septiembre de 2025](https://drive.google.com/file/d/1ymRudTROvAiyuLR-4ozg5eAaOsP_jldO/view)
  a [septiembre de 2026](https://drive.google.com/file/d/1jyptssMIcu2FMO49-6Nwi8MAckUd-mfj/view), y
  [nomenclatura de estaciones](https://drive.google.com/file/d/1rMLYaUxmXNOfp60UiXCAaCbOh7GnD6FJ/view).
- Descarga real: **13 archivos mensuales, 4,874,666 viajes y 471 estaciones observadas**.
  La nomenclatura tiene 484 estaciones (480 con capacidad positiva).
- Ventana de modelado: **1 de octubre de 2025 a 30 de septiembre de 2026**, doce meses
  completos y recientes. Septiembre de 2025 se conserva como contexto para construir
  rezagos sin rellenar el inicio con información artificial.
- Población de la ventana: **4,497,877 viajes y 471 estaciones observadas** antes de
  limpieza; se retiran 295,263 viajes cuyo origen y destino coinciden, según la decisión
  del TDD. También se excluyen eventos sin capacidad vigente.
- Versión reproducible obtenida el 15-10-2026: `928ac1753461`. El comando de ingesta
  genera `data/raw/manifest.json` con URL y SHA-256 de cada archivo; los datos crudos no
  se versionan en Git.


### Variables originales y tratamiento

| Archivo | Variable | Uso |
|---|---|---|
| viajes | `Inicio_del_viaje` | evento de salida y calendario |
| viajes | `Fin_del_viaje` | evento de llegada |
| viajes | `Origen_Id` | estación de salida; no identifica a una persona |
| viajes | `Destino_Id` | estación de llegada |
| nomenclatura | `id` | unión estación-capacidad y efecto categórico de estación |
| nomenclatura | `dpcapacity` | normalización del desbalance y feature de capacidad |

`Viaje_Id`, `Usuario_Id`, `Genero` y `Año_de_nacimiento` no se cargan para P2. No son
necesarios para la operación de una estación; `Usuario_Id` es un identificador
pseudónimo y género/año de nacimiento son cuasi-identificadores que aumentan el riesgo
de reidentificación y de sesgo sin aportar una señal operacional justificable.

### Construcción de los objetivos

En cada estación y hora se ordenan los eventos. Una salida aporta `+1` y una llegada
`-1`; eventos con la misma marca temporal se suman antes de acumular. Sean `C_s` la
capacidad y `D_s,h(t)` el desbalance acumulado desde el inicio de la hora:

- `risk_no_bikes = 1` si `max_t D_s,h(t) >= α C_s`;
- `risk_no_docks = 1` si `-min_t D_s,h(t) >= α C_s`.

El baseline usa `α=0.5`. Estos objetivos miden **presión de demanda**, no desabasto
observado. Para convertirlos en eventos reales se necesitarían snapshots de inventario
o telemetría de disponibilidad.

### Condiciones de uso y privacidad

MiBici invita en su [portal de datos abiertos](https://mibici.net/es/datos-abiertos) a
“explorar, experimentar y compartir”. La página no muestra una licencia abierta
específica; por ello se conservará atribución y enlace al publicador, no se afirmará una
licencia no publicada y se verificará autorización antes de redistribuir copias.

El [aviso de privacidad](https://mibici.net/es/aviso-de-privacidad) declara que AMIM es
responsable del tratamiento conforme a las leyes general y estatal aplicables. En este
proyecto los CSV crudos permanecen ignorados por Git, no se publican IDs de usuario ni
atributos demográficos, y los artefactos sólo contienen agregados estación-hora,
métricas y modelos. Se aplica minimización, acceso limitado y ausencia de intentos de
reidentificación.

### Partición sin información del futuro

Se ordenan las horas y se usa el primer 80% para entrenamiento y el último 20% para
prueba. No hay mezcla aleatoria. Todos los conteos y medias móviles usan `shift(1)`, por
lo que una fila de la hora `h` sólo utiliza eventos terminados antes de `h`. El
preprocesamiento (imputación, escalado y one-hot) se ajusta exclusivamente con train
mediante un `Pipeline` de scikit-learn. La prueba simula predicción rolling de una hora:
al avanzar el reloj sí se incorporan observaciones ya concluidas, nunca la hora objetivo.

## 3. Pipeline de ML

1. **Ingesta:** listar la carpeta pública, descargar CSV de forma atómica, calcular
   SHA-256 y manifestar la versión.
2. **Limpieza:** parsear fechas, eliminar campos críticos inválidos, duraciones negativas,
   viajes origen=destino y estaciones sin capacidad positiva.
3. **Agregación:** convertir viajes en salidas/llegadas y crear una serie horaria completa
   entre la primera y última actividad de cada estación.
4. **Features:** capacidad; estación; calendario cíclico (hora, día de semana y mes);
   fin de semana; salidas, llegadas y flujo neto rezagados una hora; medias de 3, 24 y
   168 horas; y riesgo observado en la hora anterior. Representan demanda inmediata,
   patrón diario/semanal, estacionalidad y heterogeneidad de capacidad/ubicación.
5. **Entrenamiento:** regresión logística multisalida con regularización L2,
   `class_weight=balanced`, `solver=lbfgs` y `max_iter=1000`.
6. **Evaluación:** conjunto futuro, métricas por objetivo y matrices de confusión.
7. **Tracking:** parámetros, versión de datos, commit, métricas, figura y modelo en MLflow.

No se usan como features los conteos ni desbalances de la hora objetivo. La duración de
un viaje individual tampoco se usa: al momento de predecir no se conoce la duración de
los viajes que todavía no empiezan y usarla introduciría fuga temporal.

## 4. Modelos y métricas

- **Referencia mínima:** `DummyClassifier(strategy="prior")`, útil para comprobar que el
  modelo aporta señal frente a la prevalencia.
- **Baseline:** regresión logística regularizada sin ponderación de clases. Es
  interpretable, entrega probabilidades y establece una referencia reproducible.
- **Candidato:** la misma regresión logística con `class_weight="balanced"`. Mantener
  iguales las variables, el split temporal y los demás hiperparámetros permite medir
  directamente el efecto del balanceo sobre las clases minoritarias.

La métrica principal es **F1** por objetivo porque los positivos son minoritarios y se
necesita equilibrar alarmas falsas con riesgos omitidos. Se reportan además **recall**
(un falso negativo puede impedir el rebalanceo), **precision** (las unidades de
rebalanceo son limitadas), **average precision/PR-AUC** (compara el ranking sin depender
de un único umbral y es informativa con desbalance) y ROC-AUC como diagnóstico
secundario. El umbral de decisión deberá ajustarse después con el costo real de una
visita innecesaria frente a un desabasto omitido.

## 5. Ejecución

```bash
make requirements
python -m mibici_ml.dataset prepare
python -m mibici_ml.modeling.train train-p2
make mlflow-ui
```

El experimento se llama `mibici-equipo50` y genera los runs
`p2-baseline-logistic-regression` y `p2-candidato-logistic-regression`. Ambos incluyen
etiquetas `team`, `question` y `git_commit`; parámetros de fechas/versión; métricas;
matrices de confusión; metadatos de datos y el modelo serializado.
