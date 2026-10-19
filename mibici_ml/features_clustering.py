"""Perfil de uso de cada estación para el clustering (P3).

Lee los viajes limpios de data/interim y arma una tabla con una fila por estación. Las
variables van por día activo o como proporciones, porque el sistema creció durante el
periodo y no todas las estaciones tienen la misma cantidad de días con datos.

El clustering es descriptivo: no hay un momento de predicción ni una variable objetivo, así
que no hay información futura que se pueda filtrar. Todas las variables se calculan solo con
los viajes del rango de fechas de config.py.

Uso:
    python -m mibici_ml.features_clustering
"""

from loguru import logger
import pandas as pd
import typer

from mibici_ml.config import (
    CLUSTERING_FEATURES_FILE,
    CLUSTERING_MIN_ACTIVE_DAYS,
    CLUSTERING_STATIONS_CLEAN_FILE,
    CLUSTERING_TRIPS_FILE,
)

app = typer.Typer()

# Franjas según la hora de inicio del viaje. El sistema casi no opera de 1 a 5 h, así que la
# noche junta de las 20 a las 6 h.
FRANJAS = {
    "manana": range(6, 10),
    "mediodia": range(10, 16),
    "tarde": range(16, 20),
    "noche": [*range(20, 24), *range(0, 6)],
}
HORA_A_FRANJA = {hora: franja for franja, horas in FRANJAS.items() for hora in horas}


def cargar_datos():
    """Lee los viajes y las estaciones limpios que genera dataset_clustering.py."""
    for ruta in [CLUSTERING_TRIPS_FILE, CLUSTERING_STATIONS_CLEAN_FILE]:
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe {ruta}. Corre primero: python -m mibici_ml.dataset_clustering"
            )
    viajes = pd.read_parquet(CLUSTERING_TRIPS_FILE)
    estaciones = pd.read_parquet(CLUSTERING_STATIONS_CLEAN_FILE)
    return viajes, estaciones


def contar_dias_activos(viajes):
    """Días con al menos una salida o llegada en cada estación.

    Un viaje cuenta para el día en que inicia, también en la estación de destino.
    """
    dia = viajes["inicio"].dt.normalize()
    dias = pd.concat(
        [
            pd.DataFrame({"estacion_id": viajes["origen_id"], "dia": dia}),
            pd.DataFrame({"estacion_id": viajes["destino_id"], "dia": dia}),
        ]
    )
    return dias.drop_duplicates().groupby("estacion_id").size()


def calcular_perfil(viajes):
    """Calcula el perfil de uso de cada estación, con una fila por estación.

    Las salidas se cuentan en la estación de origen y las llegadas en la de destino.

    Variables:
        dias_activos: días con al menos una salida o llegada.
        salidas, llegadas: total de viajes que salen y que llegan a la estación.
        viajes_dia: salidas más llegadas por día activo; mide qué tanto se usa la estación.
        balance: (llegadas - salidas) / (llegadas + salidas). Es positivo si llegan más bicis
            de las que salen y negativo si salen más de las que llegan.
        prop_manana, prop_mediodia, prop_tarde, prop_noche: proporción de las salidas que
            inician en cada franja horaria (ver FRANJAS). Las cuatro suman 1.
        prop_fin_semana: proporción de las salidas en sábado o domingo.
        prop_redondo: proporción de las salidas que regresan a la misma estación, que suele
            ser un uso recreativo.
        duracion_mediana: mediana de la duración, en minutos, de los viajes que salen de ahí.
    """
    origen = viajes["origen_id"].rename("estacion_id")
    destino = viajes["destino_id"].rename("estacion_id")

    perfil = pd.DataFrame(
        {
            "dias_activos": contar_dias_activos(viajes),
            "salidas": viajes.groupby(origen).size(),
            "llegadas": viajes.groupby(destino).size(),
        }
    )
    perfil = perfil.fillna(0).astype(int)
    total = perfil["salidas"] + perfil["llegadas"]
    perfil["viajes_dia"] = total / perfil["dias_activos"]
    perfil["balance"] = (perfil["llegadas"] - perfil["salidas"]) / total

    franja = viajes["inicio"].dt.hour.map(HORA_A_FRANJA)
    por_franja = pd.crosstab(origen, franja, normalize="index")
    perfil = perfil.join(por_franja.reindex(columns=list(FRANJAS)).add_prefix("prop_"))

    fin_semana = viajes["inicio"].dt.dayofweek >= 5
    redondo = viajes["origen_id"] == viajes["destino_id"]
    perfil["prop_fin_semana"] = fin_semana.groupby(origen).mean()
    perfil["prop_redondo"] = redondo.groupby(origen).mean()
    perfil["duracion_mediana"] = viajes.groupby(origen)["duracion_min"].median()
    return perfil.rename_axis("estacion_id").reset_index()


def filtrar_estaciones(perfil, estaciones):
    """Une el perfil con los datos de cada estación y deja solo las que sirven para agrupar.

    Quita las estaciones sin viajes, las retiradas (prefijo "ex-") y las que tienen menos de
    CLUSTERING_MIN_ACTIVE_DAYS días activos, porque con tan poca historia su perfil no es
    confiable. Regresa el perfil filtrado y un resumen de cuántas estaciones quitó cada regla.
    """
    resumen = {"estaciones_nomenclatura": len(estaciones)}
    perfil = estaciones.merge(perfil, on="estacion_id", how="inner")
    resumen["sin_viajes"] = resumen["estaciones_nomenclatura"] - len(perfil)

    resumen["retiradas"] = int((~perfil["activa"]).sum())
    perfil = perfil[perfil["activa"]]

    pocos_dias = perfil["dias_activos"] < CLUSTERING_MIN_ACTIVE_DAYS
    resumen["pocos_dias_activos"] = int(pocos_dias.sum())
    perfil = perfil[~pocos_dias]

    resumen["estaciones_finales"] = len(perfil)
    return perfil.drop(columns="activa").reset_index(drop=True), resumen


@app.command()
def main():
    """Calcula el perfil de cada estación y lo guarda en data/processed."""
    viajes, estaciones = cargar_datos()
    perfil = calcular_perfil(viajes)
    perfil, resumen = filtrar_estaciones(perfil, estaciones)
    for motivo, cantidad in resumen.items():
        logger.info(f"  {motivo}: {cantidad}")

    CLUSTERING_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    perfil.to_parquet(CLUSTERING_FEATURES_FILE, index=False)
    logger.success(f"Perfil de {len(perfil)} estaciones en {CLUSTERING_FEATURES_FILE}")


if __name__ == "__main__":
    app()
