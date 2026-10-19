from pathlib import Path

import pandas as pd

from mibici_ml.config import SELECTED_SOURCE_FILES
from mibici_ml.dataset import clean_p3_stations, download_drive_data, normalize_column


def test_download_uses_exactly_the_selected_sources(tmp_path, monkeypatch):
    entries = [
        {"name": name, "file_id": f"id-{index}"}
        for index, name in enumerate(SELECTED_SOURCE_FILES)
    ]
    entries.append({"name": "datos_abiertos_fuera_de_rango.csv", "file_id": "extra"})

    monkeypatch.setattr("mibici_ml.dataset.list_public_drive_folder", lambda folder_id: entries)

    def fake_download(file_id: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(file_id, encoding="utf-8")

    monkeypatch.setattr("mibici_ml.dataset.download_public_file", fake_download)

    manifest = download_drive_data(tmp_path, folder_id="test-folder")

    assert [item["name"] for item in manifest["files"]] == SELECTED_SOURCE_FILES
    assert not (tmp_path / "datos_abiertos_fuera_de_rango.csv").exists()
    assert len(manifest["data_version"]) == 12


def test_station_ingestion_normalizes_columns_and_numeric_types(tmp_path):
    path = tmp_path / "nomenclatura.csv"
    pd.DataFrame(
        {
            "ID": [1],
            "Name": ["GDL Centro"],
            "Latitude": ["20.67"],
            "Longitude": ["-103.34"],
            "DPCapacity": ["12"],
        }
    ).to_csv(path, index=False)

    stations = clean_p3_stations(path)

    assert normalize_column("Año de operación") == "ano_de_operacion"
    assert stations.loc[0, "estacion_id"] == 1
    assert stations.loc[0, "capacidad"] == 12
    assert stations["estacion_id"].dtype == "int32"
