"""Download and inventory the public MiBici data used by the project."""

from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
import re
from urllib.request import Request, urlopen

from loguru import logger
import pandas as pd
import typer

from mibici_ml.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer(no_args_is_help=True)

DEFAULT_FOLDER_ID = "1XdbZGmBX2BnosPNioK7jD1kwKoSEvwG4"
FOLDER_VIEW_URL = "https://drive.google.com/embeddedfolderview?id={folder_id}#list"
DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
)
TRIP_COLUMNS = [
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
]


def _open_url(url: str):
    request = Request(url, headers={"User-Agent": "mibici-equipo50/1.0"})
    return urlopen(request, timeout=120)


def list_public_drive_folder(folder_id: str = DEFAULT_FOLDER_ID) -> list[dict[str, str]]:
    """Return names and IDs from a publicly shared Google Drive folder."""
    with _open_url(FOLDER_VIEW_URL.format(folder_id=folder_id)) as response:
        page = response.read().decode("utf-8")

    ids = re.findall(r'class="flip-entry" id="entry-([^"]+)', page)
    raw_names = re.findall(r'class="flip-entry-title">(.*?)</div>', page)
    names = [html.unescape(re.sub(r"<.*?>", "", name)) for name in raw_names]
    if not ids or len(ids) != len(names):
        raise RuntimeError("No fue posible leer la carpeta pública de Google Drive.")
    return [{"file_id": file_id, "name": name} for file_id, name in zip(ids, names)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_public_file(file_id: str, destination: Path) -> None:
    """Download one public Drive file atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with _open_url(DOWNLOAD_URL.format(file_id=file_id)) as source, temporary.open("wb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            target.write(chunk)
    temporary.replace(destination)


def download_drive_data(
    output_dir: Path = RAW_DATA_DIR,
    folder_id: str = DEFAULT_FOLDER_ID,
    force: bool = False,
) -> dict:
    """Download CSV files and write a checksummed local manifest."""
    entries = [
        item for item in list_public_drive_folder(folder_id) if item["name"].endswith(".csv")
    ]
    if not entries:
        raise RuntimeError("La carpeta no contiene archivos CSV públicos.")

    manifest_entries = []
    for entry in entries:
        destination = output_dir / entry["name"]
        if force or not destination.exists():
            logger.info(f"Descargando {entry['name']}")
            download_public_file(entry["file_id"], destination)
        else:
            logger.info(f"Usando archivo existente {entry['name']}")
        manifest_entries.append(
            {
                **entry,
                "url": f"https://drive.google.com/file/d/{entry['file_id']}/view",
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )

    version_payload = "".join(item["sha256"] for item in manifest_entries).encode()
    manifest = {
        "folder_id": folder_id,
        "folder_url": f"https://drive.google.com/drive/folders/{folder_id}",
        "data_version": hashlib.sha256(version_payload).hexdigest()[:12],
        "files": manifest_entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def read_csv_compatible(path: Path, **kwargs) -> pd.DataFrame:
    """Read MiBici CSV files, whose current exports use Latin-1."""
    return pd.read_csv(path, encoding="latin1", na_values=["NULL"], **kwargs)


def profile_raw_data(input_dir: Path = RAW_DATA_DIR) -> dict:
    """Calculate the actual row, station, date, and quality counts."""
    files = sorted(input_dir.glob("datos_abiertos_*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay archivos de viajes en {input_dir}")

    stations: set[int] = set()
    total_rows = 0
    invalid_rows = 0
    same_station_rows = 0
    minimum_date = None
    maximum_date = None
    monthly = []

    for path in files:
        frame = read_csv_compatible(path, usecols=TRIP_COLUMNS)
        start = pd.to_datetime(frame["Inicio_del_viaje"], errors="coerce")
        end = pd.to_datetime(frame["Fin_del_viaje"], errors="coerce")
        ids = pd.concat([frame["Origen_Id"], frame["Destino_Id"]]).dropna().astype(int)
        stations.update(ids.tolist())
        critical_invalid = frame[TRIP_COLUMNS].isna().any(axis=1) | start.isna() | end.isna()
        same_station = frame["Origen_Id"].eq(frame["Destino_Id"])
        total_rows += len(frame)
        invalid_rows += int(critical_invalid.sum())
        same_station_rows += int(same_station.sum())
        file_min = start.min()
        file_max = end.max()
        minimum_date = file_min if minimum_date is None else min(minimum_date, file_min)
        maximum_date = file_max if maximum_date is None else max(maximum_date, file_max)
        monthly.append(
            {
                "file": path.name,
                "trips": len(frame),
                "stations": int(ids.nunique()),
                "start": file_min.isoformat(),
                "end": file_max.isoformat(),
            }
        )

    return {
        "months": len(files),
        "trips": total_rows,
        "stations": len(stations),
        "start": minimum_date.isoformat(),
        "end": maximum_date.isoformat(),
        "invalid_critical_rows": invalid_rows,
        "same_station_rows": same_station_rows,
        "monthly": monthly,
    }


@app.command("download-drive")
def download_drive_command(
    output_dir: Path = RAW_DATA_DIR,
    folder_id: str = DEFAULT_FOLDER_ID,
    force: bool = False,
):
    """Download all public CSV files from the project Drive folder."""
    manifest = download_drive_data(output_dir, folder_id, force)
    logger.success(f"Versión de datos: {manifest['data_version']}")


@app.command("profile")
def profile_command(input_dir: Path = RAW_DATA_DIR):
    """Print and save a profile of the downloaded raw data."""
    profile = profile_raw_data(input_dir)
    (input_dir / "profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(json.dumps(profile, indent=2, ensure_ascii=False))


def read_trip_file(path: Path) -> tuple[pd.DataFrame, str]:
    """Read a monthly trips CSV using a supported encoding."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            return df, encoding
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode file: {path.name}")


@app.command("ingest")
def ingest(
    input_dir: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "mibici_trips.parquet",
    manifest_path: Path = PROCESSED_DATA_DIR / "manifest.json",
):
    """Validate, deduplicate, and save monthly MiBici trip data."""
    input_files = sorted(input_dir.glob("datos_abiertos_*.csv"))

    if not input_files:
        raise FileNotFoundError(f"No monthly trips files found in {input_dir}")

    logger.info("Found {} monthly files", len(input_files))

    processed_frames = []
    file_records = []

    for path in input_files:
        logger.info("Processing {}", path.name)

        df, encoding = read_trip_file(path)

        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
        if missing_columns:
            raise ValueError(f"{path.name} is missing required columns: {missing_columns}")

        original_rows = len(df)
        duplicate_rows = int(df["Viaje_Id"].duplicated(keep="first").sum())

        # Keep only the fields required for P1.
        df = df[MODEL_COLUMNS].copy()

        df["Inicio_del_viaje"] = pd.to_datetime(df["Inicio_del_viaje"], errors="coerce")
        df["Fin_del_viaje"] = pd.to_datetime(df["Fin_del_viaje"], errors="coerce")

        invalid_start = int(df["Inicio_del_viaje"].isna().sum())
        invalid_end = int(df["Fin_del_viaje"].isna().sum())

        if invalid_start or invalid_end:
            raise ValueError(
                f"{path.name} contains invalid timestamps: "
                f"start={invalid_start}, end={invalid_end}"
            )

        if df["Viaje_Id"].isna().any():
            raise ValueError(f"{path.name} contains missing Viaje_Id values")

        if df[["Origen_Id", "Destino_Id"]].isna().any().any():
            raise ValueError(f"{path.name} contains missing station IDs")

        # Deduplicate within each monthly file and record the row count.
        df = df.drop_duplicates(subset=["Viaje_Id"], keep="first")

        duration_minutes = (df["Fin_del_viaje"] - df["Inicio_del_viaje"]).dt.total_seconds() / 60

        record = {
            "file": path.name,
            "sha256": sha256_file(path),
            "encoding": encoding,
            "rows_original": original_rows,
            "duplicate_rows_removed": duplicate_rows,
            "rows_after_deduplication": len(df),
            "invalid_start_timestamps": invalid_start,
            "invalid_end_timestamps": invalid_end,
            "negative_durations": int((duration_minutes < 0).sum()),
            "durations_over_24_hours": int((duration_minutes > 1440).sum()),
        }

        file_records.append(record)
        processed_frames.append(df)

        logger.info(
            "{}: {} rows -> {} rows; {} duplicates removed",
            path.name,
            original_rows,
            len(df),
            duplicate_rows,
        )

    trips = pd.concat(processed_frames, ignore_index=True)

    # Validate IDs across the complete dataset as well.
    cross_file_duplicates = int(trips["Viaje_Id"].duplicated(keep=False).sum())

    if cross_file_duplicates:
        raise ValueError(
            "Repeated Viaje_Id values were found across monthly files. "
            "Review the source data before continuing."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    trips.to_parquet(output_path, index=False, compression="snappy")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": output_path.name,
        "dataset_sha256": sha256_file(output_path),
        "source_directory": str(input_dir),
        "source_files": file_records,
        "summary": {
            "source_files": len(input_files),
            "rows_before_deduplication": sum(record["rows_original"] for record in file_records),
            "duplicate_rows_removed": sum(
                record["duplicate_rows_removed"] for record in file_records
            ),
            "rows_after_deduplication": len(trips),
            "start_date": trips["Inicio_del_viaje"].min().isoformat(),
            "end_date": trips["Inicio_del_viaje"].max().isoformat(),
            "unique_origin_stations": int(trips["Origen_Id"].nunique()),
            "unique_destination_stations": int(trips["Destino_Id"].nunique()),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.success("Parquet saved to {}", output_path)
    logger.success("Manifest saved to {}", manifest_path)
    logger.info("Final row count: {}", len(trips))


if __name__ == "__main__":
    app()
