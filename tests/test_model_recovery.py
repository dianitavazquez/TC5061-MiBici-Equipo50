"""Unit tests for deterministic P2 model recovery and loading."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mibici_ml.modeling import model_recovery


class FakeClient:
    def get_model_version(self, name, version):
        return SimpleNamespace(name=name, version=version, run_id="run-version")

    def get_model_version_by_alias(self, name, alias):
        assert alias == "champion"
        return SimpleNamespace(name=name, version="2", run_id="run-alias")


def test_registered_model_uri_supports_version_and_alias():
    assert model_recovery.registered_model_uri("p2", version=2) == "models:/p2/2"
    assert model_recovery.registered_model_uri("p2", alias="champion") == "models:/p2@champion"


def test_registered_model_uri_requires_selector():
    with pytest.raises(ValueError, match="versión o un alias"):
        model_recovery.registered_model_uri("p2", alias=None)


def test_resolve_alias_returns_fixed_version():
    version, alias = model_recovery.resolve_model_version(FakeClient(), "p2")

    assert version.version == "2"
    assert version.run_id == "run-alias"
    assert alias == "champion"


def test_recover_model_downloads_fixed_version_and_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(model_recovery, "MlflowClient", lambda **_: FakeClient())

    def fake_download(*, artifact_uri, dst_path):
        assert artifact_uri == "models:/p2/2"
        model_dir = Path(dst_path) / "artifact"
        model_dir.mkdir()
        (model_dir / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
        (model_dir / "model.pkl").write_bytes(b"trusted-test-model")
        return str(model_dir)

    monkeypatch.setattr(model_recovery.mlflow.artifacts, "download_artifacts", fake_download)
    destination = tmp_path / "serving" / "p2"

    manifest = model_recovery.recover_model(
        tracking_uri="http://mlflow:5000",
        name="p2",
        destination=destination,
    )

    assert manifest["version"] == 2
    assert manifest["model_uri"] == "models:/p2/2"
    assert manifest["alias_at_download"] == "champion"
    assert (destination / "MLmodel").is_file()
    assert (destination / "model-info.json").is_file()
    assert set(manifest["files_sha256"]) == {"MLmodel", "model.pkl"}


def test_load_bundled_model_uses_local_mlflow_directory(tmp_path, monkeypatch):
    (tmp_path / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
    monkeypatch.setattr(model_recovery, "load_pyfunc_model", lambda uri: ("loaded", uri))

    loaded = model_recovery.load_p2_model(source="bundled", model_path=tmp_path)

    assert loaded == ("loaded", str(tmp_path.resolve()))


def test_load_registry_model_prefers_explicit_version(monkeypatch):
    monkeypatch.setattr(model_recovery, "load_pyfunc_model", lambda uri: uri)

    loaded = model_recovery.load_p2_model(
        source="registry",
        tracking_uri="http://mlflow:5000",
        name="p2",
        version="7",
    )

    assert loaded == "models:/p2/7"
