from types import SimpleNamespace

from mibici_ml.modeling import registry_bootstrap


def test_bundled_manifests_define_champion_and_challenger():
    manifests = {
        variant["alias"]: registry_bootstrap.read_model_manifest(variant["model_path"])
        for variant in registry_bootstrap.MODEL_VARIANTS
    }

    assert manifests["champion"]["version"] == 2
    assert manifests["challenger"]["version"] == 3
    assert manifests["champion"]["registered_model"] == "mibici-equipo50-p2-riesgo"


def test_seed_variant_reuses_same_immutable_artifact(monkeypatch):
    client = SimpleNamespace(
        set_model_version_tag=lambda *args: None,
        update_model_version=lambda *args, **kwargs: None,
        set_registered_model_alias=lambda *args: None,
    )
    monkeypatch.setattr(
        registry_bootstrap,
        "_find_seeded_version",
        lambda client, name, manifest: SimpleNamespace(version="7"),
    )

    result = registry_bootstrap.seed_variant(
        client,
        model_name="mibici-equipo50-p2-riesgo",
        **registry_bootstrap.MODEL_VARIANTS[0],
    )

    assert result["registry_version"] == 7
    assert result["source_registry_version"] == 2
    assert result["reused"] is True
