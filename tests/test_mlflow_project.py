import json
import shlex

from mlflow.projects._project_spec import load_project
import pytest
from typer.testing import CliRunner

from mibici_ml import project
from mibici_ml.config import P2_MAX_ITER, PROJ_ROOT

runner = CliRunner()


def test_mlproject_declares_reproducible_entry_points():
    definition = (PROJ_ROOT / "MLproject").read_text(encoding="utf-8")

    assert "python_env: python_env.yaml" in definition
    assert "ingest:" in definition
    assert "train:" in definition
    assert "p2-search:" in definition
    assert "question:" in definition
    assert "model:" in definition
    assert "hyperparameters:" in definition

    train_entry = load_project(str(PROJ_ROOT)).get_entry_point("train")
    command = train_entry.compute_command(
        {"question": "p2", "model": "candidato", "hyperparameters": '{"c":0.5}'}, None
    )
    assert shlex.split(command)[-1] == '{"c":0.5}'


def test_parse_hyperparameters_requires_a_json_object():
    assert project.parse_hyperparameters('{"k": 4}') == {"k": 4}

    with pytest.raises(Exception, match="objeto JSON"):
        project.parse_hyperparameters("[4]")


def test_train_dispatches_p2_parameters(monkeypatch):
    received = {}

    def fake_train_p2(**kwargs):
        received.update(kwargs)

    monkeypatch.setattr(project, "train_p2", fake_train_p2)
    parameters = json.dumps({"c": 0.5, "class_weight": "balanced"})

    result = runner.invoke(
        project.app,
        ["train", "--question", "p2", "--model", "candidato", "--hyperparameters", parameters],
    )

    assert result.exit_code == 0, result.output
    assert received == {
        "models": ("candidato",),
        "model_params": {"c": 0.5, "class_weight": "balanced", "max_iter": P2_MAX_ITER},
    }


def test_train_rejects_unknown_hyperparameters():
    result = runner.invoke(
        project.app,
        ["train", "--question", "p3", "--model", "candidato", "--hyperparameters", '{"c": 1}'],
    )

    assert result.exit_code != 0
    assert "Hiperparámetros no válidos para p3" in result.output
