import json
from pathlib import Path

from mibici_ml.api.schemas import PredictionInput

ROOT = Path(__file__).resolve().parents[1]


def test_locust_payloads_match_api_contract():
    payloads = json.loads((ROOT / "load_testing/payloads.json").read_text(encoding="utf-8"))

    assert len(payloads) >= 5
    for payload in payloads:
        PredictionInput.model_validate(payload)


def test_governance_artifacts_are_versioned_and_consistent():
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    drift = json.loads(
        (ROOT / "references/observabilidad/p2_drift_2026_10.json").read_text(encoding="utf-8")
    )
    ab_test = json.loads(
        (ROOT / "references/ab_testing/p2_ab_2026_10.json").read_text(encoding="utf-8")
    )
    finops = json.loads((ROOT / "references/finops/p2_finops.json").read_text(encoding="utf-8"))

    for section in ("uso previsto", "evaluación", "monitoreo y drift", "limitaciones"):
        assert section in card.lower()
    assert drift["evidently_version"] == "0.7.23"
    assert drift["champion_performance_current"]["version"] == 2
    assert ab_test["decision"]["winner"] == "champion"
    assert ab_test["champion"]["version"] == 2
    assert finops["capacity"]["p95_ms"] <= finops["assumptions"]["p95_limit_ms"]
    assert (ROOT / "reports/evidently/p2/drift_2026_10.html").is_file()
