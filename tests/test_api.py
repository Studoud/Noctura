from __future__ import annotations

from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi.testclient import TestClient
import pytest

from noctura_api.api import create_app
from noctura_api.decision_service import create_demo_decisions
from noctura_api.energy_forecast import forecast_energy
from noctura_api.forecast import predict_remaining_today

ROOT = Path(__file__).resolve().parents[1]
ATTENDANCE_CSV = ROOT / "data" / "dataset_ml_frequentation_24h_clean.csv"
BIODIVERSITY_CSV = ROOT / "data" / "biodev_1_tidy.csv"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "storage"))


def test_training_is_manual_and_predicts_only_after_success(client: TestClient) -> None:
    assert client.get("/api/v1/ml/model").json() == {
        "status": "not_trained",
        "model": None,
    }
    untrained_prediction = client.post("/api/v1/predictions/attendance", json={
        "hour_start": 14,
        "zone_group_id": "G1",
        "zone_specifique": "quai_est",
        "is_daylight": 1,
        "is_winter": 0,
        "is_storm": 0,
        "indice_global": 11,
    })
    assert untrained_prediction.status_code == 409
    assert untrained_prediction.json()["error"]["code"] == "model_not_trained"

    trained = client.post("/api/v1/ml/train")
    assert trained.status_code == 200
    job = trained.json()
    assert job["status"] == "completed"
    assert job["excluded_synthetic_rows"] > 0
    assert client.get(f"/api/v1/ml/train/{job['job_id']}").json()["status"] == "completed"
    trained_model = client.get("/api/v1/ml/model").json()["model"]
    assert trained_model["version"] == job["model_version"]
    assert trained_model["validation"]["method"] == "grouped_holdout_by_feature_vector"
    assert trained_model["validation"]["temporal_validation"] is False
    assert any(
        "CSV d'éclairage contient des dates" in limitation
        for limitation in trained_model["validation"]["limitations"]
    )
    assert job["metrics"]["holdout_feature_groups"] > 0
    assert job["metrics"]["mae"] < job["metrics"]["baseline_mae"]
    assert job["metrics"]["rmse"] < job["metrics"]["baseline_rmse"]
    assert client.get("/api/v1/ml/evaluations").json()["count"] == 1

    result = client.post("/api/v1/predictions/attendance", json={
        "hour_start": 14,
        "zone_group_id": "G1",
        "zone_specifique": "quai_est",
        "is_daylight": 1,
        "is_winter": 0,
        "is_storm": 0,
        "indice_global": 11,
    })
    assert result.status_code == 200
    assert isinstance(result.json()["predicted"], float)
    assert "estimation, pas une mesure réelle" in result.json()["warnings"][0]


def test_today_predictions_use_weather_and_can_cover_one_or_all_zones(
    client: TestClient,
) -> None:
    trained = client.post("/api/v1/ml/train")
    assert trained.status_code == 200
    data_manager = client.app.state.data_manager
    model_service = client.app.state.model_service
    now = datetime(2026, 10, 17, 15, 0, tzinfo=ZoneInfo("Europe/Paris"))
    hourly_weather = {
        "time": [f"2026-10-17T{hour:02d}:00" for hour in range(24)],
        "is_day": [int(7 <= hour < 19) for hour in range(24)],
        "weather_code": [95 if hour == 16 else 0 for hour in range(24)],
        "cloud_cover": [60 for _ in range(24)],
    }

    single_zone = predict_remaining_today(
        data_manager,
        model_service,
        zone="quai_est",
        now=now,
        hourly_weather=hourly_weather,
    )
    all_zones = predict_remaining_today(
        data_manager,
        model_service,
        now=now,
        hourly_weather=hourly_weather,
    )

    assert single_zone["prediction_count"] == 4
    assert {item["zone"] for item in single_zone["items"]} == {"quai_est"}
    assert all_zones["prediction_count"] == 4 * 8
    assert all(
        isinstance(item["predicted"], float) for item in all_zones["items"]
    )
    storm_slot = next(item for item in all_zones["items"] if item["time_slot"] == "16h-18h")
    assert storm_slot["features"]["is_storm"] == 1
    assert storm_slot["features"]["indice_global_source"] == (
        "dataset_median_including_synthetic_rows_for_zone_and_time_slot"
    )
    assert storm_slot["training_coverage"] == "no_observed_training_rows_for_hour"
    assert any("lignes marquées synthétiques" in warning for warning in all_zones["warnings"])


def test_attendance_forecast_supports_tomorrow_and_empty_today(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained = client.post("/api/v1/ml/train")
    assert trained.status_code == 200
    now = datetime(2026, 10, 17, 23, 30, tzinfo=ZoneInfo("Europe/Paris"))
    hourly_weather = {
        "time": [f"2026-10-18T{hour:02d}:00" for hour in range(24)],
        "is_day": [int(7 <= hour < 19) for hour in range(24)],
        "weather_code": [0 for _ in range(24)],
        "cloud_cover": [50 for _ in range(24)],
    }
    monkeypatch.setattr(
        "noctura_api.forecast.fetch_hourly_weather",
        lambda: pytest.fail("la météo ne doit pas être appelée sans créneau futur"),
    )

    today = predict_remaining_today(
        client.app.state.data_manager,
        client.app.state.model_service,
        now=now,
    )
    tomorrow = predict_remaining_today(
        client.app.state.data_manager,
        client.app.state.model_service,
        day="tomorrow",
        zone="quai_est",
        now=now,
        hourly_weather=hourly_weather,
    )

    assert today["status"] == "no_future_slots"
    assert today["prediction_count"] == 0
    assert tomorrow["date"] == "2026-10-18"
    assert tomorrow["day"] == "tomorrow"
    assert tomorrow["prediction_count"] == 12
    assert {item["zone"] for item in tomorrow["items"]} == {"quai_est"}

    route_tomorrow = (
        datetime.now(ZoneInfo("Europe/Paris")) + timedelta(days=1)
    ).date().isoformat()
    route_weather = {
        "time": [f"{route_tomorrow}T{hour:02d}:00" for hour in range(24)],
        "is_day": [int(7 <= hour < 19) for hour in range(24)],
        "weather_code": [0 for _ in range(24)],
        "cloud_cover": [50 for _ in range(24)],
    }
    monkeypatch.setattr(
        "noctura_api.forecast.fetch_hourly_weather", lambda: route_weather
    )
    response = client.get(
        "/api/v1/predictions/attendance/today?day=tomorrow&zone=quai_est"
    )
    assert response.status_code == 200
    assert response.json()["day"] == "tomorrow"
    assert response.json()["prediction_count"] == 12


def test_energy_forecast_is_simulation_for_today_and_tomorrow(
    client: TestClient,
) -> None:
    now = datetime(2026, 10, 17, 16, 7, tzinfo=ZoneInfo("Europe/Paris"))
    data_manager = client.app.state.data_manager

    today = forecast_energy(data_manager, day="today", now=now)
    tomorrow = forecast_energy(data_manager, day="tomorrow", now=now)

    assert today["status"] == "estimated_simulation"
    assert today["date"] == "2026-10-17"
    assert today["period"] == "full_day"
    assert tomorrow["date"] == "2026-10-18"
    assert len(today["zones"]) == 8
    assert len(tomorrow["zones"]) == 8
    assert today["data_source"] == "data/eclairage_historique_simule.csv"
    assert today["monte_carlo"]["simulations"] == 10_000
    assert today["monte_carlo"]["historical_profiles_used"] == 4
    assert len(today["monte_carlo"]["historical_dates_used"]) == 4
    assert (
        today["summary"]["consumption_interval_90_kwh"]["p05"]
        <= today["summary"]["predicted_consumption_kwh"]
        <= today["summary"]["consumption_interval_90_kwh"]["p95"]
    )
    assert today["summary"]["potential_savings_kwh"] == round(
        today["summary"]["full_power_reference_kwh"]
        - today["summary"]["predicted_consumption_kwh"],
        2,
    )
    assert today["zones"] == sorted(
        today["zones"],
        key=lambda item: item["predicted_consumption_kwh"],
        reverse=True,
    )
    assert len(today["top_consuming_zones"]) == 3
    assert today["location"]["latitude"] == 43.0024397
    assert "simulation" in today["warnings"][0].lower()


def test_energy_and_port_frontend_endpoints(client: TestClient) -> None:
    port = client.get("/api/v1/port")
    assert port.status_code == 200
    assert port.json()["name"] == "Port de Porquerolles"
    assert port.json()["location"]["longitude"] == 6.2006567
    assert len(port.json()["zones"]) == 8

    today = client.get("/api/v1/energy/forecast?day=today")
    tomorrow = client.get("/api/v1/energy/forecast?day=tomorrow")
    assert today.status_code == 200
    assert tomorrow.status_code == 200
    assert today.json()["summary"]["predicted_consumption_kwh"] > 0
    assert tomorrow.json()["summary"]["potential_savings_percent"] is not None
    assert client.get("/api/v1/energy/forecast?day=next_week").status_code == 422
    assert today.json()["model_performance"]["name"] == "Noctora"
    assert today.json()["model_performance"]["status"] == "not_trained"
    assert today.json()["model_performance"]["metrics"] is None
    assert today.json()["decision_report"]["status"] == "no_decisions_recorded"

    trained = client.post("/api/v1/ml/train")
    assert trained.status_code == 200
    today = datetime.now(ZoneInfo("Europe/Paris"))
    recommendation = client.post(
        "/api/v1/lighting/recommendations",
        json={
            "hour_start": 20,
            "zone_group_id": "G1",
            "zone_specifique": "quai_est",
            "is_daylight": 0,
            "is_winter": 0,
            "is_storm": 0,
            "indice_global": 6,
            "timestamp": today.isoformat(),
        },
    )
    assert recommendation.status_code == 200
    forecast = client.get("/api/v1/energy/forecast?day=today").json()
    assert forecast["model_performance"]["status"] == "ready"
    assert forecast["model_performance"]["version"] == trained.json()["model_version"]
    assert {
        "mae",
        "rmse",
        "r2",
        "holdout_rows",
        "baseline_mae",
        "baseline_rmse",
        "baseline_r2",
        "holdout_feature_groups",
    } == set(forecast["model_performance"]["metrics"])
    assert forecast["decision_report"]["count"] == 1
    assert forecast["decision_report"]["pending_validation"] == 1
    decisions = client.get(
        f"/api/v1/reports/decisions?date={today.date().isoformat()}"
    )
    assert decisions.status_code == 200
    assert decisions.json()["count"] == 1


def test_demo_decisions_are_explainable_and_review_never_controls_hardware(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained = client.post("/api/v1/ml/train")
    assert trained.status_code == 200
    forecast = {
        "port": "Port de Porquerolles",
        "date": "2026-10-17",
        "items": [
            {
                "zone": "quai_sud_commerces",
                "time_slot": "20h-22h",
                "predicted": 19.0,
                "unit": "individuals_per_two_hour_slot",
            },
            {
                "zone": "capitainerie",
                "time_slot": "20h-22h",
                "predicted": 3.0,
                "unit": "individuals_per_two_hour_slot",
            },
            {
                "zone": "quai_est",
                "time_slot": "20h-22h",
                "predicted": 9.0,
                "unit": "individuals_per_two_hour_slot",
            },
        ],
    }
    monkeypatch.setattr(
        "noctura_api.api.predict_remaining_today",
        lambda *_args, **_kwargs: forecast,
    )
    generated = client.post(
        "/api/v1/lighting/decisions/generate?zone=quai_sud_commerces"
    )
    assert generated.status_code == 200
    body = generated.json()
    assert body["status"] == "simulation_only"
    assert body["decision_count"] == 1
    decision = body["items"][0]
    assert decision["action"] == "increase"
    assert decision["change_percentage_points"] == 10
    assert decision["status"] == "pending_human_review"
    assert decision["mode"] == "simulation_only"
    assert decision["hardware_command_sent"] is False
    assert "fréquentation prévue" in decision["reason"]

    all_decisions = client.post("/api/v1/lighting/decisions/generate")
    assert all_decisions.status_code == 200
    by_action = {
        item["zone"]: item["action"] for item in all_decisions.json()["items"]
    }
    assert by_action == {
        "quai_sud_commerces": "increase",
        "capitainerie": "decrease",
        "quai_est": "maintain",
    }

    decision_id = decision["decision_id"]
    reviewed = client.post(
        f"/api/v1/lighting/decisions/{decision_id}/review",
        json={"action": "approve", "reviewer": "responsable", "note": "test frontend"},
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["status"] == "approved_simulation"
    assert reviewed.json()["hardware_command_sent"] is False

    listing = client.get(
        "/api/v1/lighting/decisions?date=2026-10-17&zone=quai_sud_commerces"
        "&status=approved_simulation"
    )
    assert listing.status_code == 200
    assert listing.json()["count"] == 1
    duplicate_review = client.post(
        f"/api/v1/lighting/decisions/{decision_id}/review",
        json={"action": "reject", "reviewer": "responsable"},
    )
    assert duplicate_review.status_code == 409


def test_failed_training_keeps_previous_model(client: TestClient) -> None:
    first = client.post("/api/v1/ml/train").json()
    version = client.get("/api/v1/ml/model").json()["model"]["version"]
    small_dataset = pd.read_csv(ATTENDANCE_CSV).head(8).to_csv(index=False).encode()
    imported = client.post(
        "/api/v1/datasets/attendance",
        files={"file": ("small.csv", BytesIO(small_dataset), "text/csv")},
    )
    assert imported.status_code == 200

    response = client.post("/api/v1/ml/train")
    assert response.status_code == 409
    failed_job_id = response.json()["error"]["job_id"]
    assert client.get(f"/api/v1/ml/train/{failed_job_id}").json()["status"] == "failed"
    assert client.get("/api/v1/ml/model").json()["model"]["version"] == version
    assert first["status"] == "completed"


def test_import_validates_csvs_and_reports_source_status(client: TestClient) -> None:
    invalid_csv = b"hour_start,zone\n14,G1\n"
    invalid = client.post(
        "/api/v1/datasets/attendance",
        files={"file": ("bad.csv", BytesIO(invalid_csv), "text/csv")},
    )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "dataset_validation_failed"

    infinite = pd.read_csv(ATTENDANCE_CSV)
    infinite.loc[0, "sin_hour"] = float("inf")
    rejected_infinite = client.post(
        "/api/v1/datasets/attendance",
        files={
            "file": (
                "infinite.csv",
                BytesIO(infinite.to_csv(index=False).encode()),
                "text/csv",
            )
        },
    )
    assert rejected_infinite.status_code == 422

    with ATTENDANCE_CSV.open("rb") as source:
        imported = client.post(
            "/api/v1/datasets/attendance",
            files={"file": ("attendance.csv", source, "text/csv")},
        )
    assert imported.status_code == 200
    assert imported.json()["dataset"]["synthetic_rows"] > 0

    with BIODIVERSITY_CSV.open("rb") as source:
        biodiversity = client.post(
            "/api/v1/datasets/biodiversity",
            files={"file": ("biodiversity.csv", source, "text/csv")},
        )
    assert biodiversity.status_code == 200
    biodiversity_report = client.get("/api/v1/reports/biodiversity").json()
    assert biodiversity_report["status"] == "available"
    assert biodiversity_report["species_count"] > 0
    sources = client.get("/api/v1/datasets/status").json()["sources"]
    assert sources["attendance"]["validation_status"] == "valid"
    assert sources["biodiversity"]["available"] is True


def test_attendance_import_keeps_exact_duplicate_rows(client: TestClient) -> None:
    row = pd.read_csv(ATTENDANCE_CSV).head(1)
    duplicate_rows = pd.concat([row, row], ignore_index=True)
    response = client.post(
        "/api/v1/datasets/attendance",
        files={
            "file": (
                "duplicate.csv",
                BytesIO(duplicate_rows.to_csv(index=False).encode()),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200
    dataset = response.json()["dataset"]
    assert dataset["rows"] == 2
    assert any("Doublons exacts conservés : 1" in warning
               for warning in dataset["warnings"])


def test_recommendations_never_invent_operational_intensity(client: TestClient) -> None:
    assert client.post("/api/v1/ml/train").status_code == 200
    weather = client.post("/api/v1/datasets/weather", json={
        "source": "weather_api",
        "measurements": {
            "cloud_cover_percent": 70,
            "natural_light_factor_percent": 25,
        },
    })
    assert weather.status_code == 200
    recommendation = client.post("/api/v1/lighting/recommendations", json={
        "hour_start": 14,
        "zone_group_id": "G1",
        "zone_specifique": "quai_est",
        "is_daylight": 1,
        "is_winter": 0,
        "is_storm": 0,
        "indice_global": 11,
        "timestamp": (
            datetime.now(ZoneInfo("Europe/Paris")) + timedelta(days=1)
        ).isoformat(),
    })
    body = recommendation.json()
    assert body["recommended_intensity_percent"] is None
    assert body["status"] == "pending_validation"
    assert body["constraints"]["safety_check"] == "not_configured"
    assert body["environment"]["cloud_cover_percent"] == 70
    assert client.get("/api/v1/lighting/recommendations?zone=quai_est").json()["count"] == 1


def test_manual_recommendation_rejects_past_slots_and_invalid_zone_group(
    client: TestClient,
) -> None:
    body = {
        "hour_start": 14,
        "zone_group_id": "G1",
        "zone_specifique": "quai_est",
        "is_daylight": 1,
        "is_winter": 0,
        "is_storm": 0,
        "indice_global": 11,
    }
    future_time = (
        datetime.now(ZoneInfo("Europe/Paris")) + timedelta(days=1)
    ).isoformat()
    untrained = client.post(
        "/api/v1/lighting/recommendations",
        json={**body, "timestamp": future_time},
    )
    assert untrained.status_code == 409
    assert untrained.json()["error"]["code"] == "model_not_trained"

    past = client.post(
        "/api/v1/lighting/recommendations",
        json={**body, "timestamp": "2020-01-01T12:00:00+00:00"},
    )
    assert past.status_code == 422
    assert past.json()["error"]["code"] == "slot_in_past"

    mismatch = client.post(
        "/api/v1/predictions/attendance",
        json={**body, "zone_group_id": "G2"},
    )
    assert mismatch.status_code == 422
    assert mismatch.json()["error"]["code"] == "zone_group_mismatch"


def test_decision_generation_explains_when_today_has_no_future_slots(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client.post("/api/v1/ml/train")
    monkeypatch.setattr(
        "noctura_api.api.predict_remaining_today",
        lambda *_args, **_kwargs: {"items": [], "date": "2026-10-17"},
    )
    response = client.post("/api/v1/lighting/decisions/generate")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "no_future_slots"

    tomorrow = (datetime.now(ZoneInfo("Europe/Paris")) + timedelta(days=1)).date()

    def forecast_for_day(*_args, day: str = "today", **_kwargs):
        if day == "today":
            return {"items": [], "date": "2026-10-17"}
        return {
            "items": [
                {
                    "zone": "quai_est",
                    "time_slot": "20h-22h",
                    "predicted": 8.0,
                    "unit": "individuals_per_two_hour_slot",
                }
            ],
            "date": tomorrow.isoformat(),
            "port": "Port de Porquerolles",
        }

    monkeypatch.setattr("noctura_api.api.predict_remaining_today", forecast_for_day)
    tomorrow_response = client.post(
        "/api/v1/lighting/decisions/generate?day=tomorrow&zone=quai_est"
    )
    assert tomorrow_response.status_code == 200
    assert tomorrow_response.json()["items"][0]["date"] == tomorrow.isoformat()


def test_energy_report_does_not_claim_unmeasured_savings(client: TestClient) -> None:
    report = client.get("/api/v1/reports/energy").json()
    assert report["measured_energy_kwh"] is None
    assert report["estimated_savings_percent"] is None
    assert report["simulation_forecast_available"] is True
    assert report["simulation_forecast_endpoint"] == "/api/v1/energy/forecast?day=today"


def test_openapi_and_validation_errors_are_available(client: TestClient) -> None:
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    schema = openapi.json()
    assert "/api/v1/datasets/attendance" in schema["paths"]
    assert "examples" in schema["components"]["schemas"]["WeatherImport"]
    forecast_params = schema["paths"]["/api/v1/predictions/attendance/today"]["get"][
        "parameters"
    ]
    assert {parameter["name"] for parameter in forecast_params} == {"zone", "day"}
    invalid = client.post("/api/v1/predictions/attendance", json={
        "hour_start": 13,
        "zone_group_id": "G1",
        "zone_specifique": "quai_est",
        "is_daylight": 1,
        "is_winter": 0,
        "is_storm": 0,
        "indice_global": 11,
    })
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_request"
