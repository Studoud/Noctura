from __future__ import annotations

import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from noctura_api.data_manager import DataManager, DatasetValidationError
from noctura_api.decision_service import (
    DecisionHistoryError,
    create_demo_decisions,
    list_demo_decisions,
    review_demo_decision,
)
from noctura_api.energy_forecast import (
    EnergyHistoryError,
    forecast_energy,
)
from noctura_api.forecast import (
    HistoricalReferenceError,
    WeatherForecastError,
    available_zones,
    predict_remaining_today,
)
from noctura_api.ml_service import ModelService, TrainingRejected
from noctura_api.port_config import (
    PORT_LATITUDE,
    PORT_LONGITUDE,
    PORT_NAME,
    PORT_TIMEZONE,
)
from noctura_api.reporting import ReportingService
from noctura_api.rules import SafetyRules
from noctura_api.schemas import (
    AttendancePrediction,
    DecisionReview,
    LightingRecommendationRequest,
    WeatherImport,
)

LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _api_error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, **extra}}


def _weather_measurement(measurements: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in measurements:
            return measurements[key]
    return None


def create_app(storage_dir: Path | None = None) -> FastAPI:
    data_manager = DataManager(storage_dir)
    model_service = ModelService(data_manager)
    reporting = ReportingService(data_manager, model_service)
    rules = SafetyRules()
    zones = {item["zone_specifique"]: item["zone_group_id"] for item in available_zones()}
    available_zones_by_name = set(zones)

    def validate_zone_group(zone: str, group: str) -> None:
        expected_group = zones.get(zone)
        if expected_group is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "unknown_zone",
                    "message": f"Zone inconnue : {zone}",
                    "available_zones": available_zones(),
                },
            )
        if group != expected_group:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "zone_group_mismatch",
                    "message": f"La zone {zone} appartient au groupe {expected_group}.",
                    "expected_zone_group_id": expected_group,
                },
            )

    app = FastAPI(
        title="Noctura Lighting ML API",
        version="1.0.0",
        description=(
            "API de validation des données, prédiction de fréquentation et "
            "recommandations non opérationnelles tant que les règles de sécurité "
            "et écologiques ne sont pas validées."
        ),
    )
    app.state.data_manager = data_manager
    app.state.model_service = model_service
    app.state.reporting = reporting

    configured_origins = os.environ.get(
        "NOCTURA_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in configured_origins.split(",") if origin.strip()],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_request: Any, exc: StarletteHTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            content = {"error": exc.detail}
        else:
            content = _api_error("http_error", str(exc.detail))
        return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Any, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_api_error(
                "invalid_request",
                "La requête ne respecte pas le schéma attendu.",
                details=jsonable_encoder(exc.errors()),
            ),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Any, exc: Exception) -> JSONResponse:
        LOGGER.error(
            "Erreur interne de l'API",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=500,
            content=_api_error(
                "internal_error",
                "Une erreur interne est survenue. Consultez les journaux du serveur.",
            ),
        )

    @app.get("/api/v1/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/datasets/weather", tags=["datasets"])
    def import_weather(payload: WeatherImport) -> dict[str, Any]:
        return {
            "status": "validated",
            "dataset": data_manager.import_weather(
                payload.timestamp, payload.source, payload.measurements
            ),
        }

    async def read_csv_upload(file: UploadFile) -> bytes:
        if not file.filename or Path(file.filename).suffix.lower() != ".csv":
            raise HTTPException(
                status_code=415,
                detail={"code": "unsupported_file", "message": "Un fichier .csv est requis."},
            )
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail={"code": "file_too_large", "message": "Le fichier dépasse 20 Mio."},
            )
        return payload

    @app.post("/api/v1/datasets/biodiversity", tags=["datasets"])
    async def import_biodiversity(file: UploadFile = File(...)) -> dict[str, Any]:
        payload = await read_csv_upload(file)
        try:
            result = data_manager.import_biodiversity(payload)
        except DatasetValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "dataset_validation_failed",
                    "message": str(exc),
                    "details": exc.errors,
                },
            ) from exc
        return {"status": "validated", "dataset": result}

    @app.post("/api/v1/datasets/attendance", tags=["datasets"])
    async def import_attendance(file: UploadFile = File(...)) -> dict[str, Any]:
        payload = await read_csv_upload(file)
        try:
            result = data_manager.import_attendance(payload)
        except DatasetValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "dataset_validation_failed",
                    "message": str(exc),
                    "details": exc.errors,
                },
            ) from exc
        return {"status": "validated", "dataset": result}

    @app.get("/api/v1/datasets/status", tags=["datasets"])
    def dataset_status() -> dict[str, Any]:
        return data_manager.dataset_status()

    @app.get("/api/v1/datasets/zones", tags=["datasets"])
    def dataset_zones() -> dict[str, Any]:
        zones = available_zones()
        return {"count": len(zones), "items": zones}

    @app.get("/api/v1/port", tags=["port"])
    def get_port() -> dict[str, Any]:
        return {
            "name": PORT_NAME,
            "location": {
                "latitude": PORT_LATITUDE,
                "longitude": PORT_LONGITUDE,
                "timezone": PORT_TIMEZONE,
            },
            "zones": available_zones(),
        }

    @app.post("/api/v1/ml/train", tags=["training"])
    def train_model() -> dict[str, Any]:
        try:
            return model_service.train()
        except TrainingRejected as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "training_rejected",
                    "message": str(exc),
                    "job_id": exc.job_id,
                },
            ) from exc

    @app.get("/api/v1/ml/train/{job_id}", tags=["training"])
    def get_training_job(job_id: str) -> dict[str, Any]:
        job = model_service.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "job_not_found", "message": "Job d'entraînement introuvable."},
            )
        return job

    @app.get("/api/v1/ml/model", tags=["training"])
    def get_model() -> dict[str, Any]:
        model = model_service.active_model()
        return {"status": "ready", "model": model} if model else {
            "status": "not_trained", "model": None
        }

    @app.get("/api/v1/ml/evaluations", tags=["training"])
    def evaluations() -> dict[str, Any]:
        completed = [
            {
                "job_id": job["job_id"],
                "completed_at": job["completed_at"],
                "model_version": job.get("model_version"),
                "metrics": job["metrics"],
                "validation": job.get("validation"),
                "training_rows": job.get("training_rows"),
                "excluded_synthetic_rows": job.get("excluded_synthetic_rows"),
            }
            for job in model_service.list_jobs()
            if job["status"] == "completed"
        ]
        return {"count": len(completed), "items": completed}

    @app.post("/api/v1/predictions/attendance", tags=["predictions"])
    def predict_attendance(payload: AttendancePrediction) -> dict[str, Any]:
        validate_zone_group(payload.zone_specifique, payload.zone_group_id)
        model = model_service.active_model()
        if model is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "model_not_trained",
                    "message": "Entraînez explicitement le modèle avant de demander une prédiction.",
                },
            )
        predicted = model_service.predict(payload.model_features())
        return {
            "zone": payload.zone_specifique,
            "time_slot": f"{payload.hour_start:02d}h-{(payload.hour_start + 2) % 24:02d}h",
            "predicted": predicted,
            "unit": "individuals_per_two_hour_slot",
            "model_version": model["version"],
            "status": "predicted",
            "warnings": ["Cette valeur est une estimation, pas une mesure réelle."],
        }

    @app.get("/api/v1/predictions/attendance/today", tags=["predictions"])
    def predict_attendance_today(
        zone: str | None = Query(default=None),
        day: Literal["today", "tomorrow"] = Query(default="today"),
    ) -> dict[str, Any]:
        try:
            return predict_remaining_today(
                data_manager, model_service, zone=zone, day=day
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "unknown_zone",
                    "message": f"Zone inconnue : {exc.args[0]}",
                    "available_zones": list(available_zones()),
                },
            ) from exc
        except HistoricalReferenceError as exc:
            code = "model_not_trained" if model_service.active_model() is None else "historical_reference_missing"
            raise HTTPException(
                status_code=409,
                detail={"code": code, "message": str(exc)},
            ) from exc
        except WeatherForecastError as exc:
            raise HTTPException(
                status_code=502,
                detail={"code": "weather_unavailable", "message": str(exc)},
            ) from exc

    @app.post("/api/v1/lighting/recommendations", tags=["lighting"])
    def create_recommendation(payload: LightingRecommendationRequest) -> dict[str, Any]:
        validate_zone_group(payload.zone_specifique, payload.zone_group_id)
        local_timezone = ZoneInfo(PORT_TIMEZONE)
        slot_start = payload.timestamp.astimezone(local_timezone).replace(
            hour=payload.hour_start, minute=0, second=0, microsecond=0
        )
        if slot_start <= datetime.now(local_timezone):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "slot_in_past",
                    "message": (
                        "Le créneau choisi a déjà commencé ou est terminé. "
                        "Choisissez un créneau futur à Porquerolles."
                    ),
                },
            )
        current_rules = rules.evaluate()
        model = model_service.active_model()
        if model is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "model_not_trained",
                    "message": "Entraînez explicitement le modèle avant de créer une recommandation.",
                },
            )
        predicted = model_service.predict(payload.model_features())
        weather = data_manager.latest_weather()
        measurements = weather["measurements"] if weather else {}
        weather_warnings = (
            []
            if weather
            else ["Aucune mesure météo enregistrée; les champs météo restent null."]
        )
        timestamp = payload.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        recommendation = {
            "zone": payload.zone_specifique,
            "timestamp": timestamp.isoformat(),
            "recommended_intensity_percent": None,
            "status": "pending_validation",
            "attendance": {
                "predicted": predicted,
                "time_slot": f"{payload.hour_start:02d}h-{(payload.hour_start + 2) % 24:02d}h",
            },
            "environment": {
                "weather_source": weather["source"] if weather else None,
                "natural_light_factor_percent": _weather_measurement(
                    measurements, "natural_light_factor_percent", "facteur_lumiere_naturelle"
                ),
                "cloud_cover_percent": _weather_measurement(
                    measurements, "cloud_cover_percent", "couverture_nuageuse"
                ),
                "measurements_timestamp": weather["timestamp"] if weather else None,
            },
            "constraints": {
                "safety_check": current_rules["safety_check"],
                "ecological_check": current_rules["ecological_check"],
                "constraints_applied": current_rules["constraints_applied"],
            },
            "explanation": [
                "Aucune intensité opérationnelle n'est calculée sans seuils de sécurité et contraintes écologiques validés.",
                *current_rules["warnings"],
                *weather_warnings,
            ],
            "warnings": [*current_rules["warnings"], *weather_warnings],
            "model_version": model["version"],
        }
        data_manager.record_recommendation(recommendation)
        return recommendation

    @app.get("/api/v1/lighting/recommendations", tags=["lighting"])
    def get_recommendations(
        date_filter: date | None = Query(default=None, alias="date"),
        zone: str | None = None,
        time_slot: str | None = None,
    ) -> dict[str, Any]:
        return reporting.decisions(date_filter=date_filter, zone=zone, time_slot=time_slot)

    @app.post("/api/v1/lighting/decisions/generate", tags=["lighting decisions"])
    def generate_lighting_decisions(
        zone: str | None = Query(default=None),
        day: Literal["today", "tomorrow"] = Query(default="today"),
    ) -> dict[str, Any]:
        if zone is not None and zone not in available_zones_by_name:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "unknown_zone",
                    "message": f"Zone inconnue : {zone}",
                    "available_zones": list(available_zones()),
                },
            )
        try:
            forecast = predict_remaining_today(
                data_manager, model_service, zone=zone, day=day
            )
            if not forecast["items"]:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "no_future_slots",
                        "message": (
                            "Aucun créneau futur ne reste aujourd'hui. "
                            "Demandez les propositions pour demain avec day=tomorrow."
                        ),
                    },
                )
            return create_demo_decisions(data_manager, forecast, zone=zone)
        except HistoricalReferenceError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "model_not_trained", "message": str(exc)},
            ) from exc
        except WeatherForecastError as exc:
            raise HTTPException(
                status_code=502,
                detail={"code": "weather_unavailable", "message": str(exc)},
            ) from exc
        except DecisionHistoryError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "decision_simulation_unavailable", "message": str(exc)},
            ) from exc

    @app.get("/api/v1/lighting/decisions", tags=["lighting decisions"])
    def get_lighting_decisions(
        date_filter: date | None = Query(default=None, alias="date"),
        zone: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        if zone is not None and zone not in available_zones_by_name:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "unknown_zone",
                    "message": f"Zone inconnue : {zone}",
                    "available_zones": list(available_zones()),
                },
            )
        return list_demo_decisions(
            data_manager, date_filter=date_filter, zone=zone, status=status
        )

    @app.post(
        "/api/v1/lighting/decisions/{decision_id}/review",
        tags=["lighting decisions"],
    )
    def review_lighting_decision(
        decision_id: str, payload: DecisionReview
    ) -> dict[str, Any]:
        decision = review_demo_decision(
            data_manager,
            decision_id=decision_id,
            action=payload.action,
            reviewer=payload.reviewer,
            note=payload.note,
        )
        if decision is None:
            existing = next(
                (
                    item
                    for item in data_manager.lighting_decisions()
                    if item["decision_id"] == decision_id
                ),
                None,
            )
            status_code = 409 if existing else 404
            code = "decision_already_reviewed" if existing else "decision_not_found"
            message = (
                "Cette proposition a déjà été examinée."
                if existing
                else "Proposition introuvable."
            )
            raise HTTPException(
                status_code=status_code,
                detail={"code": code, "message": message},
            )
        return {
            "status": decision["status"],
            "decision": decision,
            "simulation_only": True,
            "hardware_command_sent": False,
        }

    @app.get("/api/v1/reports/decisions", tags=["reports"])
    def decision_report(
        date_filter: date | None = Query(default=None, alias="date"),
        zone: str | None = None,
        time_slot: str | None = None,
    ) -> dict[str, Any]:
        return reporting.decisions(date_filter=date_filter, zone=zone, time_slot=time_slot)

    @app.get("/api/v1/reports/energy", tags=["reports"])
    def energy_report() -> dict[str, Any]:
        return reporting.energy()

    @app.get("/api/v1/energy/forecast", tags=["energy"])
    def energy_forecast(
        day: Literal["today", "tomorrow"] = Query(default="today"),
    ) -> dict[str, Any]:
        try:
            forecast = forecast_energy(data_manager, day=day)
        except EnergyHistoryError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "energy_simulation_unavailable",
                    "message": str(exc),
                },
            ) from exc
        active_model = model_service.active_model()
        forecast["model_performance"] = {
            "name": "Noctora",
            "task": "attendance_prediction",
            "status": "ready" if active_model else "not_trained",
            "version": active_model.get("version") if active_model else None,
            "model_type": active_model.get("model_type") if active_model else None,
            "metrics": active_model.get("metrics") if active_model else None,
            "validation": active_model.get("validation") if active_model else None,
            "training_rows": active_model.get("training_rows") if active_model else None,
            "note": (
                "Les métriques évaluent la prédiction de fréquentation; "
                "elles ne mesurent pas la précision de la simulation énergétique."
            ),
        }
        decision_report = reporting.decisions(
            date_filter=date.fromisoformat(forecast["date"])
        )
        decisions = decision_report["items"]
        forecast["decision_report"] = {
            "date": forecast["date"],
            "count": len(decisions),
            "pending_validation": sum(
                item.get("status") == "pending_validation" for item in decisions
            ),
            "items": decisions,
            "status": (
                "no_decisions_recorded"
                if not decisions
                else "pending_validation"
                if any(
                    item.get("status") == "pending_validation"
                    for item in decisions
                )
                else "recorded"
            ),
            "note": "Les décisions d'éclairage restent non opérationnelles tant que sécurité et écologie ne sont pas validées.",
        }
        return forecast

    @app.get("/api/v1/reports/biodiversity", tags=["reports"])
    def biodiversity_report() -> dict[str, Any]:
        return reporting.biodiversity()

    return app


app = create_app()
