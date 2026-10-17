from __future__ import annotations

import json
from datetime import date
from typing import Any

from noctura_api.data_manager import PROJECT_ROOT, DataManager
from noctura_api.ml_service import ModelService


class ReportingService:
    def __init__(self, data_manager: DataManager, model_service: ModelService) -> None:
        self.data_manager = data_manager
        self.model_service = model_service

    def decisions(
        self, *, date_filter: date | None = None, zone: str | None = None,
        time_slot: str | None = None
    ) -> dict[str, Any]:
        path = self.data_manager.state_dir / "recommendations.json"
        recommendations = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        )
        if date_filter:
            recommendations = [
                item for item in recommendations
                if item["timestamp"][:10] == date_filter.isoformat()
            ]
        if zone:
            recommendations = [item for item in recommendations if item["zone"] == zone]
        if time_slot:
            recommendations = [
                item for item in recommendations if item["attendance"]["time_slot"] == time_slot
            ]
        return {"count": len(recommendations), "items": recommendations}

    def energy(self) -> dict[str, Any]:
        return {
            "status": "not_measured",
            "measured_energy_kwh": None,
            "estimated_savings_percent": None,
            "simulation_forecast_available": (
                PROJECT_ROOT / "data" / "eclairage_historique_simule.csv"
            ).exists(),
            "simulation_forecast_endpoint": "/api/v1/energy/forecast?day=today",
            "explanation": (
                "Aucune mesure électrique réelle n'est disponible. Utilisez "
                "/api/v1/energy/forecast?day=today ou day=tomorrow pour une "
                "estimation explicitement basée sur le jeu de simulation."
            ),
        }

    def biodiversity(self) -> dict[str, Any]:
        data = self.data_manager.read_biodiversity()
        if data is None:
            return {
                "status": "not_imported",
                "species_count": None,
                "years": [],
                "indeterminate_species_count": None,
                "constraints": "pending_validation",
            }
        return {
            "status": "available",
            "species_count": int(data["espece"].nunique()),
            "years": sorted(int(year) for year in data["annee"].unique()),
            "indeterminate_species_count": int(
                data.loc[data["taxon_indetermine"] == 1, "espece"].nunique()
            ),
            "constraints": "pending_validation",
            "explanation": (
                "Ces données décrivent des observations; elles ne définissent pas "
                "à elles seules des seuils écologiques validés pour l'éclairage."
            ),
        }
