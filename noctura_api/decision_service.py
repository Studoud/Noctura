from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any, Literal

import pandas as pd

from noctura_api.data_manager import DataManager

DEMO_RULES = {
    "mode": "simulation_only",
    "attendance_high_threshold": 15,
    "attendance_low_threshold": 5,
    "brightness_step_percentage_points": 10,
    "minimum_brightness_percent": 0,
    "maximum_brightness_percent": 100,
}


class DecisionHistoryError(RuntimeError):
    pass


def create_demo_decisions(
    data_manager: DataManager,
    forecast: dict[str, Any],
    *,
    zone: str | None = None,
) -> dict[str, Any]:
    try:
        history = data_manager.read_energy_history()
    except (FileNotFoundError, ValueError) as exc:
        raise DecisionHistoryError(str(exc)) from exc

    if "eclairage_pct" not in history.columns:
        raise DecisionHistoryError(
            "Le jeu historique ne contient pas la luminosité requise pour "
            "simuler les changements."
        )
    target_date = date.fromisoformat(forecast["date"])
    history["history_month"] = history["date"].dt.month
    history["history_weekday"] = history["date"].dt.dayofweek
    now = datetime.now(timezone.utc).isoformat()
    decisions: list[dict[str, Any]] = []

    for prediction in forecast["items"]:
        if zone is not None and prediction["zone"] != zone:
            continue
        zone_rows = history.loc[
            (history["zone_specifique"] == prediction["zone"])
            & (history["hour_start"] == int(prediction["time_slot"][:2]))
        ]
        if zone_rows.empty:
            raise DecisionHistoryError(
                f"Aucune référence de luminosité disponible pour {prediction['zone']} "
                f"au créneau {prediction['time_slot']}."
            )
        comparable = zone_rows.loc[
            (zone_rows["history_month"] == target_date.month)
            & (zone_rows["history_weekday"] == target_date.weekday())
        ]
        baseline_method = "same_month_and_weekday_median"
        if comparable.empty:
            comparable = zone_rows
            baseline_method = "all_available_days_same_slot_median"
        current_brightness = int(round(float(comparable["eclairage_pct"].median())))
        predicted_attendance = float(prediction["predicted"])
        action: Literal["increase", "decrease", "maintain"]
        reason: str
        delta = 0

        if predicted_attendance >= DEMO_RULES["attendance_high_threshold"]:
            action = "increase"
            reason = (
                "La fréquentation prévue atteint le seuil élevé de démonstration "
                f"({DEMO_RULES['attendance_high_threshold']} personnes)."
            )
            delta = DEMO_RULES["brightness_step_percentage_points"]
        elif predicted_attendance <= DEMO_RULES["attendance_low_threshold"]:
            action = "decrease"
            reason = (
                "La fréquentation prévue est sous le seuil bas de démonstration "
                f"({DEMO_RULES['attendance_low_threshold']} personnes)."
            )
            delta = -DEMO_RULES["brightness_step_percentage_points"]
        else:
            action = "maintain"
            reason = (
                "La fréquentation prévue est dans la plage neutre de démonstration."
            )

        proposed_brightness = max(
            DEMO_RULES["minimum_brightness_percent"],
            min(
                DEMO_RULES["maximum_brightness_percent"],
                current_brightness + delta,
            ),
        )
        actual_delta = proposed_brightness - current_brightness
        if actual_delta == 0 and action != "maintain":
            action = "maintain"
            reason += " La limite simulée de luminosité est déjà atteinte."

        decision = {
            "decision_id": str(uuid.uuid4()),
            "port": forecast["port"],
            "date": forecast["date"],
            "zone": prediction["zone"],
            "time_slot": prediction["time_slot"],
            "predicted_attendance": predicted_attendance,
            "attendance_unit": prediction["unit"],
            "action": action,
            "current_brightness_percent": current_brightness,
            "proposed_brightness_percent": proposed_brightness,
            "change_percentage_points": actual_delta,
            "reason": reason,
            "reason_codes": (
                ["high_predicted_attendance"]
                if action == "increase"
                else ["low_predicted_attendance"]
                if action == "decrease"
                else ["within_demo_neutral_range"]
            ),
            "policy": DEMO_RULES,
            "baseline_method": baseline_method,
            "status": "pending_human_review",
            "mode": "simulation_only",
            "hardware_command_sent": False,
            "generated_at": now,
            "review": None,
            "warnings": [
                "Règles de démonstration non validées pour l'exploitation réelle.",
                "La proposition et son approbation ne commandent aucun luminaire.",
                "Les niveaux actuels sont des médianes historiques simulées, pas des mesures en direct.",
            ],
        }
        data_manager.save_lighting_decision(decision)
        decisions.append(decision)

    return {
        "status": "simulation_only",
        "date": forecast["date"],
        "port": forecast["port"],
        "decision_count": len(decisions),
        "rules": DEMO_RULES,
        "items": decisions,
        "warnings": [
            "Ces propositions sont une démonstration, non des consignes validées.",
            "Aucune commande matérielle n'est envoyée, même après approbation frontend.",
        ],
    }


def list_demo_decisions(
    data_manager: DataManager,
    *,
    date_filter: date | None = None,
    zone: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    decisions = data_manager.lighting_decisions()
    if date_filter:
        decisions = [item for item in decisions if item["date"] == date_filter.isoformat()]
    if zone:
        decisions = [item for item in decisions if item["zone"] == zone]
    if status:
        decisions = [item for item in decisions if item["status"] == status]
    return {"count": len(decisions), "items": decisions}


def review_demo_decision(
    data_manager: DataManager,
    *,
    decision_id: str,
    action: Literal["approve", "reject"],
    reviewer: str,
    note: str | None = None,
) -> dict[str, Any] | None:
    decisions = data_manager.lighting_decisions()
    for decision in decisions:
        if decision["decision_id"] != decision_id:
            continue
        if decision["status"] != "pending_human_review":
            return None
        decision["status"] = (
            "approved_simulation" if action == "approve" else "rejected"
        )
        decision["review"] = {
            "action": action,
            "reviewer": reviewer,
            "note": note,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        if not data_manager.update_lighting_decision(decision_id, decision):
            return None
        return decision
    return None
