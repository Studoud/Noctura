from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np

from noctura_api.data_manager import DataManager
from noctura_api.port_config import (
    PORT_LATITUDE,
    PORT_LONGITUDE,
    PORT_NAME,
    PORT_TIMEZONE,
)

EnergyDay = Literal["today", "tomorrow"]
MONTE_CARLO_ITERATIONS = 10_000


class EnergyHistoryError(RuntimeError):
    pass


def _quantiles(values: np.ndarray) -> dict[str, float]:
    p05, p50, p95 = np.quantile(values, [0.05, 0.5, 0.95])
    return {
        "p05": round(float(p05), 2),
        "p50": round(float(p50), 2),
        "p95": round(float(p95), 2),
    }


def forecast_energy(
    data_manager: DataManager,
    *,
    day: EnergyDay = "today",
    now: datetime | None = None,
    simulations: int = MONTE_CARLO_ITERATIONS,
) -> dict[str, Any]:
    if simulations < 1:
        raise ValueError("Le nombre de simulations doit être supérieur à zéro.")

    timezone = ZoneInfo(PORT_TIMEZONE)
    current_time = (now or datetime.now(timezone)).astimezone(timezone)
    target_date = current_time.date() + timedelta(days=day == "tomorrow")

    try:
        history = data_manager.read_energy_history()
    except (FileNotFoundError, ValueError) as exc:
        raise EnergyHistoryError(str(exc)) from exc

    history = history.copy()
    history["history_month"] = history["date"].dt.month
    history["history_weekday"] = history["date"].dt.dayofweek
    comparable = history.loc[
        (history["history_month"] == target_date.month)
        & (history["history_weekday"] == target_date.weekday())
    ]
    reference_method = "same_month_and_weekday_daily_profile_bootstrap"
    if comparable.empty:
        comparable = history.loc[history["history_month"] == target_date.month]
        reference_method = "same_month_daily_profile_bootstrap"
    if comparable.empty:
        comparable = history
        reference_method = "all_available_daily_profiles_bootstrap"

    daily_profiles = comparable.pivot_table(
        index="date",
        columns="zone_specifique",
        values="energie_kwh",
        aggfunc="sum",
    ).sort_index()
    expected_zones = sorted(history["zone_specifique"].unique())
    daily_profiles = daily_profiles.reindex(columns=expected_zones).dropna()
    if daily_profiles.empty:
        raise EnergyHistoryError(
            "Aucun profil journalier complet n'est disponible pour la simulation."
        )

    installed_power: dict[str, float] = {}
    for zone, zone_history in history.groupby("zone_specifique", sort=False):
        power_values = zone_history["puissance_installee_w"].unique()
        if len(power_values) != 1:
            raise EnergyHistoryError(
                f"La puissance installée de la zone {zone} est incohérente."
            )
        installed_power[zone] = float(power_values[0])

    seed = int(target_date.strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)
    sampled_rows = rng.integers(
        0, len(daily_profiles), size=simulations, endpoint=False
    )
    sampled_profiles = daily_profiles.to_numpy(dtype="float64")[sampled_rows]
    zone_index = {zone: index for index, zone in enumerate(daily_profiles.columns)}
    sampled_zone_energy = {
        zone: sampled_profiles[:, zone_index[zone]] for zone in daily_profiles.columns
    }
    sampled_total_energy = sampled_profiles.sum(axis=1)

    total_reference = sum(
        power * 24 / 1000 for power in installed_power.values()
    )
    sampled_savings = total_reference - sampled_total_energy
    sampled_savings_percent = sampled_savings / total_reference * 100

    zones: list[dict[str, Any]] = []
    for zone in daily_profiles.columns:
        energy_samples = sampled_zone_energy[zone]
        reference_kwh = installed_power[zone] * 24 / 1000
        savings_samples = reference_kwh - energy_samples
        zones.append(
            {
                "zone": zone,
                "predicted_consumption_kwh": _quantiles(energy_samples)["p50"],
                "consumption_interval_90_kwh": _quantiles(energy_samples),
                "full_power_reference_kwh": round(reference_kwh, 2),
                "potential_savings_kwh": _quantiles(savings_samples)["p50"],
                "potential_savings_interval_90_kwh": _quantiles(savings_samples),
                "potential_savings_percent": _quantiles(
                    savings_samples / reference_kwh * 100
                )["p50"],
                "probability_of_savings_percent": round(
                    float(np.mean(savings_samples > 0) * 100), 2
                ),
                "share_of_total_percent": round(
                    float(
                        np.median(energy_samples)
                        / np.median(sampled_total_energy)
                        * 100
                    ),
                    2,
                ),
                "historical_reference_method": reference_method,
            }
        )

    zones.sort(key=lambda item: item["predicted_consumption_kwh"], reverse=True)
    unique_profiles = len(daily_profiles)
    warnings = [
        "Toutes les simulations rééchantillonnent des profils journaliers simulés de 2025; aucune mesure réelle de compteur n'est disponible.",
        f"Le Monte-Carlo fait {simulations:,} tirages avec remise à partir des profils historiques comparables. Le nombre de profils historiques distincts est indiqué; ces tirages ne signifient pas autant de journées mesurées.",
        "Les économies sont comparées à une référence théorique de fonctionnement à 100 % de la puissance installée, pas à un scénario opérationnel validé.",
    ]
    if unique_profiles < 10:
        warnings.append(
            f"Seulement {unique_profiles} journées historiques comparables sont "
            "disponibles; les intervalles et probabilités sont indicatifs et "
            "ne sont pas calibrés sur des mesures réelles."
        )
    return {
        "status": "estimated_simulation",
        "port": PORT_NAME,
        "location": {
            "latitude": PORT_LATITUDE,
            "longitude": PORT_LONGITUDE,
            "timezone": PORT_TIMEZONE,
        },
        "date": target_date.isoformat(),
        "day": day,
        "period": "full_day",
        "unit": "kWh",
        "data_source": "data/eclairage_historique_simule.csv",
        "estimation_method": "monte_carlo_bootstrap_of_comparable_daily_energy_profiles",
        "savings_reference": "installed_power_at_100_percent_for_24_hours",
        "monte_carlo": {
            "simulations": simulations,
            "historical_profiles_used": unique_profiles,
            "historical_dates_used": [
                value.date().isoformat() for value in daily_profiles.index
            ],
            "sampling": "bootstrap_with_replacement_by_complete_day",
            "seed": seed,
            "interval": "p05_p50_p95",
        },
        "summary": {
            "predicted_consumption_kwh": _quantiles(sampled_total_energy)["p50"],
            "consumption_interval_90_kwh": _quantiles(sampled_total_energy),
            "full_power_reference_kwh": round(total_reference, 2),
            "potential_savings_kwh": _quantiles(sampled_savings)["p50"],
            "potential_savings_interval_90_kwh": _quantiles(sampled_savings),
            "potential_savings_percent": _quantiles(sampled_savings_percent)["p50"],
            "potential_savings_percent_interval_90": _quantiles(
                sampled_savings_percent
            ),
            "probability_of_savings_percent": round(
                float(np.mean(sampled_savings > 0) * 100), 2
            ),
        },
        "top_consuming_zones": zones[:3],
        "zones": zones,
        "warnings": warnings,
    }
