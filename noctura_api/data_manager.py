from __future__ import annotations

import json
import io
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import pandas as pd

from data.validate_clean import COLUMNS, GROUPS, ZONES, parse as parse_attendance, row_rules

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIODIVERSITY_COLUMNS = [
    "espece",
    "nom_commun",
    "nom_scientifique",
    "taxon_indetermine",
    "annee",
    "effectif",
]
ENERGY_COLUMNS = [
    "date",
    "hour_start",
    "hour_end",
    "zone_group_id",
    "zone_group_name",
    "zone_specifique",
    "puissance_installee_w",
    "puissance_w",
    "energie_kwh",
    "eclairage_pct",
]


class DatasetValidationError(ValueError):
    def __init__(self, message: str, errors: list[str]) -> None:
        super().__init__(message)
        self.errors = errors


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(value, temporary, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def validate_attendance(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        raise DatasetValidationError(
            "Le CSV de fréquentation contient des noms de colonnes en double.",
            [f"Colonnes en double : {duplicate_columns}"],
        )
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetValidationError(
            "Colonnes de fréquentation manquantes.", [f"Colonnes manquantes : {missing}"]
        )
    if frame.empty:
        raise DatasetValidationError("Le fichier de fréquentation est vide.", ["aucune ligne"])

    warnings = []
    extra = [column for column in frame.columns if column not in COLUMNS]
    if extra:
        warnings.append(f"Colonnes supplémentaires ignorées : {extra}")

    numeric_columns = [
        "hour_start", "hour_end", "sin_hour", "cos_hour", "is_daylight", "is_winter",
        "is_storm", "is_synthetic_daytime", "indice_global", "frequentation_target",
    ]
    for column in numeric_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if np.isinf(numeric).any():
            raise DatasetValidationError(
                "Le CSV de fréquentation contient des valeurs numériques non finies.",
                [f"{column} : valeur infinie"],
            )

    parsed = parse_attendance(frame[COLUMNS])
    rules = row_rules(parsed)
    invalid_rows = ~pd.concat(rules.values(), axis=1).all(axis=1)
    errors = [
        f"{name} : {int((~valid).sum())} ligne(s)"
        for name, valid in rules.items()
        if (~valid).any()
    ]
    duplicates = parsed.duplicated()
    if duplicates.any():
        warnings.append(
            f"Doublons exacts conservés : {int(duplicates.sum())} ligne(s)."
        )
    if invalid_rows.any():
        raise DatasetValidationError(
            "Le fichier de fréquentation contient des lignes invalides.", errors
        )

    hours = parsed["hour_start"].astype("float64")
    parsed["sin_hour"] = np.sin(2 * np.pi * hours / 24).round(10) + 0.0
    parsed["cos_hour"] = np.cos(2 * np.pi * hours / 24).round(10) + 0.0
    for column in (
        "hour_start",
        "hour_end",
        "indice_global",
        "frequentation_target",
        "is_daylight",
        "is_winter",
        "is_storm",
        "is_synthetic_daytime",
    ):
        parsed[column] = parsed[column].astype("int64")
    unknown_zones = sorted(set(parsed["zone_specifique"]) - set(ZONES))
    unknown_groups = sorted(set(parsed["zone_group_id"]) - set(GROUPS))
    if unknown_zones:
        warnings.append(f"Zones hors référentiel : {unknown_zones}")
    if unknown_groups:
        warnings.append(f"Groupes hors référentiel : {unknown_groups}")
    for column in ("is_daylight", "is_winter", "is_storm", "is_synthetic_daytime"):
        if parsed[column].nunique() == 1:
            warnings.append(
                f"{column} est constante ({int(parsed[column].iloc[0])}) : "
                "aucune information discriminante pour l'entraînement."
            )
    return parsed[COLUMNS], warnings


def validate_biodiversity(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        raise DatasetValidationError(
            "Le CSV de biodiversité contient des noms de colonnes en double.",
            [f"Colonnes en double : {duplicate_columns}"],
        )
    missing = [column for column in BIODIVERSITY_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetValidationError(
            "Colonnes de biodiversité manquantes.", [f"Colonnes manquantes : {missing}"]
        )
    if frame.empty:
        raise DatasetValidationError("Le fichier de biodiversité est vide.", ["aucune ligne"])

    frame = frame[BIODIVERSITY_COLUMNS].copy()
    errors: list[str] = []
    for column in ("espece", "nom_scientifique"):
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or (frame[column] == "").any():
            errors.append(f"{column} : valeur manquante")
    frame["nom_commun"] = frame["nom_commun"].astype("string").str.strip()

    for column in ("taxon_indetermine", "annee", "effectif"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if (
            numeric.isna().any()
            or not np.isfinite(numeric.astype("float64")).all()
            or (numeric != numeric.round()).any()
        ):
            errors.append(f"{column} : valeur manquante ou non entière")
        elif (numeric < 0).any():
            errors.append(f"{column} : valeur négative")
        frame[column] = numeric
    if not errors:
        if not frame["taxon_indetermine"].isin([0, 1]).all():
            errors.append("taxon_indetermine : valeurs autorisées 0 ou 1")
        if not frame["annee"].between(1000, 9999).all():
            errors.append("annee : format YYYY requis")
        if frame.duplicated(["espece", "annee"]).any():
            errors.append("doublons espece/annee")
    if errors:
        raise DatasetValidationError("Le fichier de biodiversité contient des erreurs.", errors)

    frame["taxon_indetermine"] = frame["taxon_indetermine"].astype("int64")
    frame["annee"] = frame["annee"].astype("int64")
    frame["effectif"] = frame["effectif"].astype("int64")
    return frame, []


class DataManager:
    def __init__(self, storage_dir: Path | None = None) -> None:
        configured = storage_dir or Path(
            os.environ.get("NOCTURA_STORAGE_DIR", PROJECT_ROOT / "storage")
        )
        self.storage_dir = Path(configured)
        self.datasets_dir = self.storage_dir / "datasets"
        self.state_dir = self.storage_dir / "state"
        self.models_dir = self.storage_dir / "models"
        for directory in (self.datasets_dir, self.state_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _status_path(self) -> Path:
        return self.state_dir / "datasets.json"

    def _statuses(self) -> dict[str, Any]:
        path = self._status_path()
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _save_status(self, source: str, status: dict[str, Any]) -> None:
        with self._lock:
            statuses = self._statuses()
            statuses[source] = status
            _write_json(self._status_path(), statuses)

    def _store_csv(self, source: str, frame: pd.DataFrame, filename: str) -> None:
        path = self.datasets_dir / filename
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=self.datasets_dir,
            suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            frame.to_csv(temporary, index=False)
        os.replace(temporary_path, path)
        self._save_status(
            source,
            {
                "available": True,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "validation_status": "valid",
                "quality": {"valid_rows": int(len(frame)), "invalid_rows": 0},
                "warnings": [],
                "origin": "uploaded",
            },
        )

    def import_attendance(self, payload: bytes) -> dict[str, Any]:
        try:
            raw = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DatasetValidationError(
                "Impossible de lire le CSV de fréquentation.", [str(exc)]
            ) from exc
        parsed, warnings = validate_attendance(raw)
        self._store_csv("attendance", parsed, "attendance.csv")
        status = self._statuses()["attendance"]
        status["warnings"] = warnings
        status["quality"]["synthetic_rows"] = int(
            parsed["is_synthetic_daytime"].sum()
        )
        self._save_status("attendance", status)
        return {"rows": len(parsed), "synthetic_rows": int(parsed["is_synthetic_daytime"].sum()),
                "warnings": warnings}

    def import_biodiversity(self, payload: bytes) -> dict[str, Any]:
        try:
            raw = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DatasetValidationError(
                "Impossible de lire le CSV de biodiversité.", [str(exc)]
            ) from exc
        parsed, warnings = validate_biodiversity(raw)
        self._store_csv("biodiversity", parsed, "biodiversity.csv")
        status = self._statuses()["biodiversity"]
        status["warnings"] = warnings
        self._save_status("biodiversity", status)
        return {"rows": len(parsed), "species": int(parsed["espece"].nunique()),
                "warnings": warnings}

    def import_weather(
        self, timestamp: datetime, source: str, measurements: dict[str, Any]
    ) -> dict[str, Any]:
        payload = {
            "timestamp": timestamp.isoformat(),
            "source": source,
            "measurements": measurements,
        }
        _write_json(self.datasets_dir / "weather.json", payload)
        status = {
            "available": True,
            "last_updated": timestamp.isoformat(),
            "validation_status": "valid",
            "quality": {"measurements": len(measurements)},
            "warnings": [],
            "origin": source,
        }
        self._save_status("weather", status)
        return payload

    def latest_weather(self) -> dict[str, Any] | None:
        path = self.datasets_dir / "weather.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def record_recommendation(self, recommendation: dict[str, Any]) -> None:
        path = self.state_dir / "recommendations.json"
        with self._lock:
            recommendations = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            recommendations.append(recommendation)
            _write_json(path, recommendations)

    def lighting_decisions(self) -> list[dict[str, Any]]:
        path = self.state_dir / "lighting_decisions.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def save_lighting_decision(self, decision: dict[str, Any]) -> None:
        path = self.state_dir / "lighting_decisions.json"
        with self._lock:
            decisions = self.lighting_decisions()
            decisions.append(decision)
            _write_json(path, decisions)

    def update_lighting_decision(
        self, decision_id: str, updated_decision: dict[str, Any]
    ) -> bool:
        path = self.state_dir / "lighting_decisions.json"
        with self._lock:
            decisions = self.lighting_decisions()
            for index, decision in enumerate(decisions):
                if decision["decision_id"] == decision_id:
                    decisions[index] = updated_decision
                    _write_json(path, decisions)
                    return True
        return False

    def read_attendance(self) -> pd.DataFrame:
        uploaded = self.datasets_dir / "attendance.csv"
        path = uploaded if uploaded.exists() else PROJECT_ROOT / "data" / "dataset_ml_frequentation_24h_clean.csv"
        if not path.exists():
            raise FileNotFoundError("Aucune donnée de fréquentation importée ou fournie.")
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        parsed, _ = validate_attendance(raw)
        return parsed

    def read_biodiversity(self) -> pd.DataFrame | None:
        uploaded = self.datasets_dir / "biodiversity.csv"
        path = uploaded if uploaded.exists() else PROJECT_ROOT / "data" / "biodev_1_tidy.csv"
        if not path.exists():
            return None
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        parsed, _ = validate_biodiversity(raw)
        return parsed

    def read_energy_history(self) -> pd.DataFrame:
        path = PROJECT_ROOT / "data" / "eclairage_historique_simule.csv"
        if not path.exists():
            raise FileNotFoundError(
                "Le fichier de simulation énergétique est absent."
            )
        frame = pd.read_csv(path)
        missing = [column for column in ENERGY_COLUMNS if column not in frame.columns]
        if missing:
            raise DatasetValidationError(
                "Le fichier de simulation énergétique a un schéma invalide.",
                [f"Colonnes manquantes : {missing}"],
            )
        if frame.empty:
            raise DatasetValidationError(
                "Le fichier de simulation énergétique est vide.", ["aucune ligne"]
            )
        for column in (
            "hour_start",
            "hour_end",
            "puissance_installee_w",
            "puissance_w",
            "energie_kwh",
            "eclairage_pct",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame[column].isna().any() or not np.isfinite(frame[column]).all():
                raise DatasetValidationError(
                    "Le fichier de simulation énergétique contient des valeurs invalides.",
                    [f"{column} : valeur manquante ou non finie"],
                )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].isna().any():
            raise DatasetValidationError(
                "Le fichier de simulation énergétique contient des dates invalides.",
                ["date : valeur manquante ou invalide"],
            )
        if (
            (frame["hour_start"] < 0).any()
            or (frame["hour_start"] > 22).any()
            or (frame["hour_start"] % 2 != 0).any()
            or (frame["hour_end"] != (frame["hour_start"] + 2) % 24).any()
            or (frame["puissance_installee_w"] <= 0).any()
            or (frame["puissance_w"] < 0).any()
            or (frame["energie_kwh"] < 0).any()
            or (frame["eclairage_pct"] < 0).any()
            or (frame["eclairage_pct"] > 100).any()
            or frame["zone_specifique"].isna().any()
        ):
            raise DatasetValidationError(
                "Le fichier de simulation énergétique contient des lignes incohérentes.",
                ["créneau, zone, puissance et énergie doivent être valides"],
            )
        duplicate_slots = frame.duplicated(
            ["date", "hour_start", "zone_specifique"]
        )
        if duplicate_slots.any():
            raise DatasetValidationError(
                "Le fichier de simulation énergétique contient des doublons.",
                [f"créneau/date/zone : {int(duplicate_slots.sum())} doublon(s)"],
            )
        return frame

    def dataset_status(self) -> dict[str, Any]:
        statuses = self._statuses()
        for source, filename, seed in (
            ("attendance", "attendance.csv", PROJECT_ROOT / "data" / "dataset_ml_frequentation_24h_clean.csv"),
            ("biodiversity", "biodiversity.csv", PROJECT_ROOT / "data" / "biodev_1_tidy.csv"),
        ):
            if source in statuses:
                continue
            path = self.datasets_dir / filename
            if not path.exists():
                path = seed
            if path.exists():
                if source == "attendance":
                    frame, warnings = validate_attendance(
                        pd.read_csv(path, dtype=str, keep_default_na=False)
                    )
                else:
                    frame = validate_biodiversity(
                        pd.read_csv(path, dtype=str, keep_default_na=False)
                    )[0]
                    warnings = []
                statuses[source] = {
                    "available": True,
                    "last_updated": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                    "validation_status": "valid",
                    "quality": {"valid_rows": len(frame), "invalid_rows": 0},
                    "warnings": warnings,
                    "origin": "uploaded" if path.parent == self.datasets_dir else "bundled",
                }
        for source in ("weather", "biodiversity", "attendance"):
            statuses.setdefault(
                source,
                {
                    "available": False,
                    "last_updated": None,
                    "validation_status": "not_imported",
                    "quality": None,
                    "warnings": [],
                    "origin": None,
                },
            )
        if statuses["attendance"]["available"] and statuses["attendance"]["origin"] == "bundled":
            attendance = self.read_attendance()
            statuses["attendance"]["quality"]["synthetic_rows"] = int(
                attendance["is_synthetic_daytime"].sum()
            )
        return {"sources": statuses}
