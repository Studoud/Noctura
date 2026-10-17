from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from noctura_api.data_manager import DataManager, _write_json

LOGGER = logging.getLogger(__name__)
MIN_TRAINING_ROWS = 20
CATEGORICAL_FEATURES = ["zone_group_id", "zone_specifique"]
NUMERIC_FEATURES = [
    "hour_start",
    "sin_hour",
    "cos_hour",
    "is_daylight",
    "is_winter",
    "is_storm",
    "indice_global",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class TrainingRejected(ValueError):
    def __init__(self, job_id: str, message: str) -> None:
        super().__init__(message)
        self.job_id = job_id


class ModelService:
    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.jobs_dir = data_manager.state_dir / "training_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._training_lock = Lock()

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        try:
            normalized_id = str(uuid.UUID(job_id))
        except ValueError:
            return None
        path = self._job_path(normalized_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def list_jobs(self) -> list[dict[str, Any]]:
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.jobs_dir.glob("*.json"))
        ]

    def active_model(self) -> dict[str, Any] | None:
        path = self.data_manager.state_dir / "active_model.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def predict(self, features: dict[str, Any]) -> float | None:
        metadata = self.active_model()
        if metadata is None:
            return None
        model_path = self.data_manager.models_dir / f"{metadata['version']}.joblib"
        pipeline = joblib.load(model_path)
        prediction = pipeline.predict(pd.DataFrame([features]))[0]
        return round(max(0.0, float(prediction)), 2)

    def train(self) -> dict[str, Any]:
        with self._training_lock:
            return self._train()

    def _train(self) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "metrics": None,
            "error": None,
        }
        job_path = self._job_path(job_id)
        _write_json(job_path, job)
        try:
            data = self.data_manager.read_attendance()
            training = data.loc[data["is_synthetic_daytime"] == 0].copy()
            if len(training) < MIN_TRAINING_ROWS:
                raise TrainingRejected(
                    job_id,
                    f"Au moins {MIN_TRAINING_ROWS} lignes de fréquentation non synthétiques "
                    f"sont requises; {len(training)} disponible(s).",
                )
            if training["frequentation_target"].nunique() < 2:
                raise TrainingRejected(
                    job_id, "La cible doit contenir au moins deux valeurs distinctes."
                )

            X = training[FEATURES]
            y = training["frequentation_target"].astype("float64")
            feature_groups = pd.MultiIndex.from_frame(X).factorize()[0]
            if len(np.unique(feature_groups)) < 2:
                raise TrainingRejected(
                    job_id,
                    "Au moins deux combinaisons de caractéristiques distinctes "
                    "sont requises pour évaluer le modèle.",
                )
            train_indices, test_indices = next(
                GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(
                    X, y, groups=feature_groups
                )
            )
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            if len(X_train) < MIN_TRAINING_ROWS or len(X_test) < 2:
                raise TrainingRejected(
                    job_id,
                    "La séparation par groupes ne laisse pas assez de lignes "
                    "pour entraîner et évaluer le modèle.",
                )
            if y_train.nunique() < 2:
                raise TrainingRejected(
                    job_id,
                    "La cible d'entraînement doit contenir au moins deux valeurs "
                    "distinctes après la séparation par groupes.",
                )
            preprocessor = ColumnTransformer(
                [
                    ("numeric", "passthrough", NUMERIC_FEATURES),
                    ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
                ]
            )
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("regressor", DecisionTreeRegressor(
                        max_depth=4, min_samples_leaf=2, random_state=42
                    )),
                ]
            )
            pipeline.fit(X_train, y_train)
            holdout_predictions = pipeline.predict(X_test)
            baseline_predictions = np.full(len(y_test), y_train.mean())
            metrics = {
                "mae": round(float(mean_absolute_error(y_test, holdout_predictions)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, holdout_predictions))), 4),
                "r2": round(float(r2_score(y_test, holdout_predictions)), 4),
                "holdout_rows": int(len(X_test)),
                "baseline_mae": round(
                    float(mean_absolute_error(y_test, baseline_predictions)), 4
                ),
                "baseline_rmse": round(
                    float(np.sqrt(mean_squared_error(y_test, baseline_predictions))), 4
                ),
                "baseline_r2": round(
                    float(r2_score(y_test, baseline_predictions)), 4
                ),
                "holdout_feature_groups": int(
                    len(np.unique(feature_groups[test_indices]))
                ),
            }
            pipeline.fit(X, y)

            version = f"attendance-{uuid.uuid4().hex[:12]}"
            model_path = self.data_manager.models_dir / f"{version}.joblib"
            with tempfile.NamedTemporaryFile(
                dir=self.data_manager.models_dir, suffix=".joblib.tmp", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                joblib.dump(pipeline, temporary_path)
                os.replace(temporary_path, model_path)
            finally:
                temporary_path.unlink(missing_ok=True)

            trained_at = datetime.now(timezone.utc).isoformat()
            metadata = {
                "version": version,
                "trained_at": trained_at,
                "target": "frequentation_target",
                "model_type": "DecisionTreeRegressor(max_depth=4)",
                "metrics": metrics,
                "training_rows": int(len(training)),
                "excluded_synthetic_rows": int(len(data) - len(training)),
                "features": FEATURES,
                "validation": {
                    "method": "grouped_holdout_by_feature_vector",
                    "train_rows": int(len(X_train)),
                    "holdout_rows": int(len(X_test)),
                    "train_feature_groups": int(
                        len(np.unique(feature_groups[train_indices]))
                    ),
                    "holdout_feature_groups": int(
                        len(np.unique(feature_groups[test_indices]))
                    ),
                    "temporal_validation": False,
                    "limitations": [
                        (
                            "Le jeu de fréquentation utilisé pour entraîner Noctora "
                            "ne contient pas de date."
                        ),
                        (
                            "Le CSV d'éclairage contient des dates de 2025, mais "
                            "ses valeurs de fréquentation sont simulées et ne sont "
                            "pas les cibles observées du jeu d'entraînement Noctora."
                        ),
                        "Le holdout évalue des combinaisons de caractéristiques absentes de l'entraînement, pas la généralisation à de futurs jours observés.",
                        "La qualité des prédictions réelles ne peut pas être confirmée sans observations indépendantes.",
                    ],
                },
            }
            _write_json(self.data_manager.state_dir / "active_model.json", metadata)
            job.update(
                status="completed",
                completed_at=trained_at,
                metrics=metrics,
                validation=metadata["validation"],
                model_version=version,
                training_rows=int(len(training)),
                excluded_synthetic_rows=int(len(data) - len(training)),
            )
            _write_json(job_path, job)
            return job
        except TrainingRejected as exc:
            job.update(
                status="failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                error={"code": "insufficient_or_invalid_data", "message": str(exc)},
            )
            _write_json(job_path, job)
            raise
        except Exception:
            job.update(
                status="failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                error={
                    "code": "training_failed",
                    "message": "L'entraînement a échoué; le modèle actif précédent est conservé.",
                },
            )
            _write_json(job_path, job)
            LOGGER.exception("Échec de l'entraînement du job %s", job_id)
            raise
