from __future__ import annotations

from typing import Any


class SafetyRules:
    """Bloque toute consigne opérationnelle tant que des seuils validés manquent."""

    def evaluate(self) -> dict[str, Any]:
        return {
            "can_operate": False,
            "safety_check": "not_configured",
            "ecological_check": "pending_validation",
            "constraints_applied": [],
            "warnings": [
                "Aucun seuil de sécurité validé n'est configuré.",
                "Les contraintes écologiques doivent être validées par un responsable.",
            ],
        }
