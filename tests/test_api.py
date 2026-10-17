import unittest
from unittest import mock

from fastapi.testclient import TestClient

import api
from ml import features as F

client = TestClient(api.app)


class TestApi(unittest.TestCase):
    def test_sante(self):
        r = client.get("/api/sante")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["statut"], "ok")

    def test_decision_scenario(self):
        r = client.get("/api/decision", params={"date": "2025-01-15", "heure": "03:00", "scenario": "tempete", "mode": "profil"})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(len(d["zones"]), len(F.ZONES))
        self.assertEqual(d["contexte"]["is_storm"], 1)
        self.assertTrue(all(z["niveau"] >= 50 for z in d["zones"]))
        self.assertIn("scénario", d["meteo_source"])

    def test_decision_ml(self):
        r = client.get("/api/decision", params={"date": "2025-07-26", "heure": "23:00", "scenario": "calme", "mode": "ml"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(all(0 <= z["niveau"] <= 100 for z in r.json()["zones"]))

    def test_journee(self):
        r = client.get("/api/decision/journee", params={"date": "2025-07-26", "scenario": "calme", "mode": "profil"})
        self.assertEqual(r.status_code, 200)
        journee = r.json()
        self.assertEqual(len(journee), 12)
        self.assertEqual(journee[6]["puissance_moyenne"], 0)  # 13h00 : plein jour

    def test_parametres_invalides(self):
        self.assertEqual(client.get("/api/decision", params={"date": "15/01/2025"}).status_code, 422)
        self.assertEqual(client.get("/api/decision", params={"scenario": "ouragan"}).status_code, 422)
        self.assertEqual(client.get("/api/decision", params={"mode": "magie"}).status_code, 422)

    def test_meteo_live_simulee(self):
        with mock.patch("cerveau.prediction._meteo_live_en_cache", return_value={"temperature_c": 20.0}):
            r = client.get("/api/meteo")
        self.assertEqual(r.json()["temperature_c"], 20.0)

    def test_donnees(self):
        zones = client.get("/api/zones").json()
        self.assertIn("quai_sud_commerces", zones["zones_critiques"])
        self.assertEqual(len(zones["zones"]), len(F.ZONES))
        self.assertIn("tempete", client.get("/api/scenarios").json())
        self.assertIn(client.get("/api/biodiversite").status_code, (200, 404))
        self.assertIn("ml", client.get("/api/modele").json())


if __name__ == "__main__":
    unittest.main()
