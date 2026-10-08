import unittest

from fastapi.testclient import TestClient

from app import app


class FastAPIAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_home_page_shows_prediction_form(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Student Academic Performance Predictor", response.text)
        self.assertIn("Predict", response.text)

    def test_prediction_endpoint_returns_result(self):
        response = self.client.post(
            "/predict",
            json={
                "school": "GP",
                "sex": "F",
                "age": 18,
                "address": "U",
                "famsize": "GT3",
                "Pstatus": "A",
                "Medu": 3,
                "Fedu": 2,
                "Mjob": "teacher",
                "Fjob": "services",
                "reason": "course",
                "guardian": "mother",
                "traveltime": 1,
                "studytime": 2,
                "failures": 0,
                "schoolsup": "yes",
                "famsup": "no",
                "paid": "no",
                "activities": "yes",
                "nursery": "yes",
                "higher": "yes",
                "internet": "yes",
                "romantic": "no",
                "famrel": 4,
                "freetime": 3,
                "goout": 2,
                "Dalc": 1,
                "Walc": 1,
                "health": 4,
                "absences": 4,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["prediction"], {"Low", "Medium", "High"})
        self.assertIn("probabilities", response.json())
        self.assertIn("recommendation", response.json())


if __name__ == "__main__":
    unittest.main()
