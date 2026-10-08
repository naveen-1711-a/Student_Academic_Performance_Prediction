import json
import tkinter as tk
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from tkinter import messagebox, ttk


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.pkl"
SCHEMA_PATH = PROJECT_ROOT / "models" / "input_schema.json"

app = FastAPI(title="Student Academic Performance Predictor")


class StudentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    school: Literal["GP", "MS"]
    sex: Literal["F", "M"]
    age: int = Field(ge=15, le=22)
    address: Literal["U", "R"]
    famsize: Literal["GT3", "LE3"]
    Pstatus: Literal["A", "T"]
    Medu: int = Field(ge=0, le=4)
    Fedu: int = Field(ge=0, le=4)
    Mjob: Literal["teacher", "health", "services", "at_home", "other"]
    Fjob: Literal["teacher", "health", "services", "at_home", "other"]
    reason: Literal["home", "reputation", "course", "other"]
    guardian: Literal["mother", "father", "other"]
    traveltime: int = Field(ge=1, le=4)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=3)
    schoolsup: Literal["yes", "no"]
    famsup: Literal["yes", "no"]
    paid: Literal["yes", "no"]
    activities: Literal["yes", "no"]
    nursery: Literal["yes", "no"]
    higher: Literal["yes", "no"]
    internet: Literal["yes", "no"]
    romantic: Literal["yes", "no"]
    famrel: int = Field(ge=1, le=5)
    freetime: int = Field(ge=1, le=5)
    goout: int = Field(ge=1, le=5)
    Dalc: int = Field(ge=1, le=5)
    Walc: int = Field(ge=1, le=5)
    health: int = Field(ge=1, le=5)
    absences: int = Field(ge=0, le=75)


MODEL = joblib.load(MODEL_PATH)


class StudentPerformanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Academic Performance Predictor")
        self.root.geometry("960x760")
        self.root.minsize(860, 700)

        self.model = self._load_model()
        self.schema = self._load_schema()

        self._build_ui()

    def _load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. Run training first."
            )
        return joblib.load(MODEL_PATH)

    def _load_schema(self):
        if not SCHEMA_PATH.exists():
            return self._default_schema()

        with SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
            return json.load(schema_file)

    def _default_schema(self):
        return {
            "features": [
                "school", "sex", "age", "address", "famsize", "Pstatus",
                "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
                "traveltime", "studytime", "failures", "schoolsup", "famsup",
                "paid", "activities", "nursery", "higher", "internet",
                "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
                "health", "absences"
            ],
            "validation": {
                "age": {"type": "range", "min": 15, "max": 22},
                "Medu": {"type": "range", "min": 0, "max": 4},
                "Fedu": {"type": "range", "min": 0, "max": 4},
                "traveltime": {"type": "range", "min": 1, "max": 4},
                "studytime": {"type": "range", "min": 1, "max": 4},
                "failures": {"type": "range", "min": 0, "max": 3},
                "famrel": {"type": "range", "min": 1, "max": 5},
                "freetime": {"type": "range", "min": 1, "max": 5},
                "goout": {"type": "range", "min": 1, "max": 5},
                "Dalc": {"type": "range", "min": 1, "max": 5},
                "Walc": {"type": "range", "min": 1, "max": 5},
                "health": {"type": "range", "min": 1, "max": 5},
                "absences": {"type": "range", "min": 0, "max": 75}
            }
        }

    def _build_ui(self):
        self.root.configure(bg="#f1f5f9")

        title = tk.Label(
            self.root,
            text="Student Academic Performance Predictor",
            font=("Segoe UI", 20, "bold"),
            bg="#f1f5f9",
            fg="#0f172a"
        )
        title.pack(pady=(18, 8))

        container = ttk.Frame(self.root, padding=20)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, highlightthickness=0, bg="#f1f5f9")
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        fields = self._create_fields(scroll_frame)
        self.fields = fields

        buttons = ttk.Frame(scroll_frame)
        buttons.grid(
            row=len(fields),
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(20, 0)
        )

        ttk.Button(buttons, text="Predict", command=self.predict).grid(
            row=0, column=0, padx=(0, 8), sticky="w"
        )
        ttk.Button(buttons, text="Reset", command=self.reset).grid(
            row=0, column=1, padx=(0, 8), sticky="w"
        )
        ttk.Button(buttons, text="Exit", command=self.root.destroy).grid(
            row=0, column=2, sticky="w"
        )

        self.result_card = self._create_card(
            scroll_frame,
            "Prediction",
            row=len(fields),
            color="#dbeafe"
        )
        self.result_var = tk.StringVar(value="")
        self.result_label = ttk.Label(
            self.result_card,
            textvariable=self.result_var,
            font=("Segoe UI", 24, "bold"),
            foreground="#1d4ed8",
            wraplength=700,
            justify="center"
        )
        self.result_label.pack(fill="x", pady=(0, 6))

        self.probability_card = self._create_card(
            scroll_frame,
            "Prediction Probabilities",
            row=len(fields) + 1,
            color="#dcfce7"
        )
        self.probability_text = tk.Text(
            self.probability_card,
            height=8,
            wrap="word",
            state="disabled",
            bg="#dcfce7",
            relief="flat",
            padx=12,
            pady=8,
            font=("Segoe UI", 11)
        )
        self.probability_text.pack(fill="both", expand=True)

        self.recommendation_card = self._create_card(
            scroll_frame,
            "Recommended Action",
            row=len(fields) + 2,
            color="#fef3c7"
        )
        self.recommendation_var = tk.StringVar(value="")
        self.recommendation_label = ttk.Label(
            self.recommendation_card,
            textvariable=self.recommendation_var,
            wraplength=700,
            justify="left",
            font=("Segoe UI", 11)
        )
        self.recommendation_label.pack(fill="both", expand=True, padx=12, pady=8)

    def _create_card(self, parent, title, row, color):
        card = tk.Frame(
            parent,
            bg=color,
            highlightbackground="#cbd5e1",
            highlightthickness=1,
            padx=18,
            pady=16
        )
        card.grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(20, 0),
            padx=2
        )

        tk.Label(
            card,
            text=title,
            bg=color,
            fg="#0f172a",
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w")

        separator = tk.Frame(card, height=1, bg="#94a3b8")
        separator.pack(fill="x", pady=(8, 12))

        return card

    def _create_fields(self, parent):
        categories = [
            ("school", "School", ["GP", "MS"]),
            ("sex", "Sex", ["F", "M"]),
            ("age", "Age", None, 15, 22),
            ("address", "Address", ["U", "R"]),
            ("famsize", "Family Size", ["GT3", "LE3"]),
            ("Pstatus", "Parent Status", ["A", "T"]),
        ]

        numeric_fields = [
            ("Medu", "Mother Education", 0, 4),
            ("Fedu", "Father Education", 0, 4),
            ("traveltime", "Travel Time", 1, 4),
            ("studytime", "Study Time", 1, 4),
            ("failures", "Past Failures", 0, 3),
            ("famrel", "Family Relationship", 1, 5),
            ("freetime", "Free Time", 1, 5),
            ("goout", "Going Out", 1, 5),
            ("Dalc", "Workday Alcohol", 1, 5),
            ("Walc", "Weekend Alcohol", 1, 5),
            ("health", "Health", 1, 5),
            ("absences", "Absences", 0, 75),
        ]

        choice_fields = [
            ("Mjob", "Mother Job", ["teacher", "health", "services", "at_home", "other"]),
            ("Fjob", "Father Job", ["teacher", "health", "services", "at_home", "other"]),
            ("reason", "Reason", ["home", "reputation", "course", "other"]),
            ("guardian", "Guardian", ["mother", "father", "other"]),
            ("schoolsup", "School Support", ["yes", "no"]),
            ("famsup", "Family Support", ["yes", "no"]),
            ("paid", "Paid Classes", ["yes", "no"]),
            ("activities", "Activities", ["yes", "no"]),
            ("nursery", "Nursery", ["yes", "no"]),
            ("higher", "Higher Education", ["yes", "no"]),
            ("internet", "Internet", ["yes", "no"]),
            ("romantic", "Romantic Relationship", ["yes", "no"]),
        ]

        fields = {}
        row = 0

        for field_name, label, values, *range_values in categories:
            ttk.Label(parent, text=f"{label}:", anchor="w").grid(
                row=row, column=0, padx=(0, 10), pady=5, sticky="w"
            )
            if values is None:
                value = tk.StringVar(value="")
                ttk.Entry(parent, textvariable=value, width=18).grid(
                    row=row, column=1, pady=5, sticky="ew"
                )
            else:
                value = tk.StringVar()
                combobox = ttk.Combobox(
                    parent,
                    textvariable=value,
                    values=values,
                    state="readonly",
                    width=16
                )
                combobox.grid(row=row, column=1, pady=5, sticky="ew")
            fields[field_name] = value
            row += 1

        for field_name, label, minimum, maximum in numeric_fields:
            ttk.Label(parent, text=f"{label}:", anchor="w").grid(
                row=row, column=0, padx=(0, 10), pady=5, sticky="w"
            )
            value = tk.StringVar(value="")
            ttk.Entry(parent, textvariable=value, width=18).grid(
                row=row, column=1, pady=5, sticky="ew"
            )
            fields[field_name] = value
            row += 1

        for field_name, label, values in choice_fields:
            ttk.Label(parent, text=f"{label}:", anchor="w").grid(
                row=row, column=0, padx=(0, 10), pady=5, sticky="w"
            )
            value = tk.StringVar()
            combobox = ttk.Combobox(
                parent,
                textvariable=value,
                values=values,
                state="readonly",
                width=16
            )
            combobox.grid(row=row, column=1, pady=5, sticky="ew")
            fields[field_name] = value
            row += 1

        ttk.Label(
            parent,
            text="Values are validated before prediction.",
            foreground="#475569",
            font=("Segoe UI", 9)
        ).grid(row=row, column=0, columnspan=2, pady=(10, 0), sticky="w")

        parent.grid_columnconfigure(1, weight=1)
        return fields

    def _validate_fields(self):
        values = {}
        for field_name, value_var in self.fields.items():
            value = value_var.get().strip()
            if value == "":
                raise ValueError(f"{field_name} is required.")
            values[field_name] = value

        for field_name, data in self.schema.get("validation", {}).items():
            if field_name not in values:
                continue
            try:
                number = int(values[field_name])
            except ValueError:
                continue
            if not data["min"] <= number <= data["max"]:
                raise ValueError(
                    f"{field_name} must be between {data['min']} and {data['max']}."
                )

        return values

    def predict(self):
        try:
            raw_values = self._validate_fields()
            model_input = {}
            for field_name, value in raw_values.items():
                normalized = value.strip().lower()
                if field_name in {"school", "sex", "address", "famsize", "Pstatus"}:
                    model_input[field_name] = normalized.upper()
                elif field_name in {"Mjob", "Fjob", "reason", "guardian"}:
                    model_input[field_name] = normalized
                elif field_name in {"schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"}:
                    model_input[field_name] = normalized
                else:
                    model_input[field_name] = int(value)

            student = pd.DataFrame([model_input])
            prediction = self.model.predict(student)[0]
            probabilities = self.model.predict_proba(student)[0]
            classes = self.model.classes_

            result_text = prediction
            formatted_probabilities = "\n".join(
                f"{class_name}: {probability:.2%}"
                for class_name, probability in zip(classes, probabilities)
            )

            recommendation = self._recommendation(prediction)
            self.result_var.set(result_text)
            self.probability_text.configure(state="normal")
            self.probability_text.delete("1.0", "end")
            self.probability_text.insert("end", formatted_probabilities)
            self.probability_text.configure(state="disabled")
            self.recommendation_var.set(recommendation)
        except ValueError as error:
            messagebox.showerror("Invalid Input", str(error))
        except Exception as error:
            messagebox.showerror("Prediction Error", f"Unable to predict: {error}")

    def _recommendation(self, result):
        recommendations = {
            "Low": "High-risk student: assign an academic mentor, monitor attendance, and provide additional support.",
            "Medium": "Medium-risk student: monitor progress regularly and provide targeted academic support.",
            "High": "High-performing student: encourage continued study and advanced learning opportunities."
        }
        return recommendations.get(result, "Performance prediction completed.")

    def reset(self):
        for value_var in self.fields.values():
            value_var.set("")
        self.result_var.set("")
        self.probability_text.configure(state="normal")
        self.probability_text.delete("1.0", "end")
        self.probability_text.configure(state="disabled")


def _recommendation(result: str) -> str:
    recommendations = {
        "Low": "High-risk student: assign an academic mentor, monitor attendance, and provide additional support.",
        "Medium": "Medium-risk student: monitor progress regularly and provide targeted academic support.",
        "High": "High-performing student: encourage continued study and advanced learning opportunities.",
    }
    return recommendations.get(result, "Performance prediction completed.")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Student Performance Predictor</title>
        <style>
            :root { color-scheme: light; font-family: Arial, sans-serif; }
            body { margin: 0; background: #f1f5f9; color: #0f172a; }
            main { max-width: 1100px; margin: 0 auto; padding: 32px 20px 56px; }
            h1 { text-align: center; }
            form { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
            .field { background: white; padding: 12px; border-radius: 10px; box-shadow: 0 1px 4px rgba(15, 23, 42, .08); }
            label { display: block; font-weight: 700; margin-bottom: 6px; }
            input, select { width: 100%; box-sizing: border-box; padding: 9px; border: 1px solid #cbd5e1; border-radius: 6px; }
            button { margin-top: 22px; padding: 12px 24px; border: 0; border-radius: 8px; background: #2563eb; color: white; font-weight: 700; cursor: pointer; }
            #result { display: none; margin-top: 28px; padding: 22px; border-radius: 12px; background: #dbeafe; }
            .probabilities { white-space: pre-line; }
        </style>
    </head>
    <body>
        <main>
            <h1>Student Academic Performance Predictor</h1>
            <form id="student-form">
                <div class="field"><label>School</label><select name="school"><option>GP</option><option>MS</option></select></div>
                <div class="field"><label>Sex</label><select name="sex"><option>F</option><option>M</option></select></div>
                <div class="field"><label>Age</label><input name="age" type="number" min="15" max="22" required></div>
                <div class="field"><label>Address</label><select name="address"><option>U</option><option>R</option></select></div>
                <div class="field"><label>Family Size</label><select name="famsize"><option>GT3</option><option>LE3</option></select></div>
                <div class="field"><label>Parent Status</label><select name="Pstatus"><option>A</option><option>T</option></select></div>
                <div class="field"><label>Mother Education</label><input name="Medu" type="number" min="0" max="4" required></div>
                <div class="field"><label>Father Education</label><input name="Fedu" type="number" min="0" max="4" required></div>
                <div class="field"><label>Mother Job</label><select name="Mjob"><option>teacher</option><option>health</option><option>services</option><option>at_home</option><option>other</option></select></div>
                <div class="field"><label>Father Job</label><select name="Fjob"><option>teacher</option><option>health</option><option>services</option><option>at_home</option><option>other</option></select></div>
                <div class="field"><label>Reason</label><select name="reason"><option>home</option><option>reputation</option><option>course</option><option>other</option></select></div>
                <div class="field"><label>Guardian</label><select name="guardian"><option>mother</option><option>father</option><option>other</option></select></div>
                <div class="field"><label>Travel Time</label><input name="traveltime" type="number" min="1" max="4" required></div>
                <div class="field"><label>Study Time</label><input name="studytime" type="number" min="1" max="4" required></div>
                <div class="field"><label>Past Failures</label><input name="failures" type="number" min="0" max="3" required></div>
                <div class="field"><label>School Support</label><select name="schoolsup"><option>yes</option><option>no</option></select></div>
                <div class="field"><label>Family Support</label><select name="famsup"><option>yes</option><option>no</option></select></div>
                <div class="field"><label>Paid Classes</label><select name="paid"><option>yes</option><option>no</option></select></div>
                <div class="field"><label>Activities</label><select name="activities"><option>yes</option><option>no</option></select></div>
                <div class="field"><label>Nursery</label><select name="nursery"><option>yes</option><option>no</option></select></div>
                <div class="field"><label>Higher Education</label><select name="higher"><option>yes</option><option>no</option></select></div>
                <div class="field"><label>Internet</label><select name="internet"><option>yes</option><option>no</option></select></div>
                <div class="field"><label>Romantic Relationship</label><select name="romantic"><option>yes</option><option>no</option></select></div>
                <div class="field"><label>Family Relationship</label><input name="famrel" type="number" min="1" max="5" required></div>
                <div class="field"><label>Free Time</label><input name="freetime" type="number" min="1" max="5" required></div>
                <div class="field"><label>Going Out</label><input name="goout" type="number" min="1" max="5" required></div>
                <div class="field"><label>Workday Alcohol</label><input name="Dalc" type="number" min="1" max="5" required></div>
                <div class="field"><label>Weekend Alcohol</label><input name="Walc" type="number" min="1" max="5" required></div>
                <div class="field"><label>Health</label><input name="health" type="number" min="1" max="5" required></div>
                <div class="field"><label>Absences</label><input name="absences" type="number" min="0" max="75" required></div>
                <div style="grid-column: 1 / -1"><button type="submit">Predict</button></div>
            </form>
            <section id="result" aria-live="polite">
                <h2 id="prediction"></h2>
                <div class="probabilities" id="probabilities"></div>
                <p id="recommendation"></p>
            </section>
        </main>
        <script>
            document.getElementById('student-form').addEventListener('submit', async (event) => {
                event.preventDefault();
                const form = new FormData(event.target);
                const data = Object.fromEntries(form.entries());
                Object.keys(data).forEach((key) => {
                    data[key] = Number.isNaN(Number(data[key])) ? data[key] : Number(data[key]);
                });

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                if (!response.ok) {
                    alert(result.detail || 'Prediction failed.');
                    return;
                }

                document.getElementById('prediction').textContent = 'Prediction: ' + result.prediction;
                document.getElementById('probabilities').textContent = result.probabilities
                    .map((item) => item.class + ': ' + (item.probability * 100).toFixed(2) + '%')
                    .join('\n');
                document.getElementById('recommendation').textContent = result.recommendation;
                document.getElementById('result').style.display = 'block';
            });
        </script>
    </body>
    </html>
    """


@app.post("/predict")
def predict(student: StudentInput):
    try:
        data = pd.DataFrame([student.model_dump()])
        prediction = MODEL.predict(data)[0]
        probabilities = MODEL.predict_proba(data)[0]
        classes = MODEL.classes_
        probability_items = [
            {"class": class_name, "probability": float(probability)}
            for class_name, probability in zip(classes, probabilities)
        ]
        return {
            "prediction": prediction,
            "probabilities": probability_items,
            "recommendation": _recommendation(prediction),
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error


def main():
    root = tk.Tk()
    app_ui = StudentPerformanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
