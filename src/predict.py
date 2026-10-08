import json
from pathlib import Path

import pandas as pd
import joblib


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.pkl"
SCHEMA_PATH = PROJECT_ROOT / "models" / "input_schema.json"

model = joblib.load(MODEL_PATH)

print("=" * 60)
print("STUDENT ACADEMIC PERFORMANCE PREDICTION")
print("=" * 60)


# ==========================================
# 2. GET USER INPUT
# ==========================================

FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
    "health", "absences"
]

INPUT_SCHEMA = {
    "school": {"type": "choice", "values": ["GP", "MS"]},
    "sex": {"type": "choice", "values": ["F", "M"]},
    "age": {"type": "range", "min": 15, "max": 22},
    "address": {"type": "choice", "values": ["R", "U"]},
    "famsize": {"type": "choice", "values": ["GT3", "LE3"]},
    "Pstatus": {"type": "choice", "values": ["A", "T"]},
    "Medu": {"type": "range", "min": 0, "max": 4},
    "Fedu": {"type": "range", "min": 0, "max": 4},
    "Mjob": {"type": "choice", "values": ["teacher", "health", "services", "at_home", "other"]},
    "Fjob": {"type": "choice", "values": ["teacher", "health", "services", "at_home", "other"]},
    "reason": {"type": "choice", "values": ["home", "reputation", "course", "other"]},
    "guardian": {"type": "choice", "values": ["mother", "father", "other"]},
    "traveltime": {"type": "range", "min": 1, "max": 4},
    "studytime": {"type": "range", "min": 1, "max": 4},
    "failures": {"type": "range", "min": 0, "max": 3},
    "schoolsup": {"type": "choice", "values": ["yes", "no"]},
    "famsup": {"type": "choice", "values": ["yes", "no"]},
    "paid": {"type": "choice", "values": ["yes", "no"]},
    "activities": {"type": "choice", "values": ["yes", "no"]},
    "nursery": {"type": "choice", "values": ["yes", "no"]},
    "higher": {"type": "choice", "values": ["yes", "no"]},
    "internet": {"type": "choice", "values": ["yes", "no"]},
    "romantic": {"type": "choice", "values": ["yes", "no"]},
    "famrel": {"type": "range", "min": 1, "max": 5},
    "freetime": {"type": "range", "min": 1, "max": 5},
    "goout": {"type": "range", "min": 1, "max": 5},
    "Dalc": {"type": "range", "min": 1, "max": 5},
    "Walc": {"type": "range", "min": 1, "max": 5},
    "health": {"type": "range", "min": 1, "max": 5},
    "absences": {"type": "range", "min": 0, "max": 75}
}


with SCHEMA_PATH.open("w", encoding="utf-8") as schema_file:
    json.dump({"features": FEATURES, "validation": INPUT_SCHEMA}, schema_file, indent=2)
    schema_file.write("\n")


def normalize_choice(value, field_name):
    normalized_value = value.strip()
    if normalized_value == "":
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized_value.lower()


def get_validated_input(prompt, field_name):
    while True:
        try:
            value = input(prompt)
            if field_name in INPUT_SCHEMA:
                schema = INPUT_SCHEMA[field_name]
                if schema["type"] == "choice":
                    normalized_value = normalize_choice(value, field_name)
                    allowed_values = [item.lower() for item in schema["values"]]
                    if normalized_value not in allowed_values:
                        raise ValueError(
                            f"Invalid {field_name}. Allowed values: {', '.join(schema['values'])}."
                        )
                    return normalized_value
                if schema["type"] == "range":
                    try:
                        number = int(value)
                    except ValueError:
                        raise ValueError(
                            f"Invalid {field_name}. Enter a whole number."
                        ) from None
                    if not schema["min"] <= number <= schema["max"]:
                        raise ValueError(
                            f"Invalid {field_name}. Expected {schema['min']} to {schema['max']}."
                        )
                    return number
            raise ValueError(f"No validation rule exists for {field_name}.")
        except ValueError as error:
            print(f"Error: {error}")


print("\nEnter Student Details\n")

school = get_validated_input("School (GP/MS): ", "school").upper()
sex = get_validated_input("Sex (M/F): ", "sex").upper()
age = get_validated_input("Age: ", "age")
address = get_validated_input("Address (U/R): ", "address").upper()
famsize = get_validated_input("Family Size (GT3/LE3): ", "famsize").upper()
Pstatus = get_validated_input("Parent Status (T/A): ", "Pstatus").upper()
Medu = get_validated_input("Mother Education (0-4): ", "Medu")
Fedu = get_validated_input("Father Education (0-4): ", "Fedu")
Mjob = get_validated_input("Mother Job (teacher/health/services/at_home/other): ", "Mjob")
Fjob = get_validated_input("Father Job (teacher/health/services/at_home/other): ", "Fjob")
reason = get_validated_input("Reason (home/reputation/course/other): ", "reason")
guardian = get_validated_input("Guardian (mother/father/other): ", "guardian")
traveltime = get_validated_input("Travel Time (1-4): ", "traveltime")
studytime = get_validated_input("Study Time (1-4): ", "studytime")
failures = get_validated_input("Number of Past Failures (0-4): ", "failures")
schoolsup = get_validated_input("Extra School Support (yes/no): ", "schoolsup")
famsup = get_validated_input("Family Educational Support (yes/no): ", "famsup")
paid = get_validated_input("Extra Paid Classes (yes/no): ", "paid")
activities = get_validated_input("Extra Activities (yes/no): ", "activities")
nursery = get_validated_input("Attended Nursery (yes/no): ", "nursery")
higher = get_validated_input("Wants Higher Education (yes/no): ", "higher")
internet = get_validated_input("Internet Access (yes/no): ", "internet")
romantic = get_validated_input("Romantic Relationship (yes/no): ", "romantic")
famrel = get_validated_input("Family Relationship Quality (1-5): ", "famrel")
freetime = get_validated_input("Free Time (1-5): ", "freetime")
goout = get_validated_input("Going Out Frequency (1-5): ", "goout")
Dalc = get_validated_input("Workday Alcohol Consumption (1-5): ", "Dalc")
Walc = get_validated_input("Weekend Alcohol Consumption (1-5): ", "Walc")
health = get_validated_input("Current Health Status (1-5): ", "health")
absences = get_validated_input("Number of Absences: ", "absences")


# ==========================================
# 3. CREATE DATAFRAME
# ==========================================

student = pd.DataFrame({
    "school": [school],
    "sex": [sex],
    "age": [age],
    "address": [address],
    "famsize": [famsize],
    "Pstatus": [Pstatus],

    "Medu": [Medu],
    "Fedu": [Fedu],

    "Mjob": [Mjob],
    "Fjob": [Fjob],

    "reason": [reason],
    "guardian": [guardian],

    "traveltime": [traveltime],
    "studytime": [studytime],
    "failures": [failures],

    "schoolsup": [schoolsup],
    "famsup": [famsup],
    "paid": [paid],
    "activities": [activities],

    "nursery": [nursery],
    "higher": [higher],
    "internet": [internet],
    "romantic": [romantic],

    "famrel": [famrel],
    "freetime": [freetime],
    "goout": [goout],

    "Dalc": [Dalc],
    "Walc": [Walc],

    "health": [health],
    "absences": [absences]
})


# ==========================================
# 4. PREDICT
# ==========================================

prediction = model.predict(student)

result = prediction[0]


# ==========================================
# 5. DISPLAY RESULT
# ==========================================

print("\n")
print("=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(f"\nStudent Performance: {result}")


# ==========================================
# 6. PROBABILITY
# ==========================================

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(student)

    classes = model.classes_

    print("\nPrediction Probability:")

    for class_name, probability in zip(
        classes,
        probabilities[0]
    ):
        print(
            f"{class_name}: {probability:.2%}"
        )


# ==========================================
# 7. RECOMMENDATION
# ==========================================

print("\n")
print("=" * 60)
print("RECOMMENDED ACTION")
print("=" * 60)


if result == "Low":

    print("""
HIGH RISK STUDENT

Recommended:
1. Assign academic mentor
2. Monitor attendance
3. Provide additional classes
4. Create personalized study plan
5. Contact parent/guardian
6. Monitor performance weekly
""")


elif result == "Medium":

    print("""
MEDIUM RISK STUDENT

Recommended:
1. Monitor academic progress
2. Encourage regular study
3. Provide additional resources
4. Monitor attendance
5. Review performance monthly
""")


else:

    print("""
HIGH PERFORMING STUDENT

Recommended:
1. Continue monitoring
2. Provide advanced learning resources
3. Encourage academic competitions
4. Provide leadership opportunities
""")


print("\nPrediction completed successfully.")

