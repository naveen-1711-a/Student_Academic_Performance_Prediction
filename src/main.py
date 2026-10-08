import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# ==========================================
# 1. LOAD DATA
# ==========================================

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    DATA_PATH,
    sep=";"
)

print("=" * 60)
print("STUDENT ACADEMIC PERFORMANCE PREDICTION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 2. DATA QUALITY CHECK (Noise, Missing Values, Outliers, Duplicates)
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nOutlier Detection (IQR Method on Numerical Features):")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
    if outliers > 0:
        print(f"  - {col}: {outliers} outliers identified")

# ==========================================
# 3. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# ==========================================
# 4. CREATE TARGET
# ==========================================

def performance_category(grade):

    if grade < 10:
        return "Low"

    elif grade < 15:
        return "Medium"

    else:
        return "High"


df["Performance"] = df["G3"].apply(
    performance_category
)


print("\nPerformance Distribution:")
print(df["Performance"].value_counts())


# ==========================================
# 5. TARGET VISUALIZATION
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Performance"
)

plt.title("Student Performance Distribution")
plt.xlabel("Performance")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig(
    OUTPUTS_DIR / "performance_distribution.png"
)

# plt.show() # Commented out to run headlessly


# ==========================================
# 6. PREPARE FEATURES
# ==========================================

# G1 and G2 are excluded to avoid
# target leakage for early prediction.

X = df.drop(
    columns=[
        "G1",
        "G2",
        "G3",
        "Performance"
    ]
)

y = df["Performance"]


print("\nFeature Shape:")
print(X.shape)


# ==========================================
# 7. IDENTIFY FEATURE TYPES
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ==========================================
# 8. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==========================================
# 9. PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==========================================
# 10. MODELS
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            class_weight="balanced_subsample"
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

    "SVM":
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
}


# ==========================================
# 11. TRAIN AND EVALUATE (WITH PCA)
# ==========================================

results = []

trained_models = {}

predictions = {}


for name, classifier in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "pca",
                PCA(n_components=0.95, random_state=42) # Keep 95% variance
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    trained_models[name] = pipeline
    predictions[name] = y_pred


# ==========================================
# 12. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

results_df = results_df.sort_values(
    "F1 Score",
    ascending=False
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.bar(
    results_df["Model"],
    results_df["F1 Score"],
    color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
)
plt.title("Model F1 Score Comparison")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "model_f1_comparison.png")
# plt.show() # Commented out to run headlessly


# ==========================================
# 13. BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

best_predictions = predictions[
    best_model_name
]


print("\nBest Model:")
print(best_model_name)

MODEL_PATH = MODELS_DIR / "student_performance_model.pkl"
joblib.dump(best_model, MODEL_PATH)
print(f"Saved best model: {MODEL_PATH}")


# ==========================================
# 14. CONFUSION MATRIX
# ==========================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    OUTPUTS_DIR / "confusion_matrix.png"
)

# plt.show() # Commented out to run headlessly


# ==========================================
# 15. SAVE RESULTS
# ==========================================

results_df.to_csv(
    OUTPUTS_DIR / "model_results.csv",
    index=False
)

results_df.to_json(
    OUTPUTS_DIR / "model_results.json",
    orient="records",
    indent=2
)

print("\nResults saved successfully.")
print(f"CSV: {OUTPUTS_DIR / 'model_results.csv'}")
print(f"JSON: {OUTPUTS_DIR / 'model_results.json'}")