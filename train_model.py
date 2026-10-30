from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "malaria_dataset.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Fever", "Headache", "Chills", "Sweating",
    "Vomiting", "Nausea", "Weakness", "Muscle_Pain"
]
TARGET = "Malaria_Status"

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. Add an appropriately sourced "
            "CSV with the required columns. See README.txt."
        )

    data = pd.read_csv(DATA_FILE)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    data = data[required].copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    if not data[TARGET].isin([0, 1]).all():
        raise ValueError("Malaria_Status must contain only 0 and 1.")
    data[TARGET] = data[TARGET].astype(int)

    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        invalid = data[col].dropna().isin([0, 1])
        if not invalid.all():
            raise ValueError(f"{col} must contain only 0, 1, or missing values.")

    data = data.drop_duplicates()
    if data[TARGET].nunique() != 2:
        raise ValueError("Dataset must contain both target classes.")
    if data[TARGET].value_counts().min() < 2:
        raise ValueError("Each class needs at least two records for a stratified split.")

    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("model", RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight="balanced"
            ))
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            ("model", SVC(class_weight="balanced"))
        ]),
        "K-Nearest Neighbours": Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Sensitivity": recall_score(y_test, pred, zero_division=0),
            "Specificity": tn / (tn + fp) if tn + fp else 0,
            "F1_Score": f1_score(y_test, pred, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        })
        fitted[name] = model
        print(f"\n--- {name} ---")
        print(classification_report(y_test, pred, zero_division=0))
        print("Confusion matrix (rows actual [0,1], columns predicted [0,1]):")
        print(confusion_matrix(y_test, pred, labels=[0, 1]))

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)
    # Select by F1 as a simple baseline rule; consider sensitivity and clinical context too.
    selected_name = results.sort_values("F1_Score", ascending=False).iloc[0]["Model"]
    selected_model = fitted[selected_name]
    bundle = {
        "model": selected_model,
        "features": FEATURES,
        "target": TARGET,
        "model_name": selected_name,
        "note": "Model performance is dataset-specific; not clinically validated."
    }
    joblib.dump(bundle, MODEL_DIR / "malaria_risk_model.joblib")
    print("\nDataset rows after cleaning:", len(data))
    print("Training rows:", len(X_train), "| Testing rows:", len(X_test))
    print("Selected baseline model by test-set F1:", selected_name)
    print("Results saved to outputs/model_comparison_results.csv")
    print("Model saved to models/malaria_risk_model.joblib")
    print("\nIMPORTANT: Do not treat these metrics as clinical evidence unless the dataset")
    print("is appropriately sourced and independently validated.")

if __name__ == "__main__":
    main()
