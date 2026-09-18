from pathlib import Path
import joblib
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"


def load_dataset():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")
    return X, y, wine.target_names


def main():
    print("=" * 60)
    print("AI/ML WINE CLASSIFICATION PROJECT")
    print("=" * 60)

    X, y, target_names = load_dataset()
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {list(target_names)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42
        )
    }

    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="accuracy"
        )

        rows.append({
            "model": name,
            "test_accuracy": accuracy,
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std": cv_scores.std()
        })

        print(f"\n{name}")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"5-fold CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(classification_report(
            y_test, predictions, target_names=target_names, zero_division=0
        ))

    comparison = pd.DataFrame(rows)
    RESULT_DIR.mkdir(exist_ok=True)
    comparison.to_csv(RESULT_DIR / "model_comparison.csv", index=False)

    # Train final Random Forest on the training split and save it.
    final_model = models["Random Forest"]
    final_model.fit(X_train, y_train)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "wine_classifier.joblib")

    # Save feature names and target names for prediction validation.
    metadata = {
        "features": list(X.columns),
        "target_names": list(target_names)
    }
    joblib.dump(metadata, MODEL_DIR / "metadata.joblib")

    print("\nModel comparison saved to results/model_comparison.csv")
    print("Final model saved to models/wine_classifier.joblib")
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
