from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "wine_classifier.joblib"
RESULT_DIR = ROOT / "results"


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run: python src/train.py")
        return

    wine = load_wine()
    X = wine.data
    y = wine.target

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=wine.target_names,
        zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=wine.target_names
    )

    RESULT_DIR.mkdir(exist_ok=True)
    display.plot()
    plt.title("Wine Class Classification - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    print("\nSaved: results/confusion_matrix.png")


if __name__ == "__main__":
    main()
