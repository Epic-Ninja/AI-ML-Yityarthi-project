from pathlib import Path
import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "wine_classifier.joblib"
META_PATH = ROOT / "models" / "metadata.joblib"


DEMO_SAMPLE = [
    13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76,
    0.26, 1.28, 4.38, 1.05, 3.40, 1050.0
]


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run: python src/train.py")
        return

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(META_PATH)

    print("=" * 60)
    print("WINE CLASS PREDICTION")
    print("=" * 60)
    print("\nEnter 13 feature values in this order:")
    print(", ".join(metadata["features"]))

    raw = input(
        "\nPress Enter to use the demonstration sample, "
        "or enter 13 values separated by spaces: "
    ).strip()

    if raw:
        values = [float(x) for x in raw.split()]
    else:
        values = DEMO_SAMPLE

    if len(values) != 13:
        raise ValueError("Exactly 13 numerical values are required.")

    sample = np.array(values, dtype=float).reshape(1, -1)
    predicted_class = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]

    print(f"\nPredicted class: {predicted_class}")
    print(f"Class label: {metadata['target_names'][predicted_class]}")
    print("\nClass probabilities:")
    for label, probability in zip(metadata["target_names"], probabilities):
        print(f"  {label}: {probability:.4f}")


if __name__ == "__main__":
    main()
