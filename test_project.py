import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train import load_dataset


def test_dataset_shape():
    X, y, target_names = load_dataset()
    assert X.shape == (178, 13)
    assert len(y) == 178
    assert len(target_names) == 3


def test_model_can_train():
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target,
        test_size=0.2,
        random_state=42,
        stratify=wine.target
    )

    model = RandomForestClassifier(
        n_estimators=50, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert accuracy_score(y_test, predictions) >= 0.70
