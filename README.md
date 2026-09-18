
# AI/ML Project: Wine Class Classification

## 1. Project Overview

This project applies supervised machine learning to classify wine samples into one of three classes using their chemical properties.

The project demonstrates concepts from the Fundamentals of AI and ML syllabus, especially:

- Data representation
- Statistical features
- Supervised learning
- Classification
- Training and testing
- Model validation
- Hyperparameters
- Overfitting and underfitting
- Bias and variance
- Model evaluation

The dataset is the standard Wine dataset available through scikit-learn. It contains 178 samples, 13 numerical features, and 3 target classes.

## 2. Problem Statement

Given the chemical measurements of a wine sample, predict which of the three wine classes it belongs to.

## 3. Objectives

1. Load and inspect a labelled dataset.
2. Represent the observations using numerical features.
3. Split the data into training and testing sets.
4. Train multiple supervised classification algorithms.
5. Compare their test performance.
6. Use cross-validation to examine generalization.
7. Generate a confusion matrix and classification report.
8. Save the selected model for future predictions.

## 4. Algorithms Used

The project compares:

- Logistic Regression
- Decision Tree
- Random Forest

Random Forest is used as the final saved model.

## 5. Dataset

The project uses the built-in `load_wine()` dataset from scikit-learn.

Features include chemical measurements such as:

- Alcohol
- Malic acid
- Ash
- Alcalinity of ash
- Magnesium
- Total phenols
- Flavanoids
- Nonflavanoid phenols
- Proanthocyanins
- Color intensity
- Hue
- OD280/OD315 of diluted wines
- Proline

Target: wine class (0, 1, or 2).

No external dataset download is required.

## 6. Repository Structure

```text
AI_ML_Wine_Classification_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── run.sh
├── run.bat
├── src/
│   ├── train.py
│   ├── predict.py
│   └── evaluation.py
├── tests/
│   └── test_project.py
├── results/
├── models/
└── docs/
    ├── methodology.md
    └── PROJECT_REPORT.md
```

## 7. Environment Setup

Recommended:

- Python 3.10+
- pip
- Terminal/Command Prompt

Create a virtual environment:

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bat
python -m venv venv
venv\Scripts\activate
```

## 8. Install Dependencies

```bash
pip install -r requirements.txt
```

## 9. Train the Models

Run:

```bash
python src/train.py
```

The program will:

1. Load the Wine dataset.
2. Split it into training and testing data.
3. Standardize features where required.
4. Train Logistic Regression, Decision Tree and Random Forest.
5. Evaluate each model.
6. Perform 5-fold cross-validation.
7. Save the Random Forest model and scaler.
8. Save comparison results.

## 10. Evaluate the Saved Model

Run:

```bash
python src/evaluation.py
```

This creates a confusion matrix at:

```text
results/confusion_matrix.png
```

## 11. Make a Prediction

First train the model:

```bash
python src/train.py
```

Then run:

```bash
python src/predict.py
```

The program asks for the 13 feature values and returns the predicted wine class.

A sample input can be supplied by pressing Enter to use the built-in demonstration sample.

## 12. Run Tests

Install pytest if it is not already installed:

```bash
pip install pytest
```

Then:

```bash
pytest
```

## 13. One-Command Execution

### macOS/Linux

```bash
chmod +x run.sh
./run.sh
```

### Windows

```bat
run.bat
```

## 14. Academic Relevance

This project directly demonstrates CO3 and CO4 concepts from the course syllabus:

- Probability/statistical interpretation of data
- Data representation
- Feature-based learning
- Supervised learning
- Classification
- Hyperparameters
- Validation sets / cross-validation
- Bias and variance
- Overfitting and underfitting
- Model evaluation

## 15. Limitations

The Wine dataset is a standard educational dataset and is relatively small. Therefore, results obtained from it should not be interpreted as evidence of performance on every real-world wine classification problem.

## 16. Future Scope

Possible extensions include:

- Hyperparameter optimization
- Feature selection
- PCA-based dimensionality reduction
- K-means clustering for exploratory analysis
- Neural network classification
- Web-based prediction interface
- Real-time deployment as an API

## 17. Authors

Student Name: Bhavesh Dhidaria

Registration Number: 24BAC10021

Course: Fundamentals of AI and ML

University: VIT Bhopal University

Academic Year: 2026-27
