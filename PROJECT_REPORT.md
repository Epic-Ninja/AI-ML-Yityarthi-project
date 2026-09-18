# PROJECT REPORT

## AI/ML Based Wine Class Classification

### 1. Introduction

Artificial Intelligence and Machine Learning provide computational methods for learning patterns from data. Classification is a supervised learning task in which a model learns from labelled examples and predicts the class of previously unseen observations.

This project implements a wine class classification system using chemical measurements as input features.

### 2. Problem Statement

To develop a machine learning system that can classify a wine sample into one of three classes based on its measured chemical properties.

### 3. Objectives

- Represent real-valued observations as machine learning features.
- Apply supervised learning algorithms.
- Compare different classification approaches.
- Validate the models using cross-validation.
- Evaluate generalization using an unseen test set.
- Generate a confusion matrix and classification report.
- Build a command-line prediction program.

### 4. Dataset

The project uses the Wine dataset distributed with scikit-learn.

Dataset characteristics:

- Samples: 178
- Input features: 13
- Classes: 3
- Feature type: numerical

The features describe chemical properties such as alcohol, malic acid, magnesium, phenols, flavanoids, colour intensity, hue and proline.

### 5. Proposed System

The system follows this pipeline:

Dataset
→ Data representation
→ Train/Test split
→ Model training
→ Cross-validation
→ Test evaluation
→ Model saving
→ New-sample prediction

### 6. Algorithms

Three supervised classifiers are compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Random Forest is saved as the final model.

### 7. Implementation

The implementation is divided into three main programs:

- `train.py`: loads the dataset, trains models, validates them and saves the final model.
- `evaluation.py`: evaluates the saved model and generates the confusion matrix.
- `predict.py`: accepts a new 13-feature observation and produces a predicted class.

### 8. Evaluation Metrics

Accuracy is calculated as the proportion of correct predictions.

Precision measures the proportion of predicted samples of a class that are actually members of that class.

Recall measures the proportion of actual samples of a class that are correctly detected.

F1-score combines precision and recall.

A confusion matrix provides class-wise counts of correct and incorrect predictions.

### 9. Results

Run:

```bash
python src/train.py
```

The program generates:

`results/model_comparison.csv`

Run:

```bash
python src/evaluation.py
```

The program generates:

`results/confusion_matrix.png`

The numerical results in the report should be copied from these generated files after the project is executed.

### 10. Relation to Course Outcomes

The project demonstrates topics from the syllabus:

- Probability and statistical interpretation of data
- Data representations
- Supervised learning
- Classification
- Hyperparameters
- Validation
- Bias and variance
- Overfitting and underfitting
- Machine learning applications

### 11. Limitations

The dataset is relatively small and is primarily intended for educational and benchmarking use. Model performance on this dataset does not establish performance on every real-world wine classification scenario.

### 12. Future Scope

Future work can include:

- PCA for dimensionality reduction
- K-means clustering for unsupervised exploration
- Grid-search hyperparameter optimization
- Neural network classification
- Explainable AI techniques
- Web/API deployment
- Larger real-world datasets

### 13. Conclusion

The project demonstrates an end-to-end supervised machine learning workflow for multiclass classification. It includes data representation, model training, validation, evaluation and prediction through a command-line interface.

### 14. References

1. Scikit-learn documentation, Wine dataset and classification algorithms.
2. Course material: Fundamentals of AI and ML.
3. Géron, A., Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow.
