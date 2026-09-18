# Methodology

## 1. Problem Formulation

The task is formulated as a multiclass supervised classification problem.

Input: 13 numerical chemical features of a wine sample.

Output: one of three wine classes.

## 2. Data Representation

The Wine dataset represents every observation as a 13-dimensional numerical feature vector.

This corresponds to the course concept of data representation for machine learning.

## 3. Data Preparation

The data is split using an 80:20 training/testing ratio. Stratification is used so that all classes are represented proportionally in both partitions.

## 4. Feature Scaling

Logistic Regression is trained inside a Pipeline containing StandardScaler. This prevents information from the test set from being used when fitting the scaler.

Decision Tree and Random Forest are not dependent on feature scaling, so they are trained directly on the original numerical features.

## 5. Supervised Learning

Each sample has a known target class. The algorithms learn a mapping from the input feature vector to the class label.

## 6. Models

### Logistic Regression

Provides a linear classification baseline.

### Decision Tree

Uses hierarchical feature-based decisions to classify observations.

### Random Forest

Combines multiple decision trees to improve generalization and reduce dependence on a single tree.

## 7. Validation

Five-fold cross-validation is performed on the training data. The mean and standard deviation of validation accuracy are reported.

## 8. Evaluation

The final test set is kept separate from model fitting. Accuracy, precision, recall, F1-score and a confusion matrix are used to evaluate classification performance.

## 9. Overfitting and Bias-Variance

Decision trees can become overly complex and overfit training data. Restricting tree depth is one way to control model complexity.

Random Forest reduces variance by averaging predictions across multiple decision trees.

The difference between cross-validation performance and test performance is considered when discussing generalization.

## 10. Reproducibility

A fixed random state of 42 is used for dataset splitting and model initialization where applicable. This makes the experiment reproducible.
