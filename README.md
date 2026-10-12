# Classification Model Evaluation

## Overview

A reproducible machine learning project comparing a baseline classifier with Logistic Regression and Random Forest on the Breast Cancer Wisconsin Diagnostic dataset.

The goal is to evaluate model performance on unseen data, investigate classification errors, and understand why accuracy alone is not enough to assess a model.

## Dataset

* **Source:** Breast Cancer Wisconsin Diagnostic dataset, provided by scikit-learn.
* **Samples:** 569
* **Features:** 30 numerical measurements
* **Task:** Binary classification — malignant vs. benign.
* **Split:** 80% training, 20% testing, using stratified sampling.

## Models

1. Dummy Classifier — baseline.
2. Logistic Regression — standardized numerical features.
3. Random Forest — nonlinear ensemble model.

## Evaluation

Models are compared using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices. Error analysis focuses on false positives and false negatives.

## Reproducibility

Install the dependencies and run:

```bash
pip install -r requirements.txt
python src/train.py
```

The script generates evaluation metrics and misclassified test samples in the `results/` directory.

## Results

Results will be added after running the experiment. No performance values are assumed in advance.

## Limitations

This is an educational experiment using a small, established dataset. Its results should not be interpreted as evidence of clinical usefulness or readiness for real-world medical diagnosis.

## Future Improvements

* Perform cross-validation on the training data.
* Tune model hyperparameters.
* Evaluate calibration and decision thresholds.
* Investigate feature importance and individual prediction errors.
