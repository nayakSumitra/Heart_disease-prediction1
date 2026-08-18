# ============================================================
# HEART DISEASE PREDICTION - MODEL TRAINING
# Based on the provided Jupyter Notebook
# ============================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# 2. LOAD DATASET


# Make sure Heart_Disease_Prediction.csv is in the
# same folder as this training.py file.

df = pd.read_csv("Heart_Disease_Prediction.csv")

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# 3. CHECK DATA

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTarget distribution:")
print(df["Heart Disease"].value_counts())

# 4. CONVERT TARGET VARIABLE

# Notebook:
# Absence = 0
# Presence = 1

df["Heart Disease"] = df["Heart Disease"].map({
    "Absence": 0,
    "Presence": 1
})


print("\nEncoded target distribution:")
print(df["Heart Disease"].value_counts())

print("\nMissing target values:")
print(df["Heart Disease"].isna().sum())

# 5. SEPARATE FEATURES AND TARGET

X = df.drop("Heart Disease", axis=1)

y = df["Heart Disease"]


print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeatures used by the model:")
print(X.columns.tolist())

# 6. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# 7. CREATE LOGISTIC REGRESSION PIPELINE

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "model",
        LogisticRegression(max_iter=3000)
    )
])

# 8. CROSS VALIDATION
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("\n" + "=" * 60)
print("CROSS VALIDATION")
print("=" * 60)

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("Cross Validation Scores:")
print(scores)

print("\nMean CV Accuracy:")
print(scores.mean())

# 9. HYPERPARAMETER TUNING
param_grid = {
    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "model__solver": [
        "liblinear",
        "lbfgs"
    ]
}


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=1
)

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

print("Training GridSearchCV...")

grid_search.fit(
    X_train,
    y_train
)

# 10. BEST PARAMETERS

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV ROC-AUC:")
print(grid_search.best_score_)

# 11. GET FINAL MODEL

best_model = grid_search.best_estimator_

# 12. FINAL PREDICTION
final_pred = best_model.predict(X_test)

final_prob = best_model.predict_proba(X_test)[:, 1]

# 13. MODEL EVALUATION

accuracy = accuracy_score(
    y_test,
    final_pred
)

precision = precision_score(
    y_test,
    final_pred
)

recall = recall_score(
    y_test,
    final_pred
)

f1 = f1_score(
    y_test,
    final_pred
)

roc_auc = roc_auc_score(
    y_test,
    final_prob
)


print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# 14. CONFUSION MATRIX

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    final_pred
)

print(cm)
# 15. CLASSIFICATION REPORT

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        final_pred
    )
)

# 16. SAVE FINAL MODEL
model_filename = "heart_disease_model.pkl"

joblib.dump(
    best_model,
    model_filename
)


print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    f"Model saved successfully as: {model_filename}"
)

# 17. VERIFY SAVED MODEL

loaded_model = joblib.load(
    model_filename
)

print("\nSaved model loaded successfully!")

# Verify that the loaded model gives predictions
test_prediction = loaded_model.predict(
    X_test
)

print(
    "Saved model verification accuracy:",
    accuracy_score(
        y_test,
        test_prediction)
)

# 18. SAVE CLEAN DATASET
df.to_csv(
    "Clean Heart Disease Prediction.csv",
    index=False
)

print("\nClean dataset saved as:")
print("Clean Heart Disease Prediction.csv")

print("\nTraining completed successfully!")
