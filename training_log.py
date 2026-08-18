# ============================================================
# HEART DISEASE PREDICTION - MODEL TRAINING
# ============================================================

import logging
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


# ============================================================
# 1. LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log", mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. LOAD DATASET
# ============================================================

logger.info("=" * 60)
logger.info("HEART DISEASE MODEL TRAINING STARTED")
logger.info("=" * 60)

logger.info("Loading dataset...")

df = pd.read_csv("Heart_Disease_Prediction.csv")

logger.info("Dataset loaded successfully.")
logger.info(f"Dataset shape: {df.shape}")

logger.info(f"Columns: {df.columns.tolist()}")


# ============================================================
# 3. CHECK DATA
# ============================================================

logger.info("=" * 60)
logger.info("DATA CHECK")
logger.info("=" * 60)

logger.info(f"Missing values:\n{df.isnull().sum()}")

logger.info(f"Duplicate rows: {df.duplicated().sum()}")

logger.info(
    f"Target distribution:\n"
    f"{df['Heart Disease'].value_counts()}"
)


# ============================================================
# 4. CONVERT TARGET VARIABLE
# ============================================================

logger.info("=" * 60)
logger.info("ENCODING TARGET VARIABLE")
logger.info("=" * 60)

df["Heart Disease"] = df["Heart Disease"].map({
    "Absence": 0,
    "Presence": 1
})

logger.info(
    f"Encoded target distribution:\n"
    f"{df['Heart Disease'].value_counts()}"
)

logger.info(
    f"Missing target values: "
    f"{df['Heart Disease'].isna().sum()}"
)


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Heart Disease", axis=1)

y = df["Heart Disease"]

logger.info("=" * 60)
logger.info("FEATURES AND TARGET")
logger.info("=" * 60)

logger.info(f"Feature shape: {X.shape}")
logger.info(f"Target shape: {y.shape}")

logger.info(
    f"Features used: {X.columns.tolist()}"
)


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

logger.info("=" * 60)
logger.info("TRAIN-TEST SPLIT")
logger.info("=" * 60)

logger.info(f"X_train shape: {X_train.shape}")
logger.info(f"X_test shape: {X_test.shape}")
logger.info(f"y_train shape: {y_train.shape}")
logger.info(f"y_test shape: {y_test.shape}")


# ============================================================
# 7. CREATE LOGISTIC REGRESSION PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(max_iter=3000)
    )
])

logger.info("Logistic Regression pipeline created.")


# ============================================================
# 8. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

logger.info("=" * 60)
logger.info("CROSS VALIDATION")
logger.info("=" * 60)

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

logger.info(
    f"Cross-validation scores: {scores}"
)

logger.info(
    f"Mean CV Accuracy: {scores.mean():.4f}"
)


# ============================================================
# 9. HYPERPARAMETER TUNING
# ============================================================

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

logger.info("=" * 60)
logger.info("HYPERPARAMETER TUNING")
logger.info("=" * 60)

logger.info("Starting GridSearchCV...")

grid_search.fit(
    X_train,
    y_train
)

logger.info("GridSearchCV completed successfully.")


# ============================================================
# 10. BEST PARAMETERS
# ============================================================

logger.info("=" * 60)
logger.info("BEST MODEL PARAMETERS")
logger.info("=" * 60)

logger.info(
    f"Best Parameters: "
    f"{grid_search.best_params_}"
)

logger.info(
    f"Best CV ROC-AUC: "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# 11. GET FINAL MODEL
# ============================================================

best_model = grid_search.best_estimator_

logger.info("Best model selected successfully.")


# ============================================================
# 12. FINAL PREDICTION
# ============================================================

final_pred = best_model.predict(X_test)

final_prob = best_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

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


logger.info("=" * 60)
logger.info("FINAL MODEL EVALUATION")
logger.info("=" * 60)

logger.info(f"Accuracy : {accuracy:.4f}")
logger.info(f"Precision: {precision:.4f}")
logger.info(f"Recall   : {recall:.4f}")
logger.info(f"F1 Score : {f1:.4f}")
logger.info(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    final_pred
)

logger.info("=" * 60)
logger.info("CONFUSION MATRIX")
logger.info("=" * 60)

logger.info(f"\n{cm}")


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    final_pred
)

logger.info("=" * 60)
logger.info("CLASSIFICATION REPORT")
logger.info("=" * 60)

logger.info(f"\n{report}")


# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

model_filename = "heart_disease_model.pkl"

joblib.dump(
    best_model,
    model_filename
)

logger.info("=" * 60)
logger.info("MODEL SAVING")
logger.info("=" * 60)

logger.info(
    f"Model saved successfully as: "
    f"{model_filename}"
)


# ============================================================
# 17. VERIFY SAVED MODEL
# ============================================================

loaded_model = joblib.load(
    model_filename
)

logger.info(
    "Saved model loaded successfully."
)

test_prediction = loaded_model.predict(
    X_test
)

verification_accuracy = accuracy_score(
    y_test,
    test_prediction
)

logger.info(
    f"Saved model verification accuracy: "
    f"{verification_accuracy:.4f}"
)


# ============================================================
# 18. SAVE CLEAN DATASET
# ============================================================

clean_dataset_filename = (
    "Clean Heart Disease Prediction.csv"
)

df.to_csv(
    clean_dataset_filename,
    index=False
)

logger.info(
    f"Clean dataset saved as: "
    f"{clean_dataset_filename}"
)


# ============================================================
# 19. TRAINING COMPLETE
# ============================================================

logger.info("=" * 60)
logger.info("TRAINING COMPLETED SUCCESSFULLY")
logger.info("=" * 60)