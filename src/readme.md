🫀 Heart Disease Prediction

An end-to-end Machine Learning project that predicts the presence or
absence of heart disease from clinical patient measurements.

Educational disclaimer: This project is for learning and
demonstration purposes only. It is not a medical diagnostic system and
should not be used to make healthcare decisions.

📌 Project Overview

Machine Learning type: Supervised Learning

Problem type: Binary Classification

Target variable: Heart Disease

Classes: 0 = Absence, 1 = Presence

Train/test split: 80% / 20%

Random state: 42

Final model: Tuned Logistic Regression

Hyperparameter tuning: GridSearchCV

Cross-validation: 5-fold StratifiedKFold

Primary tuning metric: ROC-AUC

🎯 Objective

The objective is to build a classification model that predicts whether a
patient is likely to have heart disease using features such as age,
blood pressure, cholesterol, chest pain type, maximum heart rate,
exercise-induced angina, ST depression, and other clinical measurements.

The project follows a complete Machine Learning workflow:

Data collection

Dataset inspection

Exploratory Data Analysis (EDA)

Missing-value and duplicate checks

Target-distribution analysis

Outlier inspection

Data cleaning

Train/test splitting

Feature preprocessing

Model training

Model comparison

Cross-validation

Hyperparameter tuning

Final model evaluation

Feature-importance/coefficients analysis

Model saving

Prediction application

Streamlit deployment

📊 Dataset

The notebook uses the Heart Disease Prediction dataset obtained from
Kaggle and based on the UCI Heart Disease dataset.

The cleaned dataset contains 270 records and 14 columns:

13 input features

1 target column

Features

Feature                     Description

Age                       Patient age
Sex                       Sex encoded numerically
Chest pain type           Chest pain category
BP                        Blood pressure
Cholesterol               Cholesterol level
FBS over 120              Fasting blood sugar indicator
EKG results               EKG result category
Max HR                    Maximum heart rate
Exercise angina           Exercise-induced angina
ST depression             ST-segment depression
Slope of ST               ST slope category
Number of vessels fluro   Number of major vessels
Thallium                  Thallium test category
Heart Disease             Target variable

The target variable is already numerically encoded, so additional target
encoding is not required.

🔎 Exploratory Data Analysis

The notebook performs:

Dataset shape and structure inspection

Data-type inspection

Descriptive statistics

Missing-value analysis

Duplicate-value analysis

Target distribution

Age distribution

Cholesterol distribution

Maximum-heart-rate distribution

Age vs. heart disease

Cholesterol vs. heart disease

Maximum heart rate vs. heart disease

Chest pain vs. target

Correlation analysis

⚖️ Class Balance

The target distribution is reasonably balanced:

Absence: 150 records --- 55.56%

Presence: 120 records --- 44.44%

Because the difference between the classes is relatively small, the
notebook does not use SMOTE, oversampling, or undersampling.

🧹 Data Preprocessing

The notebook uses an 80/20 stratified train/test split:

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

For Logistic Regression and SVM, StandardScaler is used inside a
scikit-learn Pipeline.

The categorical values in the dataset are already represented by
numerical codes, so additional categorical encoding is not required.

🤖 Models Tested

Three classification models were evaluated.

1. Logistic Regression

Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

2. Support Vector Machine (SVM)

Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        probability=True,
        random_state=42
    ))
])

3. Random Forest

RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

📈 Model Comparison

The notebook produced the following test-set results:

Model            Accuracy    Precision       Recall     F1 Score      ROC-AUC

Logistic       0.8519       0.7857   0.9167   0.8462       0.8986
Regression

SVM                0.8148       0.7692       0.8333       0.8000       0.8861

Best initial model

Based on the notebook's initial test results, Logistic Regression
performed best overall:

Highest accuracy: 85.19%

Highest recall: 91.67%

Highest F1 score: 84.62%

Highest ROC-AUC: 89.86%

This is also a useful model for this project because Logistic Regression
is relatively simple and interpretable.

🔁 Cross-Validation

The notebook uses 5-fold stratified cross-validation:

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

For the initial Logistic Regression model:

Cross-validation scores:
0.8704
0.8889
0.7778
0.7778
0.8704

Mean CV Accuracy: 0.8370

⚙️ Hyperparameter Tuning

Hyperparameter tuning is used to find better settings for the Logistic
Regression model.

The notebook searches:

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear", "lbfgs"]
}

GridSearchCV evaluates the combinations using 5-fold cross-validation
and ROC-AUC.

Best parameters

The notebook found:

C = 0.01
solver = lbfgs

Best cross-validation ROC-AUC:

0.9158

On Windows, if GridSearchCV(n_jobs=-1) causes a
TerminatedWorkerError, use n_jobs=1 to avoid parallel
worker-process issues.

⭐ Final Model: Tuned Logistic Regression

The tuned Logistic Regression model achieved on the held-out test set:

Metric             Score

Accuracy      85.19%
Precision     83.33%
Recall        83.33%
F1 Score      83.33%
ROC-AUC       90.83%

Confusion matrix:

[[26, 4],
 [ 4, 20]]

The tuned model therefore becomes the final model selected by the
notebook.

📊 Model Evaluation

The project evaluates models using:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix

Classification Report

Cross-validation

The notebook also visualizes the Logistic Regression:

Confusion matrix

ROC curve

Feature coefficients

🔍 Feature Importance

For Logistic Regression, feature coefficients are examined to understand
which features contribute most strongly to the model.

The notebook calculates:

coefficients = best_model.named_steps["model"].coef_[0]

and ranks features using the absolute value of their coefficients.

This provides an interpretable view of feature influence, although
coefficients should not be interpreted as medical causation.

💾 Saving the Model

The final model can be saved with joblib:

import joblib

joblib.dump(
    best_model,
    "model_dir/heart_disease_model.joblib"
)

Load it later with:

model = joblib.load(
    "model_dir/heart_disease_model.joblib"
)

Make sure the model_dir folder exists before saving.

🖥️ Prediction

The project can use the trained model through a command-line prediction
script.

Run:

python prediction.py

The user can enter:

Age

Sex

Chest pain type

Blood pressure

Cholesterol

FBS

EKG results

Maximum heart rate

Exercise angina

ST depression

Slope of ST

Number of vessels

Thallium

The model then returns the predicted class and probability.

🌐 Streamlit Application

The project can also provide a browser-based interface using Streamlit.

Run:

python -m streamlit run app.py

The application allows the user to enter patient values through a
graphical interface and displays the model prediction.

📝 Logging

The project can record training/application events with Python logging:

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logging.info("Model Training started successfully")
logging.warning("Many values are missing")
logging.error("Failed to load dataset")
logging.critical(
    "Heart Disease prediction application cannot continue"
)

⚙️ Configuration

If using a .env file:

DATASET_NAME=C:\Users\sujee\Desktop\Heart_disease prediction1\Heart_Disease_Prediction.csv
TARGET_COLM=Heart Disease
TEST_SIZE=0.2
RANDOM_STATE=42
MODEL_PATH=model_dir/heart_disease_model.joblib

.env is a file, while .venv is the Python virtual-environment
folder.

Do not store passwords, API keys, or other secrets in a public
repository.

📦 Installation

Create a virtual environment:

python -m venv .venv

Activate it:

.\.venv\Scripts\activate

Install the required packages:

python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit python-dotenv

Or:

pip install -r requirements.txt

▶️ Running the Project

Train the model

python training.py

Run command-line prediction

python prediction.py

Run Streamlit

python -m streamlit run app.py

📁 Recommended Project Structure
## 📚 Jupyter Notebook

The Jupyter Notebook used for data analysis, visualization, model comparison,
cross-validation, hyperparameter tuning, and model evaluation is maintained
separately from the main application project folder.

Notebook:
`heart disease prediction.ipynb`


Heart_disease prediction1/
│
├── .env
├── .venv/
│
├── Heart_Disease_Prediction.csv
├── Clean Heart Disease Prediction.csv
├── heart disease prediction.ipynb
│
├── training.py
├── training_log.py
├── prediction.py
├── app.py
├── config.py
│
├── requirements.txt
├── app.log
│
└── model_dir/
    └── heart_disease_model.joblib

📚 Project Files

File                                   Purpose

heart disease prediction.ipynb       Main notebook containing EDA,
preprocessing, model training,
evaluation, and tuning

Heart_Disease_Prediction.csv         Original dataset used for training

Clean Heart Disease Prediction.csv   Cleaned/exported dataset

training.py                          Model training script

training_log.py                      Training script with logging

prediction.py                        Command-line prediction script

app.py                               Streamlit user interface

config.py                            Project configuration

.env                                 Environment/configuration variables

requirements.txt                     Python dependencies

app.log                              Application log file

heart_disease_model.joblib           Saved trained model

🧪 Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Joblib

Streamlit

Python Logging

python-dotenv

Jupyter Notebook

🎓 Viva Summary

Why Logistic Regression?

Logistic Regression achieved the strongest overall initial performance
among the three tested models and provides good interpretability for a
binary classification problem.

Why hyperparameter tuning?

Hyperparameter tuning searches for the model settings that provide
better generalization. In this project, GridSearchCV was used to
tune C and solver for Logistic Regression using 5-fold
cross-validation and ROC-AUC.

Why no SMOTE?

The target classes are reasonably balanced, with 55.56% absence and
44.44% presence, so oversampling was not necessary.

Why StandardScaler?

Standardization puts numerical features on comparable scales and is
especially useful for Logistic Regression and SVM.

⚠️ Disclaimer

This project is an educational Machine Learning project. The model's
output is not a medical diagnosis. Real clinical decisions require
appropriate medical examination, validated clinical tools, and qualified
healthcare professionals.

Project Summary

Heart Disease Prediction --- Supervised Binary Classification using
Logistic Regression, SVM, and Random Forest, with cross-validation,
GridSearchCV hyperparameter tuning, model evaluation, model persistence,
and Streamlit deployment.# Heart Disease Prediction

**Author:** SUMITRA NAYAK
