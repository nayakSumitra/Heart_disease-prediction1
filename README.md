❤️ Heart Disease Prediction
📌 Project Overview

This project is a Machine Learning based Heart Disease Prediction System developed using Python.

The main objective of this project is to predict whether a patient has heart disease based on clinical information such as age, sex, chest pain type, blood pressure, cholesterol, maximum heart rate, and other medical attributes.

The project follows a complete Machine Learning workflow, including data understanding, exploratory data analysis, data cleaning, preprocessing, model building, evaluation, hyperparameter tuning, model saving, prediction, and Streamlit deployment.

Note: This project is created for educational purposes only and should not be used as a medical diagnosis system.

🎯 1. Problem Statement

Heart disease is one of the major health problems worldwide. Early prediction can help identify patients who may be at higher risk.

The objective of this project is to develop a Machine Learning classification model that predicts the presence or absence of heart disease using patient clinical data.

Business Objective

The system is designed to:

Predict the possibility of heart disease based on patient information.

Provide a simple and user-friendly prediction interface.

Display the predicted result and probability.

Demonstrate a complete Machine Learning workflow.

📂 2. Data Collection
Dataset Name

Heart Disease Prediction Dataset

Dataset File

Heart_Disease_Prediction.csv

Dataset Source

The dataset is based on the UCI Heart Disease dataset and is commonly available through public Machine Learning repositories such as Kaggle.

Dataset Type

This is a real-world medical dataset used for Machine Learning experimentation and educational purposes.

🔍 3. Data Understanding

The dataset was analyzed to understand its structure and contents.

The following aspects were checked:

Number of rows and columns

Column names

Data types

Target variable

Target distribution

Missing values

Duplicate records

Feature values

Dataset shape

Features

Feature

	

Description




Age

	

Age of the patient




Sex

	

Sex of the patient




Chest pain type

	

Type of chest pain




BP

	

Blood pressure




Cholesterol

	

Cholesterol level




FBS over 120

	

Fasting blood sugar indicator




EKG results

	

Electrocardiogram result




Max HR

	

Maximum heart rate




Exercise angina

	

Exercise-induced angina




ST depression

	

ST depression value




Slope of ST

	

Slope of ST




Number of vessels fluro

	

Number of major vessels




Thallium

	

Thallium test result

Target Variable

The target variable is Heart Disease.

The original target values were:

Absence

Presence

For Machine Learning, they were converted into:

Absence = 0

Presence = 1

📈 4. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and identify relationships between different features.

EDA Performed

Summary statistics

Missing value analysis

Duplicate analysis

Target distribution

Univariate analysis

Bivariate analysis

Correlation analysis

Outlier analysis

Visualizations

The analysis includes visualizations such as:

Distribution plots

Count plots

Box plots

Correlation heatmap

Feature relationship plots

Target distribution plots

These visualizations help understand the distribution of patient information and the relationship between features and heart disease.

🧹 5. Data Cleaning

The following data cleaning operations were performed:

Checked missing values.

Checked duplicate records.

Checked data types.

Converted the target variable into numerical values.

Checked for possible outliers.

Prepared the dataset for Machine Learning.

Created a cleaned version of the dataset.

The cleaned dataset is saved as:

Clean Heart Disease Prediction.csv

⚙️ 6. Data Preprocessing

The target variable was converted into binary values:

Absence → 0

Presence → 1

Feature Scaling

Feature scaling was performed using StandardScaler.

Standardization is useful for Logistic Regression because the features have different numerical ranges.

A Machine Learning Pipeline was used to combine feature scaling and the model.

✂️ 7. Train-Test Split

The dataset was divided into training and testing sets.

Split Configuration

Parameter

	

Value




Training Data

	

80%




Testing Data

	

20%




Random State

	

42




Stratification

	

Yes

Stratified splitting was used to maintain the distribution of the target variable in both training and testing datasets.

🤖 8. Model Building

Classification Machine Learning models were considered for the heart disease prediction problem.

The main final model used in the project is:

Logistic Regression

Logistic Regression is suitable for binary classification problems where the target has two possible outcomes.

In this project:

0 = Absence of Heart Disease

1 = Presence of Heart Disease

The model was combined with StandardScaler using a Machine Learning Pipeline.

📊 9. Model Evaluation

Since this is a classification problem, multiple evaluation metrics were used.

Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

ROC-AUC Score

Confusion Matrix

Classification Report

Accuracy

Measures the percentage of correctly predicted observations.

Precision

Measures how many of the predicted positive cases were actually positive.

Recall

Measures how many of the actual positive cases were correctly identified.

F1 Score

Provides a balance between precision and recall.

ROC-AUC

Measures the model's ability to distinguish between the two classes.

Confusion Matrix

Shows:

True Positive

True Negative

False Positive

False Negative

🔄 10. Cross Validation

Stratified K-Fold Cross Validation was used to evaluate model performance more reliably.

The project uses:

5 folds

Shuffling enabled

Random state = 42

Cross-validation helps determine whether the model performs consistently across different subsets of the dataset.

🎯 11. Hyperparameter Tuning

Hyperparameter tuning was performed using GridSearchCV.

The Logistic Regression hyperparameters were tuned to identify a better-performing configuration.

The parameters considered include:

Regularization parameter C

Solver

ROC-AUC was used as the scoring metric during hyperparameter tuning.

GridSearchCV helps select the best combination of hyperparameters based on cross-validation performance.

🏆 12. Final Model

After the model evaluation and hyperparameter tuning process, Logistic Regression was selected as the final prediction model.

The final Machine Learning pipeline contains:

StandardScaler + Logistic Regression

The trained model is saved for future predictions.

💾 13. Save the Model

The trained Machine Learning model is saved using Joblib.

Saved Model

heart_disease_model.joblib

The saved model can be loaded later without retraining the complete Machine Learning model.

This saved model is used by the prediction script and Streamlit application.

🔮 14. Prediction

The project includes a prediction system that accepts patient information such as:

Age

Sex

Chest pain type

Blood Pressure

Cholesterol

FBS over 120

EKG results

Maximum Heart Rate

Exercise Angina

ST Depression

Slope of ST

Number of vessels

Thallium

The system predicts one of two outcomes:

Absence

The model predicts that heart disease is not present.

Presence

The model predicts that heart disease is present.

The prediction probability is also displayed.

🌐 15. Streamlit Application

A Streamlit web application was created to provide an easy-to-use interface for the Machine Learning model.

The application allows users to:

Enter patient details.

Submit the information.

Get a heart disease prediction.

View the prediction probability.

Interact with the model through a web interface.

The Streamlit application is contained in:

app.py

🚀 16. Deployment

The Streamlit application can be deployed online using Streamlit Community Cloud.

The project is maintained in a GitHub repository and contains the required application files.

The deployment process includes:

Uploading the project to GitHub.

Adding the required dependencies in requirements.txt.

Connecting the GitHub repository to Streamlit Community Cloud.

Selecting app.py as the main application file.

Deploying the application.

📁 17. Project Structure

The main project folder contains the following files:

app.py – Streamlit application

training.py – Model training script

training_log.py – Model training with logging

prediction.py – Patient prediction script

requirements.txt – Required Python libraries

README.md – Project documentation

Heart_Disease_Prediction.csv – Original dataset

Clean Heart Disease Prediction.csv – Cleaned dataset

heart_disease_model.joblib – Trained Machine Learning model

app.log – Application and training log

.env – Project configuration file

The Jupyter Notebook is maintained separately from the main project folder.

📦 18. Technologies Used

The following technologies and Python libraries were used:

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Joblib

Streamlit

Python-dotenv

Git

GitHub

📝 19. Logging

Python logging was implemented to record important events during model training and application execution.

The logging system records different levels such as:

INFO

WARNING

ERROR

CRITICAL

The log information is stored in:

app.log

This helps in monitoring the application and identifying errors during execution.

🔐 20. Environment Configuration

A .env file is used for project configuration.

It can contain settings such as:

Dataset name

Target column

Test size

Random state

Model path

Sensitive information should not be committed to a public GitHub repository.

📋 21. Requirements

The project requires the following Python libraries:

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Joblib

Streamlit

Python-dotenv

All required packages are listed in:

requirements.txt

🔁 22. Project Workflow

Problem Statement

↓

Data Collection

↓

Data Understanding

↓

Exploratory Data Analysis

↓

Data Cleaning

↓

Data Preprocessing

↓

Train-Test Split

↓

Model Building

↓

Model Evaluation

↓

Cross Validation

↓

Hyperparameter Tuning

↓

Save Model

↓

Prediction

↓

Streamlit Application

↓

Deployment

📌 23. Conclusion

This project demonstrates a complete Machine Learning workflow for Heart Disease Prediction.

The project includes all major stages of a Machine Learning project:

Problem definition

Data collection

Data understanding

Exploratory Data Analysis

Data cleaning

Data preprocessing

Train-test splitting

Model building

Model evaluation

Cross-validation

Hyperparameter tuning

Model saving

Prediction

Streamlit application

Deployment

GitHub repository management

The project demonstrates how Machine Learning can be used to build a simple classification system for predicting the presence or absence of heart disease based on patient clinical information.

👩‍💻 Author

SUMITRA NAYAK

MCA Student | Aspiring Data Analyst

⚠️ Disclaimer

This project is created for educational and Machine Learning demonstration purposes only.

The prediction generated by this application should not be considered a medical diagnosis.

Medical decisions should always be made by a qualified healthcare professional.
