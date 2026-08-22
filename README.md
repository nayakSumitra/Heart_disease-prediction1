

## live demo
streamlit app: https://heartdisease-prediction-by-sumitranayak.streamlit.app/

<div align="center">

# ❤️ Heart Disease Prediction

### Machine Learning Based Healthcare Prediction System

<p>
  <b>Python</b> •
  <b>Scikit-learn</b> •
  <b>Streamlit</b> •
  <b>Machine Learning</b>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">
</p>

</div>

---

<div align="center">

## 🩺 Predict • Analyze • Understand

</div>

This project is an **end-to-end Machine Learning application for Heart Disease Prediction**.

The system analyzes patient clinical information and predicts whether the patient is likely to have **heart disease or not**.

The project covers the complete Machine Learning workflow:

**Data Collection → EDA → Data Cleaning → Preprocessing → Model Training → Evaluation → Hyperparameter Tuning → Prediction → Streamlit → Deployment**

> ⚠️ **Disclaimer:** This project is developed for educational and academic purposes only. It is not a medical diagnosis system and should not be used as a substitute for professional medical advice.

---

# 🎯 Project Objective

The main objective of this project is to build a Machine Learning classification model that predicts:

<table>
<tr>
<td align="center">

### 🟢 ABSENCE

No heart disease predicted

</td>

<td align="center">

### 🔴 PRESENCE

Heart disease predicted

</td>
</tr>
</table>

### 💼 Business Objective

The application allows users to enter patient information and receive a Machine Learning based prediction along with the prediction probability.

---

# 📊 Dataset

<div align="center">

| 📌 Information | Details |
|---|---|
| Dataset | Heart Disease Prediction Dataset |
| File | `Heart_Disease_Prediction.csv` |
| Type | Real-world dataset |
| Problem Type | Binary Classification |
| Target | `Heart Disease` |
| Source |Kaggle.com(UCI heart disease dataset) |

</div>

The dataset contains clinical information about patients and their heart disease status.

---

# 🧬 Features

| Feature | Description |
|---|---|
| `Age` | Age of the patient |
| `Sex` | Sex of the patient |
| `Chest pain type` | Type of chest pain |
| `BP` | Blood pressure |
| `Cholesterol` | Cholesterol level |
| `FBS over 120` | Fasting blood sugar indicator |
| `EKG results` | Electrocardiogram result |
| `Max HR` | Maximum heart rate |
| `Exercise angina` | Exercise-induced angina |
| `ST depression` | ST depression |
| `Slope of ST` | Slope of ST |
| `Number of vessels fluro` | Number of major vessels |
| `Thallium` | Thallium test result |

### 🎯 Target Encoding

| Original Value | Machine Learning Value |
|---|---:|
| Absence | `0` |
| Presence | `1` |

---

# 🔎 Exploratory Data Analysis

The dataset was explored before model development.

### 📌 EDA Performed

- Dataset shape analysis
- Column analysis
- Data type analysis
- Summary statistics
- Missing value analysis
- Duplicate analysis
- Target distribution
- Univariate analysis
- Bivariate analysis
- Correlation analysis
- Outlier analysis

### 📈 Visualizations

The project includes:

- Distribution plots
- Count plots
- Box plots
- Correlation heatmap
- Feature relationship plots
- Target distribution visualizations

---

# 🧹 Data Cleaning

The following data cleaning steps were performed:

<div align="center">

| ✔️ Task | Status |
|---|---|
| Missing value checking | ✅ Completed |
| Duplicate checking | ✅ Completed |
| Data type checking | ✅ Completed |
| Target encoding | ✅ Completed |
| Outlier checking | ✅ Completed |
| Clean dataset creation | ✅ Completed |

</div>

The cleaned dataset is saved as:

**`Clean Heart Disease Prediction.csv`**

---

# ⚙️ Data Preprocessing

### Target Encoding

The target variable was converted from text values into numerical values:

**Absence → 0**

**Presence → 1**

### Feature Scaling

`StandardScaler` was used to standardize the input features.

The scaler and Machine Learning model were combined using a **Scikit-learn Pipeline**.

This ensures that the same preprocessing is applied during both training and prediction.

---

# ✂️ Train-Test Split

The dataset was divided into training and testing sets.

| Parameter | Value |
|---|---:|
| Training Data | 80% |
| Testing Data | 20% |
| Random State | 42 |
| Stratification | Yes |

Stratification was used to maintain the target class distribution.

---

# 🤖 Machine Learning Model

## Logistic Regression

The primary Machine Learning model used in this project is:
🔮 Prediction

The prediction script loads the saved Machine Learning model and accepts patient information.
### `Logistic Regression`

Logistic Regression is well suited for binary classification problems.

The model predicts:


0 → Absence of Heart Disease
1 → Presence of Heart Disease.
## Machine Learning Pipeline

Patient Data  
↓  
Data Preprocessing  
↓  
Feature Scaling  
↓  
Logistic Regression  
↓  
Prediction  
↓  
Heart Disease Result
# 🔄 Cross Validation

Cross validation is used to evaluate the consistency and reliability of the Machine Learning model.

A stratified cross-validation approach is used for classification so that the class distribution is maintained across different folds.

## Benefits

- Reduces dependency on a single data split.
- Provides a more reliable performance estimate.
- Helps identify overfitting.
- Supports hyperparameter selection.
# 🎯 Hyperparameter Tuning

Hyperparameter tuning is performed using **GridSearchCV**.

The purpose of hyperparameter tuning is to find the best configuration of the Machine Learning model.

## GridSearchCV Workflow

Multiple Parameter Combinations  
↓  
GridSearchCV  
↓  
Cross Validation  
↓  
Model Performance  
↓  
Best Parameters  
↓  
Best Model

The best-performing parameter combination is selected based on the chosen evaluation metric.
# 📊 Model Evaluation

Since this is a classification problem, several classification metrics are used to evaluate the trained model.

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Correctness of positive predictions |
| Recall | Ability to identify positive cases |
| F1 Score | Balance between Precision and Recall |
| ROC-AUC | Ability to distinguish between classes |
| Confusion Matrix | Detailed classification results |

# 💾 Save the Model

After training and evaluation, the trained Machine Learning model is saved using **Joblib**.

## Model File

`heart_disease_model.pkl`

Saving the model allows it to be loaded later for predictions without retraining the model.

## Model Workflow

Training  
↓  
Best Model  
↓  
Joblib  
↓  
`heart_disease_model.pkl`  
↓  
Load Model  
↓  
Prediction
# 🔮 Prediction

The prediction script loads the saved Machine Learning model and accepts patient information.

## Patient Input Features

| Feature | Description |
|---|---|
| Age | Patient age |
| Sex | Gender |
| Chest Pain Type | Type of chest pain |
| BP | Blood pressure |
| Cholesterol | Cholesterol level |
| FBS over 120 | Fasting blood sugar indicator |
| EKG Results | Electrocardiogram result |
| Max HR | Maximum heart rate |
| Exercise Angina | Exercise-induced angina |
| ST Depression | ST depression |
| Slope of ST | ST slope |
| Number of Vessels | Number of vessels |
| Thallium | Thallium test result |

## Prediction Output

- ❤️ **PRESENCE**
- 💚 **ABSENCE**

The application also displays the prediction probability.
# 🌐 Streamlit Application

A Streamlit web application is developed to make the Machine Learning model easy to use.

Users can enter patient information through a web interface and receive a heart disease prediction.

## Streamlit Features

- ❤️ Healthcare-themed interface
- 👤 Patient information form
- 🫀 Heart health parameters
- 🔍 Prediction button
- 📊 Prediction result
- 📈 Prediction probability
- ℹ️ Model information
- ⚠️ Medical disclaimer

## Application Workflow

User  
↓  
Enter Patient Information  
↓  
Streamlit Application  
↓  
Preprocessing  
↓  
Saved ML Model  
↓  
Prediction  
↓  
Display Result
# 🚀 Deployment

The Streamlit application can be deployed online using **Streamlit Community Cloud**.

## Deployment Workflow

GitHub Repository  
↓  
Streamlit Community Cloud  
↓  
Select Repository  
↓  
Select Branch  
↓  
Select `app.py`  
↓  
Install `requirements.txt`  
↓  
Deploy  
↓  
Live Web Application
# 📁 Project Structure

```text
Heart Disease Prediction/
│
├── Heart_Disease_Prediction.csv
├── Clean_Heart_Disease_Prediction.csv
│
├── Heart_Disease_Prediction.ipynb
│
├── training.py
├── training_log.py
├── prediction.py
├── app.py
│
├── heart_disease_model.pkl
│
├── requirements.txt
├── .env
├── app.log
│
└── README.md
### 19. Technologies Used

```markdown
# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| Scikit-learn | Machine Learning |
| Matplotlib | Data Visualization |
| Seaborn | Data Visualization |
| Joblib | Model Saving |
| Streamlit | Web Application |
| Python-dotenv | Environment Configuration |
| Git | Version Control |
| GitHub | Repository Management |
# ⚙️ Environment Configuration

Project configuration can be maintained using a `.env` file.

## Configuration

```text
DATASET_NAME=Heart_Disease_Prediction.csv
TARGET_COLUMN=Heart Disease
TEST_SIZE=0.2
RANDOM_STATE=42
MODEL_PATH=heart_disease_model.pkl
### 23. Complete Workflow

```markdown
# 🔄 Complete Machine Learning Workflow

📥 Data Collection  
↓  
🔎 Data Understanding  
↓  
📊 Exploratory Data Analysis  
↓  
🧹 Data Cleaning  
↓  
⚙️ Data Preprocessing  
↓  
📏 Feature Scaling  
↓  
✂️ Train-Test Split  
↓  
🤖 Model Building  
↓  
🔄 Cross Validation  
↓  
🎯 Hyperparameter Tuning  
↓  
📈 Model Evaluation  
↓  
💾 Save Model  
↓  
🔮 Prediction  
↓  
🌐 Streamlit Application  
↓  
🚀 Deployment
# 🎓 Learning Outcomes

This project demonstrates practical knowledge of:

- Python programming
- Data analysis
- Exploratory Data Analysis
- Data cleaning
- Data preprocessing
- Feature scaling
- Classification Machine Learning
- Logistic Regression
- Cross validation
- Hyperparameter tuning
- Model evaluation
- Model serialization
- Streamlit development
- Git and GitHub
- Deployment
# 👩‍💻 Author

## SUMITRA NAYAK

**MCA Student | Aspiring Data Analyst**

Interested in Data Analysis, Machine Learning, Python, and building practical data-driven applications.

### ❤️ Heart Disease Prediction

**Built with Python • Machine Learning • Scikit-learn • Streamlit**
# ⚠️ Disclaimer

This project is developed strictly for **educational and academic purposes**.

The prediction generated by this application should not be considered a medical diagnosis and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

For actual medical concerns, always consult a qualified healthcare professional.
# 🙏 Acknowledgements

Thanks to the open-source Python and Machine Learning community for providing the tools and libraries used to develop this project.

**Python • Pandas • NumPy • Scikit-learn • Matplotlib • Seaborn • Joblib • Streamlit**
<div align="center">

# ❤️ Thank You for Visiting!

## Heart Disease Prediction

### Machine Learning | Healthcare | Python | Streamlit

**Made with ❤️ by SUMITRA NAYAK**

</div>

