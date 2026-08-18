

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
| Source | UCI Heart Disease Dataset |

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

### `Logistic Regression`

Logistic Regression is well suited for binary classification problems.

The model predicts:

```text
0 → Absence of Heart Disease
1 → Presence of Heart Disease.

