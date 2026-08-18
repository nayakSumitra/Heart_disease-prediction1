import joblib
import pandas as pd

# Load the trained model
model = joblib.load("heart_disease_model.pkl")

print("Model loaded successfully!")
import joblib
import pandas as pd


# --------------------------------------------------
# 1. Load the trained model
# --------------------------------------------------

model = joblib.load("heart_disease_model.pkl")

print("Model loaded successfully!")


# --------------------------------------------------
# 2. Get patient information
# --------------------------------------------------

print("\nEnter Patient Details")
print("-" * 40)

age = float(input("Age: "))

sex = int(input("Sex (0 = Female, 1 = Male): "))

chest_pain = int(input("Chest pain type (1-4): "))

bp = float(input("Blood Pressure (BP): "))

cholesterol = float(input("Cholesterol: "))

fbs = int(input("FBS over 120 (0 = No, 1 = Yes): "))

ekg = int(input("EKG results (0, 1, or 2): "))

max_hr = float(input("Maximum Heart Rate (Max HR): "))

exercise_angina = int(
    input("Exercise angina (0 = No, 1 = Yes): ")
)

st_depression = float(
    input("ST depression: ")
)

slope = int(
    input("Slope of ST (1, 2, or 3): ")
)

vessels = int(
    input("Number of vessels fluro (0-3): ")
)

thal = int(
    input("Thallium (3, 6, or 7): ")
)


# --------------------------------------------------
# 3. Create input DataFrame
# --------------------------------------------------

patient_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "Chest pain type": [chest_pain],
    "BP": [bp],
    "Cholesterol": [cholesterol],
    "FBS over 120": [fbs],
    "EKG results": [ekg],
    "Max HR": [max_hr],
    "Exercise angina": [exercise_angina],
    "ST depression": [st_depression],
    "Slope of ST": [slope],
    "Number of vessels fluro": [vessels],
    "Thallium": [thal]
})


# --------------------------------------------------
# 4. Make prediction
# --------------------------------------------------

prediction = model.predict(patient_data)[0]

probability = model.predict_proba(patient_data)[0][1]


# --------------------------------------------------
# 5. Display prediction
# --------------------------------------------------

print("\n" + "=" * 50)
print("HEART DISEASE PREDICTION")
print("=" * 50)

if prediction == 1:
    print("Prediction: PRESENCE")
    print(
        f"Probability of Heart Disease: "
        f"{probability * 100:.2f}%"
    )
else:
    print("Prediction: ABSENCE")
    print(
        f"Probability of Heart Disease: "
        f"{(1 - probability) * 100:.2f}%"
    )

print("=" * 50)