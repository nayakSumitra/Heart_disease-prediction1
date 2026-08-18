import streamlit as st
import pandas as pd
import joblib
# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)
# 2. LOAD TRAINED MODEL
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")


model = load_model()



# 3. TITLE
st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter the patient's medical information below "
    "to predict the presence or absence of heart disease."
)

st.divider()

# 4. PATIENT INPUTS
st.subheader("Patient Information")


age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50,
    step=1
)


sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x:
        "Female (0)" if x == 0 else "Male (1)"
)


chest_pain = st.selectbox(
    "Chest Pain Type",
    options=[1, 2, 3, 4]
)


bp = st.number_input(
    "Blood Pressure (BP)",
    min_value=50,
    max_value=250,
    value=120,
    step=1
)


cholesterol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=600,
    value=200,
    step=1
)


fbs = st.selectbox(
    "FBS over 120",
    options=[0, 1],
    format_func=lambda x:
        "No (0)" if x == 0 else "Yes (1)"
)


ekg = st.selectbox(
    "EKG Results",
    options=[0, 1, 2]
)


max_hr = st.number_input(
    "Maximum Heart Rate (Max HR)",
    min_value=50,
    max_value=250,
    value=150,
    step=1
)


exercise_angina = st.selectbox(
    "Exercise Angina",
    options=[0, 1],
    format_func=lambda x:
        "No (0)" if x == 0 else "Yes (1)"
)


st_depression = st.number_input(
    "ST Depression",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)


slope = st.selectbox(
    "Slope of ST",
    options=[1, 2, 3]
)


vessels = st.selectbox(
    "Number of Vessels Fluro",
    options=[0, 1, 2, 3]
)


thal = st.selectbox(
    "Thallium",
    options=[3, 6, 7]
)


st.divider()

# 5. PREDICTION BUTTON
if st.button(
    "Predict Heart Disease",
    type="primary",
    use_container_width=True
):

    
    # Create DataFrame with EXACT notebook column names
    
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


    
    # Make prediction
    
    prediction = model.predict(patient_data)[0]

    probability = model.predict_proba(
        patient_data
    )[0][1]
    # Display result
    
    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Heart Disease: PRESENCE"
        )

        st.write(
            f"Probability of heart disease: "
            f"**{probability * 100:.2f}%**"
        )

    else:

        st.success(
            "✅ Heart Disease: ABSENCE"
        )

        st.write(
            f"Probability of heart disease: "
            f"**{(1 - probability) * 100:.2f}%**"
        )

    # Show entered values
    with st.expander("View Patient Information"):

        st.dataframe(
            patient_data,
            use_container_width=True
        )



# 6. DISCLAIMER

st.divider()

st.caption(
    "This application is for educational purposes only "
    "and should not be used as a substitute for professional "
    "medical advice."
)
st.title("Heart Disease Prediction")

st.write("Streamlit application is working!")