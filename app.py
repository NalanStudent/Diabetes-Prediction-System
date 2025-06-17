import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Set page layout
st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

# Layout
col2, col1 = st.columns([3, 1])

# Left Column: Model selection, Inputs, and Prediction
with col1:
    # 🔽 Model selection
    model_choice = st.selectbox("Select Model", ["Random Forest" , "Logistic Regression", "K-Nearest Neighbors"])

    # Load selected model
    if model_choice == "Random Forest":
        model = joblib.load("rf_tuned.pkl")
    elif model_choice == "Logistic Regression":
        model = joblib.load("lr_tuned.pkl")
    elif model_choice == "K-Nearest Neighbors":
        model = joblib.load("knn_tuned.pkl")

    st.subheader("Enter Patient Information")
    pregnancies = st.number_input('Pregnancies', min_value=0)
    glucose = st.number_input('Glucose', min_value=0)
    blood_pressure = st.number_input('Blood Pressure', min_value=0)
    skin_thickness = st.number_input('Skin Thickness', min_value=0)
    insulin = st.number_input('Insulin', min_value=0)
    bmi = st.number_input('BMI', min_value=0.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0)
    age = st.number_input('Age', min_value=0)

    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error(f"The {model_choice} model predicts: DIABETIC")
        else:
            st.success(f"The {model_choice} model predicts: NOT DIABETIC")

# Right Column: Info and Image
with col2:
    
    st.title("Diabetes Prediction System")
    
    image = Image.open("images/DiabetesPredictionLogo.png")
    st.image(image, caption="Predictive Health System", use_column_width=500)

    st.subheader("System Purpose")
    st.markdown("""
    The objective of this system is to diagnose diabetes using a dataset of female patients who are at least 21 years old and of Pima Indian heritage.
    This data is used to train a predictive model that determines whether a patient has diabetes based on several diagnostic features, including glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age.
    """)

    st.subheader("Description of Data")
    st.markdown("""
    - **Pregnancies**: Number of pregnancies  
    - **Glucose**: Glucose level in blood  
    - **BloodPressure**: Blood pressure measurement  
    - **SkinThickness**: Thickness of the skin  
    - **Insulin**: Insulin level in blood  
    - **BMI**: Body mass index  
    - **DiabetesPedigreeFunction**: Diabetes percentage  
    - **Age**: Age  
    - **Outcome**: 1 = Diabetic, 0 = Not Diabetic
    """)

    st.subheader("Model Information (Accuracy)")
    st.markdown(f"""
    - **Random Forest** : 77.27%
    - **Logistic Regression** : 74.68%
    - **KNN** : 70.13%

    """)

    st.subheader("System By")
    st.markdown("""
    - Nalan  
    - Teevanraj  
    - Janardhan  
    - Liu
    """)
