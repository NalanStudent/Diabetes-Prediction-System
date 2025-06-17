# Diabetes Prediction System

This is a web-based application for predicting diabetes using machine learning models. The system allows users to input patient information and select from three trained models to predict whether a patient is diabetic.

## Features

- **Model Selection:** Choose between Random Forest, Logistic Regression, and K-Nearest Neighbors.
- **User Input:** Enter patient data such as pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age.
- **Prediction:** Get instant predictions and feedback on diabetes status.
- **Model Accuracy:**
  - Random Forest: 77.27%
  - Logistic Regression: 74.68%
  - KNN: 70.13%

## Project Structure

```
app.py
knn_tuned.pkl
lr_tuned.pkl
rf_tuned.pkl
images/
    banner_icon.png
    diabetes.png
    DiabetesPredictionLogo.png
    DiabetesPredictionLogoCircle2.png
```

## Getting Started

### Prerequisites

- Python 3.7+
- [Streamlit](https://streamlit.io/)
- numpy
- joblib
- pillow

Install dependencies:
```sh
pip install streamlit numpy joblib pillow
```

### Running the App

```sh
streamlit run app.py
```

Open the provided local URL in your browser to use the app.

## Data Description

- **Pregnancies:** Number of pregnancies
- **Glucose:** Glucose level in blood
- **BloodPressure:** Blood pressure measurement
- **SkinThickness:** Thickness of the skin
- **Insulin:** Insulin level in blood
- **BMI:** Body mass index
- **DiabetesPedigreeFunction:** Diabetes percentage
- **Age:** Age
- **Outcome:** 1 = Diabetic, 0 = Not Diabetic

## Authors

- Nalan  
- Teevanraj  
- Janardhan  
- Liu

## License

This project is for educational