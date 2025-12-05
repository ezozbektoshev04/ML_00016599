import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Serie A Match Predictor")

st.title("Serie A Match Result Prediction")

# Loading all models
models = joblib.load("app/models/models_dict.pkl")
model_names = list(models.keys())
# Letting users to choose which model to use
chosen_model_name = st.sidebar.selectbox("Choose model", model_names)
chosen_model = models[chosen_model_name]

st.sidebar.header("Match Statistics")

# INPUTS
HS = st.sidebar.number_input("Home Shots", 0, 40, 10)
AS = st.sidebar.number_input("Away Shots", 0, 40, 8)

HST = st.sidebar.number_input("Home Shots on Target", 0, 20, 4)
AST = st.sidebar.number_input("Away Shots on Target", 0, 20, 3)
HF = st.sidebar.number_input("Home Fouls", 0, 30, 12)
AF = st.sidebar.number_input("Away Fouls", 0, 30, 14)
HC = st.sidebar.number_input("Home Corners", 0, 15, 5)
AC = st.sidebar.number_input("Away Corners", 0, 15, 4)
HY = st.sidebar.number_input("Home Yellow Cards", 0, 10, 2)
AY = st.sidebar.number_input("Away Yellow Cards", 0, 10, 2)

HR = st.sidebar.number_input("Home Red Cards", 0, 5, 0)
AR = st.sidebar.number_input("Away Red Cards", 0, 5, 0)

# Engineering features 
ShotDiff = HS - AS
CardDiff = (HY + HR) - (AY + AR)
# Building input row 
input_df = pd.DataFrame([{
    "HS": HS, "AS": AS, "HST": HST, "AST": AST,
    "HF": HF, "AF": AF, "HC": HC, "AC": AC,
    "HY": HY, "AY": AY, "HR": HR, "AR": AR,
    "ShotDiff": ShotDiff, "CardDiff": CardDiff
}])
if st.button("Predict Result"):
    pred = chosen_model.predict(input_df)[0]
    st.subheader(f"Chosen model: **{chosen_model_name}**")
    st.subheader(f"Predicted Result: **{pred}**")

#  Showing preprocessing choices 
with st.expander("Show preprocessing steps"):
    st.markdown("""
    **Preprocessing pipeline used before training and prediction:**
    - Removed the `Referee` column because it contained only missing values.
    - Created engineered features:
        - `ShotDiff = HS - AS`
        - `CardDiff = (HY + HR) - (AY + AR)`
    - Selected numerical match statistics as model features.
    - Applied `StandardScaler` to normalize all numerical features.
    - Split dataset into training (80%) and testing (20%) using stratified sampling.
    """)
#  Show evaluation results 
with st.expander("Show model evaluation results"):
    try:
        eval_df = pd.read_csv("app/models/results.csv")
        st.dataframe(eval_df)
        st.markdown("""
        Accuracy is computed on the held-out test set (20% of the data).  
        Random Forest achieved the best performance among the models.
        """)
    except FileNotFoundError:
        st.write("Evaluation results file not found. Please generate it from the notebook.")

