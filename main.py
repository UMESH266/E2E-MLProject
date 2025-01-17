from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
import streamlit as st

# model_trainer = TrainingPipeline()
# score, model = model_trainer.start_training()
# print("Best model: ", model)
# print("Best score: ", score)

st.markdown("Students Math score predictor")

# Interface to collect user input
with st.form("Data Entry form", clear_on_submit=True):

    name = st.text_input("Name: ")
    age = st.number_input("Age: ")
    submit = st.form_submit_button("Predict")

if submit:
    data_dict = {"name": name, "age":age}
    st.write(data_dict)
    
