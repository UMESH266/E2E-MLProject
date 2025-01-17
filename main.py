from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
import streamlit as st
import pandas as pd
import numpy as np

# model_trainer = TrainingPipeline()
# score, model = model_trainer.start_training()
# print("Best model: ", model)
# print("Best score: ", score)

st.set_page_config(page_title='Math Score prediction', page_icon='⛽', layout='wide')

# Page Title
st.markdown("<h1 style='text-align: center;'>Student's Math score predictor</h1>", unsafe_allow_html=True)  

# Interface to collect user input
with st.form("Data Entry form", clear_on_submit=True):
    col1, col2, col3 = st.columns(3)
    gender = col1.selectbox("Gender: ", options=['male', 'female'])
    race = col2.selectbox("Race / Ethnicity: ", options=['group A', 'group B', 'group C', 'group D', 'group E'])
    parent_education = col3.selectbox("Parental level of education: ", options=["bachelor's degree", 'some college', "master's degree",
       "associate's degree", 'high school', 'some high school'])
    
    col1, col2 = st.columns(2)
    lunch = col1.selectbox("Lunch type: ", options=['standard', 'free/reduced'])
    test_course=col2.selectbox("Test preparation course: ", options=['none', 'completed'])
    
    col1, col2 = st.columns(2)
    writing_score = col1.number_input("Writing score: ", min_value=0, max_value=100)
    reading_score = col2.number_input("Reading score: ", min_value=0, max_value=100)

    col1, col2, col3, col4 = st.columns(4)
    submit = col4.form_submit_button("Predict score")

if submit:
    data_dict = [{"gender" : gender, 
                 "race_ethnicity": race, 
                 "parental_level_of_education": parent_education, 
                 "lunch": lunch, 
                 "test_preparation_course": test_course, 
                 "writing_score": writing_score, 
                 "reading_score": reading_score
                 }]

    data_df = pd.DataFrame(data_dict)
    st.write("User input: ")
    st.write(data_df)
    
    predictor = PredictionPipeline()
    result = predictor.predict(data_df)
    
    st.write("Predicted Math score: ")
    st.write(np.round(result, 2))
