import streamlit as st 
import numpy as np
import  tensorflow as tf
import pandas as pd
import pickle

## firstly load the trained model
model=tf.keras.models.load_model('model.h5')

## load the preprocessor
with open('preprocessor.pkl','rb') as file:
    preprocessor=pickle.load(file)

# set title of streamlit app
st.title("Customer Churn Prediction")

# make place for inputs
geography=st.selectbox('Geography',options=["France","Germany","Spain"])
gender=st.selectbox('Gender',options=["Male","Female"])
age=st.text_input('Age',placeholder=0)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_credit_card=st.selectbox('Have Credit card?',options=[0,1])
is_active_member=st.selectbox('Is an active member?',options=[0,1])

# now convert these data into a df
input_data_df=pd.DataFrame({
    "CreditScore":[credit_score],
    "Geography":[geography],
    "Gender":[gender],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_credit_card],
    "IsActiveMember":[is_active_member],
    "EstimatedSalary":[estimated_salary]
})

# now apply preprocessor on the input data
input_data_transformed=preprocessor.transform(input_data_df)

# now perform the prediction
prediction=model.predict(input_data_transformed)
prediction_probability=prediction[0][0]

# display the final result
st.write(f"There is a {prediction_probability*100}% chance that the customer will churn.")