import streamlit as st
import numpy as np
import pandas as pd
import pickle 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model

#load the model
model = load_model('model.h5')

# Load encoder
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl','rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

##streamlit
st.title('Customer Churn Prediction')

#user input
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
no_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Memeber',[0,1])

input_data = {
    'CreditScore':[credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts':[no_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]

}

#converting into dataframe
input_df = pd.DataFrame(input_data)

#feature engineering
#one hot encoder for geo
geo_res = one_hot_encoder_geo.transform([input_df['Geography']])
geo_res_df = pd.DataFrame(geo_res, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_df, geo_res_df],axis=1)

#label encoder for gender
input_df['Gender'] = label_encoder_gender.transform([input_df['Gender']])

input_df = input_df.drop('Geography', axis = 1)

#scale the input
scaled_input = scaler.transform(input_df)

prediction  = model.predict(scaled_input)

prediction_probability = prediction[0][0]

st.write(f'The churn probaility is {prediction_probability}')


if prediction_probability > 0.5 :
    st.write('The customer is not likely to leave')
else :
    st.write('There are chances of customer will be leaving')

