import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

import streamlit as st


my_model = load_model('model.h5')

## Reading the label encoder and one hot encoder as well as scaler


with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('one_hot_encoder_geo.pkl','rb') as file:
    one_hot_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


#Streamlit app

st.title('Avishek's Predictor of Customer Churn/Retention')


#Take User Input
geography = st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

#Prepare Input Data

input_data=pd.DataFrame({

    'CreditScore': [credit_score],
    ##No Geography because we have to hot encode that later
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary':[estimated_salary]

})




geo_encoded=one_hot_encoder_geo.transform([[geography]]).toarray()

encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_encoder_geo.get_feature_names_out())


#Combine the data

input_data=pd.concat([input_data.reset_index(drop=True),encoded_df],axis=1)

#Scale the data

input_data_scaled=scaler.transform(input_data)

prediction=my_model.predict(input_data_scaled)

st.write(f"Churn Probability: {prediction[0][0]:.2f}")

if(prediction[0][0]>0.5):
    st.write("The Customer will churn")
else:
    st.write("The customer will not churn")
