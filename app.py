import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open(r"saved_models/2/model/model.pkl", 'rb'))
encoder = pickle.load(open(r"saved_models/2/target_encoder/target_encoder.pkl", 'rb'))
transformer = pickle.load(open(r"saved_models/2/transformer/transformer.pkl", 'rb'))


st.title("Insurance Premium Prediction")

sex = st.selectbox("Please enter your gender", ("male", "female"))

age = st.text_input("Enter your age", 23)
age = int(age)

bmi = st.text_input("Enter your BMI", 20)
bmi = float(bmi)

children = st.selectbox("Please select number of children", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
children = int(children)

smoker = st.selectbox("Please select smoker category", ("no", "yes"))

region = st.selectbox("Please select your region", ("southwest", "northwest", "southeast", "northeast"))

features = {}
features['age'] = age
features['sex'] = sex
features['bmi'] = bmi
features['children'] = children
features['smoker'] = smoker
features['region'] = region

df = pd.DataFrame(features, index = [0])

# Data Encoding
df['region'] = encoder.transform(df['region'])
df['sex'] = df['sex'].map({"male": 1, "female": 0})
df['smoker'] = df['smoker'].map({"yes": 1, "no": 0})

# Data Transformation
df = transformer.transform(df)

# Prediction
y_pred = model.predict(df)
pred_value = float(y_pred[0])

if st.button("Predict Insurance Premium"):
    st.header(f"{round(pred_value, 2)} INR")
