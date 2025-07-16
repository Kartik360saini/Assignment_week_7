import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import matplotlib.pyplot as plt

st.title(" Rain Prediction by Date")

np.random.seed(42)
n_samples = 300

temperature = np.random.normal(30, 7, n_samples)
humidity = np.random.uniform(40, 100, n_samples)
wind_speed = np.random.uniform(0, 30, n_samples)
pressure = np.random.normal(1013, 10, n_samples)

rain = ((humidity > 70).astype(int) +
        (pressure < 1010).astype(int) +
        (temperature < 25).astype(int))
rain = (rain >= 2).astype(int)

train_data = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed,
    'pressure': pressure,
    'rain': rain
})

X = train_data[['temperature', 'humidity', 'wind_speed', 'pressure']]
y = train_data['rain']

model = LogisticRegression()
model.fit(X, y)

st.write("Select a date to predict rain :")
date_input = st.date_input("Date", value=datetime.today())

if st.button("Predict Rain"):
    date_ord = date_input.toordinal()
    np.random.seed(date_ord)
    temperature = round(np.random.normal(30, 7), 2)
    humidity = round(np.random.uniform(40, 100), 2)
    wind_speed = round(np.random.uniform(0, 30), 2)
    pressure = round(np.random.normal(1013, 10), 2)

    features_df = pd.DataFrame([{
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure
    }])

    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0][1]

    st.write(" Generated Weather Features")
    st.table(features_df)

    st.success(f" Rain prediction for {date_input}: {'Yes' if prediction == 1 else 'No'} (Probability: {probability:.2f})")

    st.write(" Prediction Probability")
    fig, ax = plt.subplots()
    ax.bar(['No Rain', 'Rain'], prob, color=['skyblue', 'steelblue'])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)
