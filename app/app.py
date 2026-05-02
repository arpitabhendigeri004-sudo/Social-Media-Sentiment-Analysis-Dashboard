import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.title("📊 Social Media Sentiment Analysis Dashboard")

option = st.radio("Choose Input Type:", ["Single Text", "Upload CSV"])

# Single text prediction
if option == "Single Text":
    text = st.text_area("Enter text here")

    if st.button("Analyze"):
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        st.success(f"Sentiment: {prediction}")

# CSV upload
else:
    file = st.file_uploader("Upload CSV with 'text' column")

    if file is not None:
        df = pd.read_csv(file)

        df['Predicted'] = model.predict(vectorizer.transform(df['text']))

        st.write(df.head())

        fig = px.pie(df, names='Predicted', title='Sentiment Distribution')
        st.plotly_chart(fig)