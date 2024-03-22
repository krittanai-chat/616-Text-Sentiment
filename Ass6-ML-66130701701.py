
import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model_bay
model_bay = pickle.load(open('naive_bayes-66130701701.sav', 'rb'))

# Load Vectorizer
vectorizer = pickle.load(open('vectorizer-66130701701.sav', 'rb'))

# Set title
st.title("Review Sentiment Prediction using Naive Bayes")

# Text input for user input
user_input = st.text_input("Enter your review:")

# Transform user input using TfidfVectorizer
user_input_vec = vectorizer.transform([user_input])

# Predict sentiment
pred = model_bay.predict(user_input_vec)

# Display prediction result
st.write("## Prediction Result:")
st.write('Sentiment:', pred[0])
