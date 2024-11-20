import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit app title
st.title("Sentiment Analysis Application")

# Input text box
user_input = st.text_area("Enter a Tweet to Analyze Sentiment:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.write("Please enter text for analysis.")
    else:
        # Preprocess input and make prediction
        input_features = vectorizer.transform([user_input])
        prediction = model.predict(input_features)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        # Display the result
        st.write(f"**Sentiment:** {sentiment}")
