import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .config("spark.hadoop.fs.native.lib.enabled", "false") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .getOrCreate()

# Path to the saved model - make sure the model path is correct
model_path = "D:\KEC\sem5\BDA\Micro_project\dataset\sentiment_analysis_app\lr_model"  # Absolute path to model

try:
    # Load the trained Logistic Regression model from the saved folder
    model = LogisticRegressionModel.load(model_path)

    # Streamlit UI setup
    st.title("Sentiment Analysis Application")
    st.write("This app predicts the sentiment of the text you provide. Enter some text below:")

    # Create a text area for user input
    user_input = st.text_area("Enter your text here:")

    # If the user clicks the button, make a prediction
    if st.button("Predict Sentiment"):
        if user_input:
            # Convert user input to a DataFrame for PySpark
            input_df = pd.DataFrame({"text": [user_input]})
            spark_df = spark.createDataFrame(input_df)

            # Use the loaded model to make a prediction
            predictions = model.transform(spark_df)

            # Extract prediction result (assuming 'prediction' column exists)
            result = predictions.select("prediction").collect()[0][0]

            # Output sentiment based on prediction
            sentiment = "Positive" if result == 1 else "Negative"  # Adjust based on model's output
            st.write(f"The predicted sentiment is: {sentiment}")
        else:
            st.write("Please enter some text for prediction.")
except Exception as e:
    st.write(f"Error loading model: {e}")
