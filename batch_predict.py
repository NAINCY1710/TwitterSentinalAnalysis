import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics import classification_report, accuracy_score

# Load vectorizer and model once
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

def predict_from_csv(uploaded_file):
    # Read uploaded CSV
    data = pd.read_csv(uploaded_file)
    
    if "text" not in data.columns:
        st.error("CSV must contain a column named 'text'.")
        return
    
    tweets = data["text"]
    
    # Transform tweets
    X_new = vectorizer.transform(tweets)
    
    # Predict
    predictions = model.predict(X_new)
    
    # Add predictions to dataframe
    data["predicted_sentiment"] = predictions
    
    # Display results
    st.subheader("Predictions")
    st.dataframe(data)
    
    # Optional: if true labels exist, show accuracy
    if "airline_sentiment" in data.columns:
        true_labels = data["airline_sentiment"]
        st.write("Accuracy:", accuracy_score(true_labels, predictions))
        st.text("Classification Report:\n" + classification_report(true_labels, predictions))
    
    # Optionally, allow download
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, "predicted_tweets.csv", "text/csv")
