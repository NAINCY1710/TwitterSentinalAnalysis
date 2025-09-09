import streamlit as st

st.title("Twitter Sentiment Analysis Project")

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page:", ["Home", "Batch Predictions", "Live Twitter Predictions"])

if page == "Home":
    st.header("Welcome!")
    st.write("Use this app to analyze the sentiment of tweets.")

elif page == "Batch Predictions":
    st.header("Batch Predictions")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        # Call your batch_predict module function here
        import batch_predict
        batch_predict.predict_from_csv(uploaded_file)  # assuming you have a function like this

elif page == "Live Twitter Predictions":
    st.header("Live Twitter Predictions")
    hashtag = st.text_input("Enter hashtag to search:")
    if st.button("Predict Sentiment"):
        if hashtag:
            # Call your twitter_api_predict module function here
            import twitter_api_predict
            twitter_api_predict.predict_from_hashtag(hashtag)  # assuming you have a function like this
        else:
            st.warning("Please enter a hashtag before predicting.")
