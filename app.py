import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved preprocessors and model
preprocessor = joblib.load("preprocessorX.pkl")
encoder = joblib.load("label_encoder_y.pkl")
model = joblib.load("RandomForestClassifier_model.pkl")

st.title("Final Prediction: Will They Reply?")
st.header("Input Your Conditions")

col1, col2 = st.columns(2)

with col1:
    time_since = st.number_input("Time Since Last Message in Minutes", min_value=0, max_value=100, value=0)
    replied_last_time = st.selectbox("Replied Last Time in Minutes", options=["Yes", "No"])
    who_texted_first = st.selectbox("Who Texted First", options=["Them", "You"])
    left_on_read = st.selectbox("Left on Read", options=["No", "Yes"])
    emoji_use = st.selectbox("Emoji Use", options=["A lot", "Sometimes", "Never"])

with col2:
    response_length = st.number_input("Response Length", min_value=0, max_value=100, value=10)
    seen_story_no_reply = st.selectbox("Seen Story No Reply", options=["No", "Yes"])
    only_you_start_convos = st.selectbox("Only You Start Convos", options=["No", "Yes"])
    last_reaction = st.selectbox("Last Reaction", options=["LOL", "Haha", "Dry response", "Ignored"])
    mood = st.selectbox("Mood", options=["Confident", "Meh", "Desperate"])

input_data = pd.DataFrame({
    "Replied Last Time": [replied_last_time],
    "Time Since Last Message": [time_since],
    "Who Texted First": [who_texted_first],
    "Response Length": [response_length],
    "Left on Read": [left_on_read],
    "Seen Story No Reply": [seen_story_no_reply],
    "Emoji Use": [emoji_use],
    "Only You Start Convos": [only_you_start_convos],
    "Last Reaction": [last_reaction],
    "Mood": [mood]
})

if st.button("Predict"):
    encoded_input = preprocessor.transform(input_data)
    transformed_X_test = pd.DataFrame(data=encoded_input, columns=preprocessor.get_feature_names_out())
    prediction = model.predict(transformed_X_test) 
    if np.all(prediction == 0):  # If the model predicts all zeros
        st.success("Final Prediction: Wait a bit")
    else:
        decoded_prediction = encoder.inverse_transform(prediction)
        st.success(f"Final Prediction: {decoded_prediction[0][0]}")
