import streamlit as st
import pandas as pd
from utils.preprocessing import preprocess_text
from models.multi_label_classifiers import get_multilabel_classifier, create_vectorizer, create_multilabel_target
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.model_selection import train_test_split
from models.multi_label_classifiers import get_multilabel_classifier, create_vectorizer, evaluate_multilabel_model, create_multilabel_target
from utils.visualization import plot_multilabel_confusion_matrix
from streamlit_extras.let_it_rain import rain
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from datetime import datetime, timedelta
import time
import base64
from pathlib import Path
from st_social_media_links import SocialMediaIcons

st.set_page_config(page_title="Prediction", layout="wide")
st.title("Make Predictions")

# Access data from session state
df = st.session_state.df

# Input text
user_input = st.text_area("Tuliskan review (terserah apa saja):",
                          "Avanza bahan bakar nya boros banget")

# Check if we have trained model stored in session state
if st.session_state.trained_model is None:
    st.warning(
        "No trained model found. Please train a model in the 'Model Training' page first.")
    model_status = st.empty()
else:
    st.success(f"Using trained {st.session_state.model_name} model")
  
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
  
# Menambahkan audio autoplay menggunakan HTML
try:
    with open(r"Lagu_stecu.mp3", "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode()

    audio_html = f"""
    <audio autoplay loop>
        <source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("File audio tidak ditemukan. Pastikan 'natal_lagu3.mp3' sudah ada di direktori project.")

if st.button("Predict"):
    st.info("Predicting...")

    # Preprocess input
    preprocessed_input = preprocess_text(user_input)

    # Check if we have trained model stored
    if st.session_state.trained_model is not None:
        # Use the trained model from session state
        model = st.session_state.trained_model
        vectorizer = st.session_state.vectorizer
        label_columns = st.session_state.label_columns

        # Transform input using stored vectorizer
        input_tfidf = vectorizer.transform([preprocessed_input])
    else:
        # Train a default model if none is available
        model_status.info(
            "No trained model found. Training a default model Random Forest....")

        # Define label columns
        label_columns = ['fuel_negative', 'fuel_neutral', 'fuel_positive',
                         'machine_negative', 'machine_neutral', 'machine_positive',
                         'part_negative', 'part_neutral', 'part_positive']

        # Create multilabel target for training
        y_multilabel = create_multilabel_target(df)

        # Split data
        X_train, _, y_train, _ = train_test_split(
            df['sentence'], y_multilabel, test_size=0.2, random_state=42)

        # Vectorize properly
        vectorizer = create_vectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        input_tfidf = vectorizer.transform([preprocessed_input])

        # Train model
        model = get_multilabel_classifier("Random Forest", n_estimators=100)
        model.fit(X_train_tfidf, y_train)

        # Save to session state for future use
        st.session_state.trained_model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.model_name = "Random Forest (default)"
        st.session_state.label_columns = label_columns

    # Make prediction
    prediction = model.predict(input_tfidf)

    # Display results
    st.success("Prediction complete!")

    # Show the input text
    st.subheader("Input Text")
    st.write(user_input)

    st.subheader("Preprocessed Text")
    st.write(preprocessed_input)

    # Show the predicted labels

    # Create nicer display of predictions
    results = []
    has_predictions = False

    for i, label in enumerate(label_columns):
        if prediction.toarray()[0, i] == 1:
            results.append(label)
            has_predictions = True

    if has_predictions:
        # for label in results:
        #     st.write(f"- {label}")

        # Display visual representation by category
        st.subheader("Prediction Summary")

        # Group by category
        fuel_preds = [col for col in results if col.startswith('fuel_')]
        machine_preds = [col for col in results if col.startswith('machine_')]
        part_preds = [col for col in results if col.startswith('part_')]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Fuel sentiment:**")
            if fuel_preds:
                for pred in fuel_preds:
                    st.markdown(
                        f"<span style='color: red;'>- {pred.replace('fuel_', '')}</span>",
                        unsafe_allow_html=True)
            else:
                st.write("No prediction")

        with col2:
            st.write("**Machine sentiment:**")
            if machine_preds:
                for pred in machine_preds:
                    st.markdown(
                        f"<span style='color: green;'>- {pred.replace('machine_', '')}</span>",
                        unsafe_allow_html=True)
            else:
                st.write("No prediction")

        with col3:
            st.write("**Part sentiment:**")
            if part_preds:
                for pred in part_preds:
                    st.markdown(
                        f"<span style='color: blue;'>- {pred.replace('part_', '')}</span>",
                        unsafe_allow_html=True)
            else:
                st.write("No prediction")
    else:
        st.write("No labels predicted.")
    
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
  
# Change Background Streamlit
set_background(r"background_music.gif")
