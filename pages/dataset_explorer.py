import streamlit as st
import matplotlib.pyplot as plt
from utils.visualization import plot_label_distribution
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

st.set_page_config(page_title="Dataset Explorer", layout="wide")
st.title("Dataset Explorer")

# Access data from session state
df = st.session_state.df

# Show basic dataset info
st.subheader("Dataset Overview")
st.write(f"Number of samples: {df.shape[0]}")
st.write(f"Number of features: {df.shape[1]}")

# Display sample data
st.subheader("Sample Data")
st.dataframe(df.head())

# Show label distribution
st.subheader("Label Distribution")
cols = st.columns(3)

for i, column in enumerate(['fuel', 'machine', 'part']):
    with cols[i % 3]:
        fig = plot_label_distribution(df, column)
        st.pyplot(fig)

# Sample sentences per sentiment
st.subheader("Sample Sentences by Sentiment")

# Select sentiment to explore
sentiment_to_explore = st.selectbox(
    "Choose sentiment to explore:",
    ["fuel", "machine", "part"]
)

# Display examples for each sentiment value
st.write(f"### {sentiment_to_explore.capitalize()} Sentiment Examples")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("#### Negative")
    for _, row in df[df[sentiment_to_explore] == 'negative'].head(3).iterrows():
        st.write(f"- {row['sentence']}")

with col2:
    st.write("#### Neutral")
    for _, row in df[df[sentiment_to_explore] == 'neutral'].head(3).iterrows():
        st.write(f"- {row['sentence']}")

with col3:
    st.write("#### Positive")
    for _, row in df[df[sentiment_to_explore] == 'positive'].head(3).iterrows():
        st.write(f"- {row['sentence']}")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
    
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
    
# Change Background Streamlit
set_background(r"background_music.gif")
