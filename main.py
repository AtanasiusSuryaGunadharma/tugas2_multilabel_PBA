from utils.data_loader import load_data
import streamlit as st
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

# Change Background Streamlit
set_background(r"background_music.gif")

# Set page configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Multi-label Text Classification",
    layout="wide"
)

# Now we can import other modules and set up session state

# Initialize session state variables
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'label_columns' not in st.session_state:
    st.session_state.label_columns = None

# Load data
if 'df' not in st.session_state:
    st.session_state.df = load_data()

# Add title and description
st.title("Automotive Reviews Multi-label Text Classification")
st.markdown("Multi-label classification for automotive reviews across different aspects: fuel, machine, and parts.")

# Main page content
st.write("""
## Welcome to the Multi-label Text Classification App
         
This application demonstrates text classification that can predict multiple labels simultaneously.

### Available Pages:

1. **Dataset Explorer** - Explore and understand the dataset
2. **Model Training** - Train and evaluate multi-label classification models
3. **Prediction** - Make predictions on new text inputs

Use the sidebar to navigate between pages.
""")

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
    
# Show dataset overview
st.subheader("Dataset Overview")
df = st.session_state.df
st.write(f"Number of samples: {df.shape[0]}")
st.write(f"Number of features: {df.shape[1]}")
st.dataframe(df.head(5))
