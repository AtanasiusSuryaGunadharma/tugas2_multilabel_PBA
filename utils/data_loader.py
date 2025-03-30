import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    """
    Load and cache the dataset from file
    """
    df = pd.read_csv(r"D:\SURYA\UAJY\Semester 6\Pengolahan Bahasa Alami\Pertemuan 6 Praktek Modul 2\modul2_Classification_Text\streamlit_modul2\web\multilabel\data\test_preprocess.csv")
    return df
