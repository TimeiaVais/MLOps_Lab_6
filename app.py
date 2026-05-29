import streamlit as st

from src.pages.dataset_tab import dataset_tab
from src.pages.error_analysis_tab import error_analysis_tab
from src.pages.explainability_tab import explainability_tab

st.set_page_config(
    page_title="CIFAR10 Dashboard",
    layout="wide"
)

st.title("CIFAR-10 MLOps Dashboard")

tab1, tab2, tab3 = st.tabs([
    "Dataset Exploration",
    "Error Analysis",
    "Prediction & Explainability"
])

with tab1:
    dataset_tab()

with tab2:
    error_analysis_tab()

with tab3:
    explainability_tab()