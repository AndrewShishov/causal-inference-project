import streamlit as st
import pandas as pd
from timeseries.synthetic_control import run_synthetic_control
from timeseries.causalimpact_model import run_causal_impact  # New module for CausalImpact analysis
from tabular.ipsw import run_ipsw

st.title("Causal Inference Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # --- SIDEBAR: DATA TYPE SELECTION ---
    data_type = st.sidebar.radio("Select Data Type", ("Timeseries", "Tabular"))

    # --- TIMESERIES BLOCK ---
    if data_type == "Timeseries":
        st.header("Timeseries Analysis")
        # Let user choose which timeseries analysis to run:
        analysis_type = st.sidebar.radio("Select Analysis Type", ("Synthetic Control", "Causal Impact"))
        if analysis_type == "Synthetic Control":
            run_synthetic_control(df)
        else:  # Causal Impact
            run_causal_impact(df)

    # --- TABULAR BLOCK ---
    elif data_type == "Tabular":
        st.header("Tabular Data Analysis")
        run_ipsw(df)
