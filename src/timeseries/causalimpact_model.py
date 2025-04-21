import numpy as np
import pandas as pd
import streamlit as st
from causalimpact import CausalImpact
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
# st.set_option('showErrorDetails', False)

import re
import pandas as pd
import streamlit as st
from causalimpact import CausalImpact

def parse_and_display_summary(summary_text):
    """
    Parse the CausalImpact summary text and display it in a structured format using Streamlit.

    Parameters:
    - summary_text: str, the summary output from CausalImpact (e.g., impact.summary()).
    """
    # Initialize variables to store extracted values
    actual_avg, actual_cum = None, None
    prediction_avg, prediction_sd_avg, prediction_cum, prediction_sd_cum = None, None, None, None
    ci_avg_lower, ci_avg_upper, ci_cum_lower, ci_cum_upper = None, None, None, None
    abs_effect_avg, abs_effect_sd_avg, abs_effect_cum, abs_effect_sd_cum = None, None, None, None
    abs_ci_avg_lower, abs_ci_avg_upper, abs_ci_cum_lower, abs_ci_cum_upper = None, None, None, None
    rel_effect_avg, rel_effect_sd_avg, rel_effect_cum, rel_effect_sd_cum = None, None, None, None
    rel_ci_avg_lower, rel_ci_avg_upper, rel_ci_cum_lower, rel_ci_cum_upper = None, None, None, None
    posterior_p, posterior_prob = None, None

    # Split the summary text into lines
    lines = summary_text.split('\n')

    # Flag to track the current section (main, absolute, or relative)
    current_section = 'main'

    # Parse each line to extract metrics
    for line in lines:
        # Extract Actual values
        actual_match = re.search(r'Actual\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
        if actual_match:
            actual_avg, actual_cum = map(float, actual_match.groups())

        # Extract Prediction (s.d.) values
        prediction_match = re.search(r'Prediction \(s\.d\.\)\s+(\d+\.\d+) \((\d+\.\d+)\)\s+(\d+\.\d+) \((\d+\.\d+)\)', line)
        if prediction_match:
            prediction_avg, prediction_sd_avg, prediction_cum, prediction_sd_cum = map(float, prediction_match.groups())

        # Extract 95% CI for main results
        if current_section == 'main' and '95% CI' in line and 'Absolute' not in line and 'Relative' not in line:
            ci_match = re.search(r'\[(\d+\.\d+), (\d+\.\d+)\]\s+\[(\d+\.\d+), (\d+\.\d+)\]', line)
            if ci_match:
                ci_avg_lower, ci_avg_upper, ci_cum_lower, ci_cum_upper = map(float, ci_match.groups())

        # Switch to absolute effect section and extract values
        if 'Absolute effect (s.d.)' in line:
            current_section = 'absolute'
            abs_effect_match = re.search(r'(\d+\.\d+) \((\d+\.\d+)\)\s+(\d+\.\d+) \((\d+\.\d+)\)', line)
            if abs_effect_match:
                abs_effect_avg, abs_effect_sd_avg, abs_effect_cum, abs_effect_sd_cum = map(float, abs_effect_match.groups())

        # Extract 95% CI for absolute effect
        if current_section == 'absolute' and '95% CI' in line:
            abs_ci_match = re.search(r'\[(\d+\.\d+), (\d+\.\d+)\]\s+\[(\d+\.\d+), (\d+\.\d+)\]', line)
            if abs_ci_match:
                abs_ci_avg_lower, abs_ci_avg_upper, abs_ci_cum_lower, abs_ci_cum_upper = map(float, abs_ci_match.groups())

        # Switch to relative effect section and extract values
        if 'Relative effect (s.d.)' in line:
            current_section = 'relative'
            rel_effect_match = re.search(r'(\d+\.\d+)% \((\d+\.\d+)%\)\s+(\d+\.\d+)% \((\d+\.\d+)%\)', line)
            if rel_effect_match:
                rel_effect_avg, rel_effect_sd_avg, rel_effect_cum, rel_effect_sd_cum = map(float, rel_effect_match.groups())

        # Extract 95% CI for relative effect
        if current_section == 'relative' and '95% CI' in line:
            rel_ci_match = re.search(r'\[(\d+\.\d+)%, (\d+\.\d+)%\]\s+\[(\d+\.\d+)%, (\d+\.\d+)%\]', line)
            if rel_ci_match:
                rel_ci_avg_lower, rel_ci_avg_upper, rel_ci_cum_lower, rel_ci_cum_upper = map(float, rel_ci_match.groups())

        # Extract posterior probabilities
        p_match = re.search(r'Posterior tail-area probability p:\s+(\d+\.\d+)', line)
        if p_match:
            posterior_p = float(p_match.group(1))

        prob_match = re.search(r'Posterior prob. of a causal effect:\s+(\d+\.\d+)%', line)
        if prob_match:
            posterior_prob = float(prob_match.group(1))

    # Create DataFrames for each section with formatted values
    # Main Results
    main_df = pd.DataFrame({
        'Metric': ['Actual', 'Prediction', 'Prediction s.d.', '95% CI'],
        'Average': [
            f"{actual_avg:.2f}" if actual_avg is not None else 'N/A',
            f"{prediction_avg:.2f}" if prediction_avg is not None else 'N/A',
            f"{prediction_sd_avg:.2f}" if prediction_sd_avg is not None else 'N/A',
            f"[{ci_avg_lower:.2f}, {ci_avg_upper:.2f}]" if ci_avg_lower is not None and ci_avg_upper is not None else 'N/A'
        ],
        'Cumulative': [
            f"{actual_cum:.2f}" if actual_cum is not None else 'N/A',
            f"{prediction_cum:.2f}" if prediction_cum is not None else 'N/A',
            f"{prediction_sd_cum:.2f}" if prediction_sd_cum is not None else 'N/A',
            f"[{ci_cum_lower:.2f}, {ci_cum_upper:.2f}]" if ci_cum_lower is not None and ci_cum_upper is not None else 'N/A'
        ]
    })

    # Absolute Effect
    absolute_df = pd.DataFrame({
        'Metric': ['Absolute effect', 's.d.', '95% CI'],
        'Average': [
            f"{abs_effect_avg:.2f}" if abs_effect_avg is not None else 'N/A',
            f"{abs_effect_sd_avg:.2f}" if abs_effect_sd_avg is not None else 'N/A',
            f"[{abs_ci_avg_lower:.2f}, {abs_ci_avg_upper:.2f}]" if abs_ci_avg_lower is not None and abs_ci_avg_upper is not None else 'N/A'
        ],
        'Cumulative': [
            f"{abs_effect_cum:.2f}" if abs_effect_cum is not None else 'N/A',
            f"{abs_effect_sd_cum:.2f}" if abs_effect_sd_cum is not None else 'N/A',
            f"[{abs_ci_cum_lower:.2f}, {abs_ci_cum_upper:.2f}]" if abs_ci_cum_lower is not None and abs_ci_cum_upper is not None else 'N/A'
        ]
    })

    # Relative Effect
    relative_df = pd.DataFrame({
        'Metric': ['Relative effect', 's.d.', '95% CI'],
        'Average': [
            f"{rel_effect_avg:.2f}%" if rel_effect_avg is not None else 'N/A',
            f"{rel_effect_sd_avg:.2f}%" if rel_effect_sd_avg is not None else 'N/A',
            f"[{rel_ci_avg_lower:.2f}%, {rel_ci_avg_upper:.2f}%]" if rel_ci_avg_lower is not None and rel_ci_avg_upper is not None else 'N/A'
        ],
        'Cumulative': [
            f"{rel_effect_cum:.2f}%" if rel_effect_cum is not None else 'N/A',
            f"{rel_effect_sd_cum:.2f}%" if rel_effect_sd_cum is not None else 'N/A',
            f"[{rel_ci_cum_lower:.2f}%, {rel_ci_cum_upper:.2f}%]" if rel_ci_cum_lower is not None and rel_ci_cum_upper is not None else 'N/A'
        ]
    })

    # Display the results in Streamlit
    st.subheader("Main Results")
    st.table(main_df)

    st.subheader("Absolute Effect")
    st.table(absolute_df)

    st.subheader("Relative Effect")
    st.table(relative_df)

    st.subheader("Posterior Probabilities")
    st.write(f"Posterior tail-area probability p: {posterior_p:.2f}" if posterior_p is not None else "Posterior tail-area probability p: N/A")
    st.write(f"Posterior prob. of a causal effect: {posterior_prob:.2f}%" if posterior_prob is not None else "Posterior prob. of a causal effect: N/A")



def run_causal_impact(df):
    """
    Run causal impact analysis using the pycausalimpact library.
    
    Parameters:
    - df: pandas DataFrame containing the data with a date column, outcome, and optional predictors.
    """
    st.subheader("Causal Impact Analysis")

    # **Select Date Column**
    date_col = st.selectbox("Select Date Column:", options=df.columns)
    if not date_col:
        st.error("Please select a date column.")
        return

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).reset_index(drop=True)  # Ensure data is sorted by date
    except Exception:
        st.error("Invalid date column format.")
        return

    # **Optionally Add Time-Based Features**
    add_time_features = st.checkbox("Add time-based features (year, month, day, etc.)")
    if add_time_features:
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek
        st.write("Added features: year, month, day, day_of_week")

    # **Select Outcome Column**
    outcome_col = st.selectbox(
        "Select Outcome Column:",
        options=[col for col in df.columns if col != date_col],
    )
    if not outcome_col:
        st.error("Please select an outcome column.")
        return

    # **Select Predictors (Covariates)**
    predictors = st.multiselect(
        "Select Predictors (optional)",
        options=[col for col in df.columns if col not in [date_col, outcome_col]],
    )

    # **Treatment Start Date**
    treatment_start = st.date_input(
        "Treatment Start Date",
        min_value=df[date_col].min().date(),
        max_value=df[date_col].max().date(),
    )
    treatment_start = pd.Timestamp(treatment_start)

    # **Run Analysis Button**
    if st.button("Run Analysis"):
        # Prepare data for pycausalimpact: set date as index, outcome as first column, followed by predictors
        data = df.set_index(date_col)[[outcome_col] + predictors]
        
        # Define pre-intervention and post-intervention periods
        pre_period = [data.index.min(), treatment_start - pd.Timedelta(days=1)]
        post_period = [treatment_start, data.index.max()]

        # **Validate Periods**
        if pre_period[1] < pre_period[0] or post_period[1] < post_period[0]:
            st.error("Invalid treatment start date.")
            return
        if len(data.loc[pre_period[0]:pre_period[1]]) < 10:
            st.error("Insufficient pre-treatment data (minimum 10 observations required).")
            return
        if len(data.loc[post_period[0]:post_period[1]]) == 0:
            st.error("No post-treatment data available.")
            return

        # **Run CausalImpact Analysis**
        st.write("Running Causal Impact analysis...")
        try:
            impact = CausalImpact(data, pre_period, post_period)
        except Exception as e:
            st.error(f"Error running CausalImpact: {str(e)}")
            return

        # **Display Results**
        # Plot: Uses pycausalimpact's built-in plotting (three panels: original, pointwise effect, cumulative effect)
        st.subheader("Causal Impact Plot")
        fig = impact.plot()
        st.pyplot(fig)

        # Summary: Display textual summary of the causal effect
        st.subheader("Summary")
        st.text(impact.summary())

        parse_and_display_summary(impact.summary(output='summary'))

    else:
        st.info("Select options and click 'Run Analysis' to proceed.")