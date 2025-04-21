import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from catboost import CatBoostRegressor
import datetime
import plotly.graph_objects as go
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_synthetic_control(df):
    st.subheader("Synthetic Control")

    # Let user select date column
    date_col = st.selectbox("Select Date Column:", options=df.columns)
    if not date_col:
        st.error("Please select a date column.")
        st.stop()

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).reset_index(drop=True)  # Ensure sorted
    except:
        st.error("Invalid date column format.")
        st.stop()

    # Add time-based features
    add_time_features = st.checkbox("Add time-based features (year, month, day, etc.)")
    if add_time_features:
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek
        st.write("Added features: year, month, day, day_of_week")

    # Select outcome column
    outcome_col = st.selectbox(
        "Select Outcome Column:",
        options=[col for col in df.columns if col != date_col],
    )
    if not outcome_col:
        st.error("Please select an outcome column.")
        st.stop()

    # Treatment start date
    treatment_start = st.date_input(
        "Treatment Start Date",
        min_value=df[date_col].min(),
        max_value=df[date_col].max(),
    )
    treatment_start = pd.Timestamp(treatment_start)

    # Model hyperparameters
    st.sidebar.subheader("Model Settings")
    iterations = st.sidebar.slider("Iterations", 100, 1000, 500)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
    depth = st.sidebar.slider("Tree Depth", 1, 10, 3)
    bootstrap_iters = st.sidebar.slider("Bootstrap Iterations", 50, 500, 100)

    # Select predictors
    predictors = st.multiselect(
        "Select Predictors",
        options=[col for col in df.columns if col not in [date_col, outcome_col]],
    )

    if st.button("Run Analysis") and predictors:
        progress = st.progress(0)
        status = st.empty()

        # Split data
        status.text("Splitting data...")
        pre_treatment = df[df[date_col] < treatment_start]
        if len(pre_treatment) < 10:
            st.error("Insufficient pre-treatment data.")
            st.stop()
        
        # Train-validation split
        split_idx = int(len(pre_treatment) * 0.8)
        train = pre_treatment.iloc[:split_idx]
        val = pre_treatment.iloc[split_idx:]
        progress.progress(10)

        # Train model
        status.text("Training model...")
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            verbose=0,
            random_seed=42
        )
        model.fit(train[predictors], train[outcome_col])
        val_pred = model.predict(val[predictors])
        rmse = np.sqrt(mean_squared_error(val[outcome_col], val_pred))
        mae = mean_absolute_error(val[outcome_col], val_pred)
        r2 = r2_score(val[outcome_col], val_pred)
        progress.progress(30)

        # Show validation metrics
        st.metric("Validation RMSE", f"{rmse:.2f}")
        st.metric("Validation MAE", f"{mae:.2f}")
        st.metric("Validation R²", f"{r2:.2f}")

        # Predict post-treatment
        status.text("Predicting...")
        post_treatment = df[df[date_col] >= treatment_start].copy()
        post_treatment["y_hat"] = model.predict(post_treatment[predictors])
        post_treatment["effect"] = post_treatment[outcome_col] - post_treatment["y_hat"]
        progress.progress(50)

        # Bootstrap
        status.text("Bootstrapping...")
        n = len(post_treatment)
        bootstrap_effects = np.zeros((n, bootstrap_iters))
        for i in range(bootstrap_iters):
            block_size = 7  # Weekly blocks (adjust as needed)
            n_blocks = int(np.ceil(len(pre_treatment)/block_size))

            # Create equal-sized blocks by truncating to divisible length
            max_length = n_blocks * block_size
            trimmed_data = pre_treatment.iloc[:max_length]

            # Create blocks with equal length
            blocks = [trimmed_data.iloc[i*block_size:(i+1)*block_size] 
                    for i in range(n_blocks)]

            # Sample block indices instead of blocks directly
            sampled_indices = np.random.choice(np.arange(n_blocks), 
                                            size=n_blocks, 
                                            replace=True)
            sampled_blocks = [blocks[i] for i in sampled_indices]

            # Concatenate sampled blocks
            boot_sample = pd.concat(sampled_blocks, axis=0).iloc[:len(pre_treatment)]
            
            
            boot_model = CatBoostRegressor(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                verbose=0,
                random_seed=42+i
            )
            boot_model.fit(boot_sample[predictors], boot_sample[outcome_col])
            pred = boot_model.predict(post_treatment[predictors])
            bootstrap_effects[:, i] = post_treatment[outcome_col].values - pred
            progress.progress(50 + int(50 * (i+1)/bootstrap_iters))

        # Calculate CIs
        post_treatment["effect_lower"] = np.percentile(bootstrap_effects, 2.5, axis=1)
        post_treatment["effect_upper"] = np.percentile(bootstrap_effects, 97.5, axis=1)
        post_treatment["effect_median"] = np.percentile(bootstrap_effects, 50, axis=1)

        # Merge results
        df = df.merge(
            post_treatment[[date_col, "y_hat", "effect", "effect_lower", "effect_upper", "effect_median"]],
            on=date_col,
            how="left"
        )

        # Plotting
        fig = px.line(df, x=date_col, y=[outcome_col, "y_hat"],
                      title="Actual vs Synthetic Control")
        fig.add_vline(x=treatment_start.timestamp()*1000, line_dash="dash", line_color="red")
        st.plotly_chart(fig)

        # Effect plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=post_treatment[date_col], y=post_treatment["effect_upper"],
                         line=dict(width=0), showlegend=False, name='95% CI'))
        fig2.add_trace(go.Scatter(x=post_treatment[date_col], y=post_treatment["effect_lower"],
                         fill="tonexty", fillcolor="rgba(0,100,200,0.2)", line=dict(width=0), name='95% CI'))
        fig2.add_trace(go.Scatter(x=post_treatment[date_col], y=post_treatment["effect_median"],
                         line=dict(color="blue"), name='Median Effect'))
        fig2.update_layout(title="Treatment Effect with Confidence Interval")
        st.plotly_chart(fig2)

        post_treatment["cumulative_effect"] = post_treatment["effect_median"].cumsum()
        post_treatment["cumulative_lower"] = post_treatment["effect_lower"].cumsum()
        post_treatment["cumulative_upper"] = post_treatment["effect_upper"].cumsum()

        # Cumulative effect plot
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=post_treatment[date_col],
            y=post_treatment["cumulative_upper"],
            line=dict(width=0),
            showlegend=False
        ))
        fig3.add_trace(go.Scatter(
            x=post_treatment[date_col],
            y=post_treatment["cumulative_lower"],
            fill="tonexty",
            fillcolor="rgba(0,200,100,0.2)",
            line=dict(width=0),
            name="95% CI"
        ))
        fig3.add_trace(go.Scatter(
            x=post_treatment[date_col],
            y=post_treatment["cumulative_effect"],
            line=dict(color="green"),
            name="Cumulative Effect"
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="grey")
        fig3.update_layout(
            title="Cumulative Treatment Effect Over Time",
            xaxis_title="Date",
            yaxis_title="Cumulative Effect"
        )
        st.plotly_chart(fig3)

        progress.progress(100)
        status.text("Analysis Complete")

    elif not predictors:
        st.warning("Please select at least one predictor.")