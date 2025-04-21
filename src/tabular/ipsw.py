import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy.stats import norm
from statsmodels.stats.weightstats import DescrStatsW

def run_ipsw(df):
    st.subheader("Inverse Propensity Score Weighting (IPSW)")

    # User inputs
    treatment_col = st.selectbox("Select Treatment Indicator Column:", options=df.columns)
    outcome_col = st.selectbox("Select Outcome Column:", options=[col for col in df.columns if col != treatment_col])
    ipsw_predictors = st.multiselect(
        "Select Predictor Columns for Propensity Score:",
        options=[col for col in df.columns if col not in [treatment_col, outcome_col]]
    )
    
    # New: Clipping threshold before run button
    epsilon = st.slider("Clipping Threshold (ε) for Propensity Scores", 
                      min_value=1e-6, max_value=0.1, 
                      value=1e-3, format="%.6f")
    
    if not ipsw_predictors:
        st.warning("Please select at least one predictor column for IPSW.")
        return

    # Handle missing data
    if df[[treatment_col, outcome_col] + ipsw_predictors].isnull().any().any():
        st.warning("Missing values detected in selected columns. Rows with missing values will be dropped.")
        df = df.dropna(subset=[treatment_col, outcome_col] + ipsw_predictors)

    if st.button("Run IPSW Analysis"):
        X = df[ipsw_predictors]
        T = df[treatment_col].astype(int)  # Ensure treatment is binary
        Y = df[outcome_col]
        
        # Propensity score modeling with improved convergence
        try:
            propensity_model = LogisticRegression(random_state=42, max_iter=1000)
            propensity_model.fit(X, T)
        except Exception as e:
            st.error(f"Propensity model failed to converge: {e}")
            return
            
        propensity_scores = propensity_model.predict_proba(X)[:, 1]

        # Dynamic clipping threshold
        propensity_scores = np.clip(propensity_scores, epsilon, 1 - epsilon)

        # Weight calculation
        weights = np.zeros_like(T, dtype=float)
        weights[T == 1] = 1 / propensity_scores[T == 1]
        weights[T == 0] = 1 / (1 - propensity_scores[T == 0])

        # ATE calculation using weighted statistics
        treated_mean = np.average(Y[T == 1], weights=weights[T == 1])
        control_mean = np.average(Y[T == 0], weights=weights[T == 0])
        ATE = treated_mean - control_mean

        # Improved variance estimation using weighted stats
        def weighted_se(y, weights):
            """Calculate weighted standard error of the mean"""
            weighted_stats = DescrStatsW(y, weights=weights, ddof=1)
            return weighted_stats.std_mean  # Direct return of scalar value

        # In the ATE calculation section:
        se_treated = weighted_se(Y[T == 1], weights[T == 1])
        se_control = weighted_se(Y[T == 0], weights[T == 0])
        se_ATE = np.sqrt(se_treated**2 + se_control**2)
            
        # Confidence intervals
        z_critical = norm.ppf(0.975)
        ci_lower = ATE - z_critical * se_ATE
        ci_upper = ATE + z_critical * se_ATE

        # Results display
        st.subheader("Primary Results")
        st.metric("Average Treatment Effect (ATE)", f"{ATE:.4f}", 
                 delta=f"[95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")

        # Enhanced visualizations
        st.subheader("Diagnostic Visualizations")
        
        # Propensity score distribution
        fig_ps = px.histogram(
            df, x=propensity_scores, color=T.replace({0: 'Control', 1: 'Treatment'}).astype(str),
            nbins=50, barmode='overlay',
            title="Propensity Score Distribution by Treatment Group",
            labels={'x': 'Propensity Score', 'color': 'Treatment Status'},
            color_discrete_map={
                'Treatment': '#636EFA',
                'Control': '#EF553B'
            },
            opacity=0.6
        )
        st.plotly_chart(fig_ps)

        # Covariate balance assessment
        balance_data = []
        for col in ipsw_predictors:
            # Before weighting
            smd_unweighted = (X.loc[T == 1, col].mean() - X.loc[T == 0, col].mean()) / X[col].std()
            
            # After weighting
            weighted_mean_treated = np.average(X.loc[T == 1, col], weights=weights[T == 1])
            weighted_mean_control = np.average(X.loc[T == 0, col], weights=weights[T == 0])
            smd_weighted = (weighted_mean_treated - weighted_mean_control) / X[col].std()
            
            balance_data.append({
                'Covariate': col,
                'SMD (Unweighted)': smd_unweighted,
                'SMD (Weighted)': smd_weighted
            })

        balance_df = pd.DataFrame(balance_data)
        
        # Create the Love plot figure
        fig_love = go.Figure()

        # Add markers for unweighted SMDs with circle markers and hover labels
        fig_love.add_trace(go.Scatter(
            x=balance_df['SMD (Unweighted)'],
            y=balance_df['Covariate'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='circle'),
            name='Unweighted SMD',
            hovertemplate='Covariate: %{y}<br>SMD (Unweighted): %{x:.3f}<extra></extra>'
        ))

        # Add markers for weighted SMDs with square markers and hover labels
        fig_love.add_trace(go.Scatter(
            x=balance_df['SMD (Weighted)'],
            y=balance_df['Covariate'],
            mode='markers',
            marker=dict(color='blue', size=10, symbol='square'),
            name='Weighted SMD',
            hovertemplate='Covariate: %{y}<br>SMD (Weighted): %{x:.3f}<extra></extra>'
        ))

        # Draw dashed lines connecting the unweighted and weighted SMDs for each covariate
        for index, row in balance_df.iterrows():
            fig_love.add_trace(go.Scatter(
                x=[row['SMD (Unweighted)'], row['SMD (Weighted)']],
                y=[row['Covariate'], row['Covariate']],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ))

        fig_love.update_layout(
            title="Covariate Balance: Love Plot",
            xaxis_title="Standardized Mean Difference (SMD)",
            yaxis_title="Covariates",
            shapes=[dict(type='line', x0=0, x1=0, y0=0, y1=1, yref='paper')],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_love)

        # Visualization
        st.subheader("Visualization")

        # Create plotting DataFrame
        plot_df = pd.DataFrame({
            outcome_col: Y,
            'Treatment': T.astype(str),
            'Weights': weights
        })

        # 1. Original Outcome Distribution
        fig_original = px.histogram(
            plot_df, 
            x=outcome_col, 
            color=plot_df['Treatment'].replace({'0': 'Control', '1': 'Treatment'}).astype(str),
            title="Original Outcome Distribution by Treatment Group",
            nbins=40,
            barmode='overlay',
            opacity=0.6,
            color_discrete_map={'Treatment': '#636EFA', 'Control': '#EF553B'},
            labels={'x': outcome_col, 'color': 'Treatment Status'}
        )
        
        # Add unweighted mean lines
        original_means = plot_df.groupby('Treatment')[outcome_col].mean()
        fig_original.add_vline(
            x=original_means['1'], 
            line_dash="dash", 
            line_color='#636EFA',
            annotation_text=f"Treated Mean:<br>{original_means['1']:.2f}",
        )
        fig_original.add_vline(
            x=original_means['0'], 
            line_dash="dash", 
            line_color='#EF553B',
            annotation_text=f"Control Mean:<br>{original_means['0']:.2f}"
        )
        st.plotly_chart(fig_original)

        # 2. Weighted Outcome Distribution
        bin_edges = np.histogram_bin_edges(plot_df[outcome_col], bins=20)
        weighted_data = []

        colors = {'1':'blue', '0':'red'}
        
        for treatment_status in ['1', '0']:
            subset = plot_df[plot_df['Treatment'] == treatment_status]
            counts, bins = np.histogram(
                subset[outcome_col], 
                bins=bin_edges, 
                weights=subset['Weights']
            )
            weighted_data.append(go.Bar(
                x=bins,
                y=counts,
                name=f'{'Treatment' if treatment_status == '1' else 'Control'}',
                opacity=0.6,
                marker_color=colors[treatment_status],
                hoverinfo='y+name'
            ))

        fig_weighted = go.Figure(data=weighted_data)
        fig_weighted.update_layout(
            title="Weighted Outcome Distribution by Treatment Group",
            barmode='overlay',
            xaxis_title=outcome_col,
            yaxis_title="Weighted Count",
            legend_title="Treatment Status"
        )

        # Add mean lines
        fig_weighted.add_vline(
            x=treated_mean, 
            line_dash="dash", 
            line_color='blue',
            annotation_text=f"Treated Mean:<br>{treated_mean:.2f}"
        )
        fig_weighted.add_vline(
            x=control_mean, 
            line_dash="dash", 
            line_color='red',
            annotation_text=f"Control Mean:<br>{control_mean:.2f}"
        )
        st.plotly_chart(fig_weighted)

        # Model diagnostics
        st.subheader("Model Diagnostics")
        
        # Propensity model performance
        auc = roc_auc_score(T, propensity_scores)
        st.write(f"Propensity Model ROC AUC: `{auc:.3f}`")
        
        # Effective sample size
        ess_treated = (np.sum(weights[T == 1])**2 / np.sum(weights[T == 1]**2))
        ess_control = (np.sum(weights[T == 0])**2 / np.sum(weights[T == 0]**2))
        st.write(f"Effective Sample Size (Treated/Control): `{ess_treated:.1f}`/`{ess_control:.1f}`")

        # Distributional checks
        fig_balance = px.box(
            df, x=T.astype(str), y=outcome_col,
            title="Outcome Distribution by Treatment Status",
            labels={'x': 'Treatment Group', 'y': outcome_col}
        )
        st.plotly_chart(fig_balance)

        # Detailed summary table
        summary_stats = pd.DataFrame({
            'Group': ['Treated', 'Control'],
            'N': [len(T[T == 1]), len(T[T == 0])],
            'Mean Outcome (Raw)': [Y[T == 1].mean(), Y[T == 0].mean()],
            'Mean Outcome (Weighted)': [treated_mean, control_mean],
            'ESS': [ess_treated, ess_control]
        })

        st.dataframe(
            summary_stats.style.format({
                'N': '{:.0f}',  # Format as integer
                'Mean Outcome (Raw)': '{:.3f}',
                'Mean Outcome (Weighted)': '{:.3f}',
                'ESS': '{:.1f}'
            }),
            use_container_width=True
        )

