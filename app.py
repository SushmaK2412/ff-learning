"""
Streamlit Dashboard for Federated Learning Financial Forecasting Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
import sys
import json
import torch

# Streamlit compatibility - use experimental_rerun for older versions
if hasattr(st, 'rerun'):
    rerun_func = st.rerun
elif hasattr(st, 'experimental_rerun'):
    rerun_func = st.experimental_rerun
else:
    def rerun_func():
        st.experimental_rerun()

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated.federated_trainer import FederatedTrainer
from models.forecasting_model import FinancialForecastingModel
from data.download_data import clean_old_data, download_institution_data

# Page configuration
st.set_page_config(
    page_title="Federated Learning for Financial Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'comparison' not in st.session_state:
    st.session_state.comparison = None

def ensure_fresh_data():
    """Ensure fresh data is available on app startup - cleans old data and downloads fresh"""
    try:
        # Clean old data files first
        clean_old_data()
        # Then download fresh data
        download_institution_data()
        return True
    except Exception as e:
        print(f"Error refreshing data: {str(e)}")
        return False

def main():
    # Ensure fresh data on startup (only once per session)
    if 'data_initialized' not in st.session_state:
        with st.spinner("🔄 Cleaning old data and downloading fresh data files..."):
            success = ensure_fresh_data()
        if success:
            st.session_state.data_initialized = True
            rerun_func()
            return
        else:
            st.error("⚠️ Failed to refresh data. Using existing data files if available.")
            st.session_state.data_initialized = True
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Federated Learning for Financial Forecasting</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Task selection with descriptions
        st.markdown("### 📋 Forecasting Task")
        st.info("""
        **What is Federated Learning?**
        
        Instead of sharing raw data, each financial institution trains a model locally and only shares model updates. This preserves privacy while improving accuracy!
        """)
        
        task_options = {
            'cash_flow_30d': '💰 Cash Flow (30 days) - Predict cash flow 30 days ahead',
            'cash_flow_60d': '💰 Cash Flow (60 days) - Predict cash flow 60 days ahead',
            'cash_flow_90d': '💰 Cash Flow (90 days) - Predict cash flow 90 days ahead',
            'default_risk': '⚠️ Default Risk - Estimate probability of loan defaults',
            'investment_return': '📈 Investment Return - Predict returns on investments'
        }
        
        task = st.selectbox(
            "Select Forecasting Task",
            options=list(task_options.keys()),
            index=0,
            format_func=lambda x: task_options[x],
            help="Choose which financial metric to forecast"
        )
        
        # Show task description
        task_descriptions = {
            'cash_flow_30d': 'Predicts how much cash a financial institution will have 30 days in the future. Helps with liquidity planning.',
            'cash_flow_60d': 'Predicts cash flow 60 days ahead. Useful for medium-term financial planning.',
            'cash_flow_90d': 'Predicts cash flow 90 days ahead. Helps with quarterly planning and budgeting.',
            'default_risk': 'Estimates the likelihood that borrowers will fail to repay loans. Critical for risk management.',
            'investment_return': 'Forecasts expected returns on investment portfolios. Helps optimize asset allocation.'
        }
        
        st.caption(f"💡 **{task_descriptions[task]}**")
        
        # Training parameters
        st.markdown("### 🎯 Training Parameters")
        st.caption("Adjust these to control how the model learns")
        
        num_rounds = st.slider(
            "Federated Rounds", 
            5, 20, 10,
            help="Number of times the model is shared and aggregated across financial institutions. More rounds = better accuracy but longer training."
        )
        local_epochs = st.slider(
            "Local Epochs per Round", 
            3, 10, 5,
            help="How many times each financial institution trains on their data before sharing. More epochs = better local learning."
        )
        
        # Data status
        st.subheader("📊 Data Status")
        data_files = [
            'data/processed/Institution_A_cash_flow_30d.csv',
            'data/processed/Institution_B_cash_flow_30d.csv',
            'data/processed/Institution_C_cash_flow_30d.csv'
        ]
        
        data_available = all(os.path.exists(f) for f in data_files)
        
        if data_available:
            st.success("✓ Data files available")
        else:
            st.warning("⚠ Data files not found")
            if st.button("📥 Download Data"):
                with st.spinner("Downloading financial data from Yahoo Finance..."):
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, "data/download_data.py"],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        st.success("✓ Data downloaded successfully!")
                        rerun_func()
                    else:
                        st.error(f"Error: {result.stderr}")
        
        st.markdown("---")
        
        # Start training button
        start_training = st.button("🚀 Start Federated Training", type="primary", use_container_width=True)
        if start_training:
            if not data_available:
                st.error("Please download data first!")
            else:
                # Reset state
                st.session_state.training_complete = False
                st.session_state.metrics = None
                st.session_state.comparison = None
                # Initialize trainer
                with st.spinner("Initializing federated learning system..."):
                    st.session_state.trainer = FederatedTrainer(
                        task=task,
                        num_rounds=num_rounds,
                        local_epochs=local_epochs
                    )
                rerun_func()
    
    # Main content area
    if st.session_state.trainer is None:
        # Show instructions or wait for sidebar button
        pass
    
    # Training section
    if st.session_state.trainer is not None and not st.session_state.training_complete:
        st.header("🔄 Federated Learning Training")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Training output
        training_output = st.empty()
        
        # Run training
        with training_output.container():
            # Create a placeholder for training logs
            log_placeholder = st.empty()
            
            # Redirect stdout to capture training logs
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                history = st.session_state.trainer.train()
            
            # Display logs
            logs = f.getvalue()
            log_placeholder.text_area("Training Logs", logs, height=300)
            
            # Update progress
            progress_bar.progress(1.0)
            status_text.success("✓ Training Complete!")
            
            # Get metrics
            st.session_state.metrics = st.session_state.trainer.get_metrics()
            
            # Run comparison
            with st.spinner("Comparing with centralized approach..."):
                st.session_state.comparison = st.session_state.trainer.compare_with_centralized()
            
            st.session_state.training_complete = True
            rerun_func()
    
    # Results section
    if st.session_state.training_complete and st.session_state.metrics:
        st.header("📈 Results & Analysis")
        
        st.info("""
        **Understanding the Results:**
        
        - **MAPE (Mean Absolute Percentage Error)**: Lower is better. Shows average prediction error as a percentage.
        - **Loss**: Model's training error. Decreases as the model learns.
        - **Data Transfer**: Amount of data shared between financial institutions. Federated learning reduces this significantly!
        - **Improvement**: How much better federated learning is compared to centralized approach.
        """)
        
        metrics = st.session_state.metrics
        history = metrics['history']
        
        # Key metrics row
        st.markdown("### 📊 Key Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Final MAPE",
                f"{history['aggregated_mape'][-1]:.2f}%",
                delta=f"-{history['aggregated_mape'][0] - history['aggregated_mape'][-1]:.2f}%"
            )
        
        with col2:
            st.metric(
                "Final Loss",
                f"{history['aggregated_loss'][-1]:.4f}",
                delta=f"-{history['aggregated_loss'][0] - history['aggregated_loss'][-1]:.4f}"
            )
        
        with col3:
            # Get data transfer from metrics (handle different key names)
            data_transfer_mb = 0
            if 'total_federated_transfer_mb' in metrics:
                data_transfer_mb = metrics['total_federated_transfer_mb']
            elif 'data_transfer_metrics' in metrics and metrics['data_transfer_metrics']:
                data_transfer_mb = metrics['data_transfer_metrics'].get('federated_mb', 0)
            elif 'total_data_transfer_mb' in metrics:
                data_transfer_mb = metrics['total_data_transfer_mb']
            else:
                # Calculate from history if available
                if 'history' in metrics and 'data_transfer' in metrics['history']:
                    total_bytes = sum(metrics['history']['data_transfer'])
                    data_transfer_mb = total_bytes / (1024 * 1024)
            
            st.metric(
                "Total Data Transfer",
                f"{data_transfer_mb:.2f} MB"
            )
        
        with col4:
            improvement = ((history['aggregated_mape'][0] - history['aggregated_mape'][-1]) / 
                          history['aggregated_mape'][0]) * 100
            st.metric(
                "Improvement",
                f"{improvement:.1f}%"
            )
        
        st.markdown("---")
        
        # Training progress charts
        st.markdown("### 📉 Training Progress")
        st.caption("Watch how the model improves over training rounds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Training Loss Over Rounds")
            st.caption("Loss decreases as the model learns. Lower is better.")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=history['round'],
                y=history['aggregated_loss'],
                mode='lines+markers',
                name='Aggregated Loss',
                line=dict(color='#1f77b4', width=2)
            ))
            fig_loss.update_layout(
                xaxis_title="Round",
                yaxis_title="Loss",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            st.markdown("#### MAPE Over Rounds")
            st.caption("MAPE (Mean Absolute Percentage Error) shows prediction accuracy. Lower is better.")
            fig_mape = go.Figure()
            fig_mape.add_trace(go.Scatter(
                x=history['round'],
                y=history['aggregated_mape'],
                mode='lines+markers',
                name='Aggregated MAPE',
                line=dict(color='#ff7f0e', width=2)
            ))
            fig_mape.update_layout(
                xaxis_title="Round",
                yaxis_title="MAPE (%)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_mape, use_container_width=True)
        
        # Client performance
        st.markdown("---")
        st.markdown("### 📊 Client Performance")
        st.caption("Each financial institution (client) trains locally and shares only model updates, not raw data. This preserves privacy!")
        client_data = []
        for client_id, client_metrics in metrics['client_metrics'].items():
            client_data.append({
                'Client': client_id,
                'Data Sent (MB)': client_metrics['data_sent'] / (1024 * 1024),
                'Data Received (MB)': client_metrics['data_received'] / (1024 * 1024)
            })
        
        if client_data:
            client_df = pd.DataFrame(client_data)
            # Use markdown table to avoid pyarrow/pandas version issues
            try:
                st.dataframe(client_df, use_container_width=True)
            except Exception:
                # Fallback: display as HTML table (works with old pandas)
                html_table = client_df.to_html(index=False, classes='dataframe', table_id='client-table')
                st.markdown(html_table, unsafe_allow_html=True)
        
        # Comparison with centralized
        if st.session_state.comparison:
            st.markdown("---")
            st.markdown("### 🔄 Federated vs Centralized Comparison")
            
            st.info("""
            **Why Federated Learning is Powerful:**
            
            - ✅ **Better Accuracy**: Learns from more diverse data across financial institutions
            - ✅ **Privacy Preserved**: Raw data never leaves each financial institution
            - ✅ **Less Data Transfer**: Only model parameters shared, not entire datasets
            - ✅ **Distributed Computing**: Uses each institution's own infrastructure
            """)
            
            comp = st.session_state.comparison
            
            # Accuracy metrics
            st.markdown("#### 📊 Forecasting Accuracy")
            st.caption("MAPE (Mean Absolute Percentage Error) - Lower values mean better predictions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Centralized MAPE",
                    f"{comp['centralized']['mape']:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Federated MAPE",
                    f"{comp['federated']['mape']:.2f}%"
                )
            
            with col3:
                acc_improvement = comp['accuracy_improvement']['improvement_percent']
                st.metric(
                    "Accuracy Improvement",
                    f"{acc_improvement:.1f}%",
                    delta=f"{acc_improvement:.1f}%"
                )
            
            # Computational Efficiency
            st.markdown("#### 💻 Computational Efficiency")
            st.caption("Federated learning distributes computation across financial institutions, reducing central server load")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Centralized Time",
                    f"{comp['centralized']['training_time']:.2f}s"
                )
            
            with col2:
                st.metric(
                    "Federated Time",
                    f"{comp['federated']['training_time']:.2f}s"
                )
            
            with col3:
                comp_eff = comp['computational_efficiency']['resource_reduction_percent']
                st.metric(
                    "Resource Reduction",
                    f"{comp_eff:.1f}%",
                    delta=f"{comp_eff:.1f}%"
                )
            
            # Data Transfer
            st.markdown("#### 📡 Data Transfer Requirements")
            st.caption("Federated learning only transfers model parameters (small), not raw data (large). This is the key privacy benefit!")
            data_transfer = metrics.get('data_transfer_metrics', {})
            if data_transfer:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Centralized Transfer",
                        f"{data_transfer.get('centralized_mb', 0):.2f} MB"
                    )
                
                with col2:
                    st.metric(
                        "Federated Transfer",
                        f"{data_transfer.get('federated_mb', 0):.2f} MB"
                    )
                
                with col3:
                    data_reduction = data_transfer.get('reduction_percent', 0)
                    st.metric(
                        "Data Transfer Reduction",
                        f"{data_reduction:.1f}%",
                        delta=f"{data_reduction:.1f}%"
                    )
            
            # Comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=['Centralized', 'Federated'],
                    y=[comp['centralized']['mape'], comp['federated']['mape']],
                    marker_color=['#d62728', '#2ca02c'],
                    text=[f"{comp['centralized']['mape']:.2f}%", f"{comp['federated']['mape']:.2f}%"],
                    textposition='auto'
                ))
                fig_comp.update_layout(
                    title="Forecasting Accuracy (MAPE)",
                    yaxis_title="MAPE (%)",
                    height=400
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            
            with col2:
                if data_transfer:
                    fig_data = go.Figure()
                    fig_data.add_trace(go.Bar(
                        x=['Centralized', 'Federated'],
                        y=[data_transfer.get('centralized_mb', 0), data_transfer.get('federated_mb', 0)],
                        marker_color=['#d62728', '#2ca02c'],
                        text=[f"{data_transfer.get('centralized_mb', 0):.2f} MB", 
                              f"{data_transfer.get('federated_mb', 0):.2f} MB"],
                        textposition='auto'
                    ))
                    fig_data.update_layout(
                        title="Data Transfer Requirements",
                        yaxis_title="Data Transfer (MB)",
                        height=400
                    )
                    st.plotly_chart(fig_data, use_container_width=True)
        
        # Data transfer visualization
        st.markdown("---")
        st.markdown("### 📡 Data Transfer Analysis")
        st.caption("Shows how much data is transferred per training round. Federated learning dramatically reduces this!")
        
        fig_transfer = go.Figure()
        fig_transfer.add_trace(go.Bar(
            x=history['round'],
            y=[d / (1024 * 1024) for d in history['data_transfer']],
            marker_color='#9467bd',
            name='Data Transfer per Round'
        ))
        fig_transfer.update_layout(
            xaxis_title="Round",
            yaxis_title="Data Transfer (MB)",
            height=400
        )
        st.plotly_chart(fig_transfer, use_container_width=True)
        
        # Summary
        st.markdown("---")
        st.markdown("### 📋 Training Summary")
        st.caption("Federated learning metrics and performance")
        
        # Get comparison metrics if available
        acc_improvement = "N/A"
        data_reduction = "N/A"
        comp_reduction = "N/A"
        
        if st.session_state.comparison:
            acc_improvement = f"{st.session_state.comparison['accuracy_improvement']['improvement_percent']:.1f}%"
            comp_reduction = f"{st.session_state.comparison['computational_efficiency']['resource_reduction_percent']:.1f}%"
        
        if metrics.get('data_transfer_metrics'):
            data_reduction = f"{metrics['data_transfer_metrics']['reduction_percent']:.1f}%"
        
        summary_text = f"""
        **Federated Learning Training Summary:**
        
        - **Forecasting Task**: {task.replace('_', ' ').title()}
        - **Training Rounds**: {num_rounds}
        - **Local Epochs per Round**: {local_epochs}
        - **Final MAPE**: {history['aggregated_mape'][-1]:.2f}%
        - **Clients Participating**: {len(metrics['client_metrics'])}
        
        **Federated Learning Performance:**
        - ✅ **Accuracy Improvement**: {acc_improvement} over centralized approach
        - ✅ **Data Transfer Reduction**: {data_reduction} compared to centralized
        - ✅ **Computational Efficiency**: {comp_reduction} resource reduction
        
        **Privacy Preservation (Core Federated Learning Principle):**
        - 🔒 Raw data never leaves client institutions (stays on each institution's servers)
        - 🔒 Only model parameters are shared (federated averaging) - no sensitive data exposed
        - 🔒 Data sovereignty maintained across all organizations (each institution controls their data)
        
        **How Federated Learning Works:**
        1. Each financial institution trains a model on their own data locally
        2. Institutions share only the trained model parameters (not the data)
        3. Models are combined using federated averaging to create a better global model
        4. The improved model is sent back to each institution
        5. Process repeats until the model converges to optimal accuracy
        
        This demonstrates the core benefits of federated learning: improved forecasting accuracy, 
        reduced data transfer, and computational efficiency gains while maintaining strict privacy requirements.
        """
        
        st.markdown(summary_text)
        
        # Federated Learning Principles Summary
        st.markdown("#### 🎯 Federated Learning Principles Demonstrated")
        
        principles_data = {
            'Principle': [
                'Privacy Preservation',
                'Data Sovereignty',
                'Distributed Training',
                'Model Aggregation',
                'Efficient Communication'
            ],
            'Description': [
                'Raw data never leaves client institutions',
                'Each financial institution maintains control of their data',
                'Training happens locally at each institution',
                'Models combined via federated averaging',
                'Only model parameters shared, not raw data'
            ],
            'Status': [
                '✅ Active',
                '✅ Active',
                '✅ Active',
                '✅ Active',
                '✅ Active'
            ]
        }
        
        principles_df = pd.DataFrame(principles_data)
        # Display principles (avoid pyarrow dependency)
        try:
            st.dataframe(principles_df, use_container_width=True, hide_index=True)
        except Exception:
            # Fallback: display as HTML table (works with old pandas)
            html_table = principles_df.to_html(index=False, classes='dataframe', table_id='principles-table')
            st.markdown(html_table, unsafe_allow_html=True)
    
    # Instructions section (when no training has started)
    if st.session_state.trainer is None:
        st.info("""
        👋 **Welcome to the Federated Learning Financial Forecasting Demo!**
        
        This demo showcases a federated learning system for financial forecasting across 
        multiple financial institutions.
        
        **To get started:**
        1. Ensure data files are available (use the sidebar to download if needed)
        2. Select your forecasting task and training parameters
        3. Click "Start Federated Training" to begin
        
        **Features:**
        - Privacy-preserving model training across 3 financial institutions
        - Real-time training progress visualization
        - Comparison with centralized approach
        - Data transfer analysis
        - Performance metrics tracking
        """)

if __name__ == "__main__":
    main()

