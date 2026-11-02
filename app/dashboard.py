import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.data_pipeline import load_real_retail_data, transform_to_rfm
from src.models.clv_bgnbd import RetailCLVPipeline

st.set_page_config(page_title="Churn & CLV Intelligence Cockpit", layout="wide")

st.title("📊 Customer Churn & Lifetime Value (CLV) Analytics")
st.markdown("Real-time predictive dashboard powered by BG/NBD & Gamma-Gamma Models on Real E-Commerce Data.")

# Data Initialization with Real Dataset
@st.cache_data
def load_and_fit_data():
    raw_df = load_real_retail_data("online_retail_II.csv")
    rfm = transform_to_rfm(raw_df)
    model = RetailCLVPipeline()
    model.fit(rfm)
    preds = model.predict(rfm, time_horizon_days=90)
    return preds

df_preds = load_and_fit_data()

# KPI Metrics Header
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Active Customers", f"{len(df_preds):,}")
col2.metric("Average Churn Risk", f"{df_preds['p_churn'].mean()*100:.1f}%")
col3.metric("Avg Predicted 90D CLV", f"${df_preds['predicted_clv'].mean():,.2f}")
col4.metric("High-Risk Accounts (>75%)", f"{(df_preds['p_churn'] > 0.75).sum()}")

st.divider()

# Layout: Segmentation & Individual Risk Lookup
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Customer Value vs. Churn Risk Matrix")
    fig = px.scatter(
        df_preds,
        x="p_churn",
        y="predicted_clv",
        size="frequency",
        color="p_churn",
        color_continuous_scale="RdYlGn_r",
        labels={"p_churn": "Churn Probability", "predicted_clv": "Predicted 90D CLV ($)"},
        hover_data=["customer_id", "frequency", "monetary_value"]
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("🔍 Individual Account Lookup")
    selected_id = st.selectbox("Select Customer ID", df_preds['customer_id'].unique())
    cust_data = df_preds[df_preds['customer_id'] == selected_id].iloc[0]
    
    st.write(f"**Customer ID:** {int(selected_id)}")
    st.write(f"**Current Status:** {'🟢 Active' if cust_data['p_active'] > 0.5 else '🔴 At Risk'}")
    st.progress(float(cust_data['p_churn']), text=f"Churn Probability: {cust_data['p_churn']*100:.1f}%")
    st.write(f"**Historical Frequency:** {int(cust_data['frequency'])} repeat purchases")
    st.write(f"**Predicted 90-Day Spend:** ${cust_data['predicted_clv']:.2f}")

    # Actionable Recommendation Engine Logic
    st.markdown("---")
    st.markdown("**💡 Prescriptive Action:**")
    if cust_data['p_churn'] > 0.6 and cust_data['predicted_clv'] > 100:
        st.error("🚨 VIP High Risk: Offer $20 Retention Voucher Immediately!")
    elif cust_data['p_churn'] > 0.6:
        st.warning("⚠️ Low-Value High Risk: Send Automated Re-engagement Email.")
    else:
        st.success("✅ Healthy Account: Enrol in Loyalty Upsell Program.")