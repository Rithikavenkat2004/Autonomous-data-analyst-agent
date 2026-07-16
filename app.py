"""
app.py
------
Streamlit front-end. Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd

from core.auto_eda import profile_dataframe, profile_to_prompt_context
from core.agent import DataAnalystAgent
from core.tools import generate_chart

st.set_page_config(page_title="Autonomous Data Analyst Agent", layout="wide")
st.title("Autonomous Data Analyst Agent")
st.caption("Upload a CSV. Ask questions in plain English. The agent writes and runs its own pandas code.")

if "agent" not in st.session_state:
    st.session_state.agent = None
    st.session_state.df = None
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None and st.session_state.df is None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.session_state.agent = DataAnalystAgent(df)
    st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

if st.session_state.df is not None:
    df = st.session_state.df

    with st.expander("Auto-EDA summary", expanded=True):
        profile = profile_dataframe(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", profile["n_rows"])
        col2.metric("Columns", profile["n_cols"])
        col3.metric("Columns with outliers", len(profile["outliers"]))
        st.dataframe(df.head(10), use_container_width=True)
        st.text(profile_to_prompt_context(profile))

    st.divider()
    st.subheader("Ask a question about your data")

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["insight"])
            if turn["success"]:
                with st.expander("Show generated code + raw result"):
                    st.code(turn["code"], language="python")
                    st.write(turn["result"])

    question = st.chat_input("e.g. Which product category has the highest average order value?")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Writing and running analysis code..."):
                response = st.session_state.agent.ask(question)
            st.write(response["insight"])
            if response["success"]:
                with st.expander("Show generated code + raw result"):
                    st.code(response["code"], language="python")
                    st.write(response["result"])
            else:
                st.error(response["error"])
        st.session_state.chat_history.append(response)

    st.divider()
    st.subheader("Quick chart builder")
    c1, c2, c3, c4 = st.columns(4)
    chart_type = c1.selectbox("Chart type", ["bar", "line", "scatter", "histogram", "box", "pie"])
    x_col = c2.selectbox("X axis", df.columns)
    y_col = c3.selectbox("Y axis (optional)", [None] + list(df.columns))
    color_col = c4.selectbox("Color by (optional)", [None] + list(df.columns))
    if st.button("Generate chart"):
        fig = generate_chart(chart_type, df, x_col, y_col, color_col)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Couldn't generate that chart with the selected columns.")
else:
    st.info("Upload a CSV to get started. No file yet? Try the sample in sample_data/sales.csv")
