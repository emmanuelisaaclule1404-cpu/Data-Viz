import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain_groq import Cha
Groq
from langchain_experimental.agents import create_pandas_dataframe_agent

# Page Setup
st.set_page_config(page_title="OpenSource Data AI", layout="wide")
st.title("📊 AI Data Analysis Dashboard")

# Security: Get the Open Source API Key from Streamlit/GitHub Secrets
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
# Sidebar File Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Read the data
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    # 1. Data Analysis (Describing Data)
    st.write("### 📋 Data Preview & Summary")
    col_a, col_b = st.columns(2)
    with col_a:
        st.dataframe(df.head(5))
    with col_b:
        st.write(df.describe()) # This 'describes' the math (mean, min, max)
# 2. Graphing Section
    st.divider()
    st.subheader("📈 Visualization Lab")
    
    viz_col1, viz_col2 = st.columns([1, 2])
    
    with viz_col1:
        chart_type = st.selectbox("Pick a Graph", ["Bar", "Line", "Scatter", "Pie"])
        x_axis = st.selectbox("Select X Axis", df.columns)
        y_axis = st.selectbox("Select Y Axis", df.columns)
    
    with viz_col2:
        fig, ax = plt.subplots()
        if chart_type == "Bar":
            sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif chart_type == "Line":
            sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif chart_type == "Scatter":
            sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif chart_type == "Pie":
            df[x_axis].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
      # 3. Chatbot Integration
    st.divider()
    st.subheader("🤖 Chat with your Data")
    
    if api_key:
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
        agent = create_pandas_dataframe_agent(llm, df, allow_dangerous_code=True)
        
        user_query = st.text_input("Ask a question (e.g., 'Filter rows where Sales > 100')")
        if user_query:
            with st.spinner("AI is analyzing..."):
                response = agent.run(user_query)
                st.write(response)
    else:
        st.warning("Please provide an API Key to use the Chatbot.")
