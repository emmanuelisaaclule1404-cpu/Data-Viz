import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_huggingface import HuggingFaceEndpoint
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- 1. SETTINGS & AUTHENTICATION ---
st.set_page_config(page_title="Free AI Data Studio", layout="wide")
st.title("📊 Free Open-Source Data Analyst")

# Look for the API key in Streamlit Secrets (for GitHub Deployment)
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    st.sidebar.warning("API Token not found in Secrets.")
    hf_token = st.sidebar.text_input("Paste Hugging Face Token", type="password")

# --- 2. DATA INPUT SECTION ---
st.sidebar.header("Step 1: Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Load data into a DataFrame (df)
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # --- 3. DATA ANALYSIS (DESCRIBING DATA) ---
    st.subheader("📋 Data Overview & Statistics")
    col_pre1, col_pre2 = st.columns([1, 1])
    
    with col_pre1:
        st.write("**Data Preview (Top 5 Rows):**")
        st.dataframe(df.head(5))
    
    with col_pre2:
        st.write("**Statistical Summary (Describe):**")
        st.write(df.describe())

    st.divider()

    # --- 4. VISUALIZATION SUITE (GRAPHS) ---
    st.subheader("📈 Visualization Lab")
    viz_col1, viz_col2 = st.columns([1, 2])

    with viz_col1:
        st.write("### Chart Settings")
        chart_type = st.selectbox("Select Chart Type", ["Bar", "Line", "Scatter", "Histogram", "Pie"])
        x_axis = st.selectbox("Select X-Axis", df.columns)
        y_axis = st.selectbox("Select Y-Axis", df.columns)
        chart_color = st.color_picker("Pick Chart Color", "#3498db")

    with viz_col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == "Bar":
            sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax, color=chart_color)
        elif chart_type == "Line":
            sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax, color=chart_color)
        elif chart_type == "Scatter":
            sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax, color=chart_color)
        elif chart_type == "Histogram":
            sns.histplot(df[x_axis], ax=ax, color=chart_color, kde=True)
        elif chart_type == "Pie":
            df[x_axis].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('') # Removes ugly overlapping label
            
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.divider()

    # --- 5. AI CHATBOT INTEGRATION ---
    st.subheader("🤖 Chat with Your Dataset")
    
    if hf_token:
        # We use Mistral 7B - a top-tier open source model
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        
        try:
            llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                huggingfacehub_api_token=hf_token,
                temperature=0.1,
                max_new_tokens=512
            )
            
            # The Agent acts as the bridge between the LLM and your Data
            agent = create_pandas_dataframe_agent(
                llm, 
                df, 
                verbose=True, 
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )
            
            user_query = st.text_input("Ask a question about your data (e.g., 'Filter for high values', 'What is the correlation?')")
            
            if user_query:
                with st.spinner("AI is analyzing your table..."):
                    response = agent.run(user_query)
                    st.success(response)
                    
        except Exception as e:
            st.error(f"AI Connection Error: {e}")
    else:
        st.info("Provide a Hugging Face Token in the sidebar to enable the Chatbot.")

else:
    # Instructions for when no file is uploaded
    st.info("Welcome! Please upload a CSV or Excel file in the sidebar to begin your analysis.")
