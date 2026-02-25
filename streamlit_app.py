import streamlit as st
import pandas as pd
import re
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# Streamlit Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Database Chatbot", layout="wide")
st.title("🤖 AI Database Chatbot (Auto Schema Discovery)")

session = get_active_session()
