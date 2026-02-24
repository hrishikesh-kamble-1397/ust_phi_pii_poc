import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# App Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

# -----------------------------------------------------------------------------
# Snowflake Session
# -----------------------------------------------------------------------------
session = get_active_session()

MODEL_NAME = "mistral-large2"
EMBED_MODEL = "snowflake-arctic-embed-m"
SIMILARITY_THRESHOLD = 0.35
MAX_CHUNKS = 30
PAGE_SIZE = 5  # Number of names shown before "+ More"

# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "app_role" not in st.session_state:
    st.session_state.app_role = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your PDFs."}
    ]
if "entity_offset" not in st.session_state:
    st.session_state.entity_offset = 0

# -----------------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------------
def authenticate_user(user_name, password):
    df = session.sql("""
        SELECT APP_ROLE
        FROM AI_POC_DB.PII_PHI_POC.APP_USER_ACCESS
        WHERE UPPER(USER_NAME) = UPPER(:1)
        AND PASSWORD = :2
        AND IS_ACTIVE = TRUE
    """, [user_name, password]).to_pandas()

    if df.empty:
        return None
    return df.iloc[0]["APP_ROLE"].lower()

# -----------------------------------------------------------------------------
# LLM
# -----------------------------------------------------------------------------
def call_llm(prompt):
    sql = "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER"
    result = session.sql(sql, params=[MODEL_NAME, prompt]).collect()
    return result[0]["ANSWER"]

# -----------------------------------------------------------------------------
# AUTO FETCH + JOIN ALL TABLES WITH PATIENT_ID
# -----------------------------------------------------------------------------
def fetch_joined_tables():

    tables_df = session.sql("""
        SHOW TABLES IN SCHEMA AI_POC_DB.SP_PII_PHI
    """).to_pandas()

    table_names = tables_df["name"].tolist()

    valid_tables = []

    for table in table_names:
        cols = session.sql(f"""
            SHOW COLUMNS IN TABLE AI_POC_DB.SP_PII_PHI.{table}
        """).to_pandas()

        col_list = [c.upper() for c in cols["column_name"].tolist()]

        if "PATIENT_ID" in col_list:
            valid_tables.append(table)

    if not valid_tables:
        return pd.DataFrame()

    # Build Dynamic Join Query
    base_table = valid_tables[0]
    join_query = f"SELECT * FROM AI_POC_DB.SP_PII_PHI.{base_table} t0 "

    for idx, table in enumerate(valid_tables[1:], start=1):
        join_query += f"""
            LEFT JOIN AI_POC_DB.SP_PII_PHI.{table} t{idx}
            ON t0.PATIENT_ID = t{idx}.PATIENT_ID
        """

    return session.sql(join_query).to_pandas()

# -----------------------------------------------------------------------------
# Extract Entities from Joined Data
# -----------------------------------------------------------------------------
def extract_entities(entity_type):

    df = fetch_joined_tables()
    if df.empty:
        return []

    text_blob = df.astype(str).agg(" ".join, axis=1).str.cat(sep="\n")

    prompt = f"""
Extract unique {entity_type} names from the text.
Only return names where complete detailed information exists.
Return comma separated list only.

Text:
{text_blob}
"""

    response = call_llm(prompt)
    names = [x.strip() for x in response.split(",") if len(x.strip()) > 2]
    return list(dict.fromkeys(names))  # remove duplicates, keep order

# -----------------------------------------------------------------------------
# Get Full Details from Joined Tables
# -----------------------------------------------------------------------------
def get_full_details(name, entity_type):

    df = fetch_joined_tables()
    if df.empty:
        return ""

    text_blob = df.astype(str).agg(" ".join, axis=1).str.cat(sep="\n")

    prompt = f"""
Provide complete detailed information about {entity_type} named {name}.
If insufficient data, return NOTHING.
Use only provided text.

Text:
{text_blob}
"""
    return call_llm(prompt)

# -----------------------------------------------------------------------------
# LOGIN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:

    st.title("🔐 Chatbot Login")

    with st.form("login_form"):
        login_user = st.text_input("Username", placeholder="e.g. Vedant")
        login_password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:
        role = authenticate_user(login_user, login_password)
        if not role:
            st.error("Invalid credentials")
            st.stop()

        st.session_state.authenticated = True
        st.session_state.username = login_user
        st.session_state.app_role = role
        st.rerun()

    st.stop()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.success("Authenticated")
st.sidebar.write("User:", st.session_state.username)
st.sidebar.write("Role:", st.session_state.app_role.upper())

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# ADMIN / OWNER ENTITY VIEW WITH "+ MORE"
# -----------------------------------------------------------------------------
if st.session_state.app_role in ["admin", "owner"]:

    st.sidebar.markdown("---")

    category = st.sidebar.radio(
        "Select Category",
        ["Patients Details", "Doctor Details"]
    )

    entity_type = "patient" if category == "Patients Details" else "doctor"
    names = extract_entities(entity_type)

    start = st.session_state.entity_offset
    end = start + PAGE_SIZE
    visible_names = names[start:end]

    for name in visible_names:
        if st.sidebar.button(name, key=f"{entity_type}_{name}"):

            details = get_full_details(name, entity_type)
            if details.strip():
                st.session_state.messages = []
                st.session_state.messages.append(
                    {"role": "user", "content": f"Show complete details of {entity_type} {name}"}
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": details}
                )

    if end < len(names):
        if st.sidebar.button("+ More"):
            st.session_state.entity_offset += PAGE_SIZE
            st.rerun()

# -----------------------------------------------------------------------------
# MAIN CHAT
# -----------------------------------------------------------------------------
st.title("📄 PDF Chatbot on Snowflake")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask about your PDFs")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            joined_df = fetch_joined_tables()

            if joined_df.empty:
                answer = "Information not found in documents."
            else:
                text_blob = joined_df.astype(str).agg(" ".join, axis=1).str.cat(sep="\n")

                full_prompt = f"""
Use only the following data.
If answer not present, say:
"Information not found in documents."

Data:
{text_blob}

Question:
{prompt}

Answer:
"""
                answer = call_llm(full_prompt)

            st.write(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
