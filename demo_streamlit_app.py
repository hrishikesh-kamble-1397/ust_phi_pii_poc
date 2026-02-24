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

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
MODEL_NAME = "mistral-large2"
EMBED_MODEL = "snowflake-arctic-embed-m"
SIMILARITY_THRESHOLD = 0.35
MAX_CHUNKS = 30

PDF_TABLE = "AI_POC_DB.PII_PHI_POC.DOCS_CHUNKS_NEW"
STRUCTURED_SCHEMA = "AI_POC_DB.SP_PII_PHI"

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
        {"role": "assistant", "content": "Ask me anything about your PDFs and structured data."}
    ]

# -----------------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------------
def authenticate_user(user_name, password):
    df = session.sql("""
        SELECT APP_ROLE
        FROM AI_POC_DB.PII_PHI_POC.APP_USER_ACCESS
        WHERE (
            UPPER(USER_NAME) = UPPER(:1)
            OR UPPER(USER_NAME) = SPLIT(UPPER(:1), '@')[0]
        )
        AND PASSWORD = :2
        AND IS_ACTIVE = TRUE
    """, [user_name, password]).to_pandas()

    if df.empty:
        return None

    return df.iloc[0]["APP_ROLE"].lower()

# -----------------------------------------------------------------------------
# LLM Call
# -----------------------------------------------------------------------------
def call_llm(prompt):
    sql = "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER"
    result = session.sql(sql, params=[MODEL_NAME, prompt]).collect()
    return result[0]["ANSWER"]

# -----------------------------------------------------------------------------
# Fetch PDF Chunks
# -----------------------------------------------------------------------------
def fetch_pdf_chunks():
    return session.sql(f"""
        SELECT CHUNK_TEXT
        FROM {PDF_TABLE}
    """).to_pandas()

# -----------------------------------------------------------------------------
# Fetch ALL Tables from Structured Schema
# -----------------------------------------------------------------------------
def fetch_structured_tables_text():

    tables = session.sql(f"""
        SHOW TABLES IN {STRUCTURED_SCHEMA}
    """).to_pandas()

    combined_text = ""

    for table in tables["name"]:
        full_table_name = f"{STRUCTURED_SCHEMA}.{table}"

        try:
            df = session.sql(f"SELECT * FROM {full_table_name}").to_pandas()
            combined_text += f"\n\nTable: {table}\n"
            combined_text += df.to_string(index=False)
        except:
            continue

    return combined_text

# -----------------------------------------------------------------------------
# Combined Context
# -----------------------------------------------------------------------------
def fetch_combined_context():
    pdf_df = fetch_pdf_chunks()
    pdf_text = "\n".join(pdf_df["CHUNK_TEXT"].tolist())

    structured_text = fetch_structured_tables_text()

    return pdf_text + "\n\n" + structured_text

# -----------------------------------------------------------------------------
# Extract Entities
# -----------------------------------------------------------------------------
def extract_entities(entity_type):

    full_text = fetch_combined_context()

    prompt = f"""
Extract unique {entity_type} names.

Only return names where complete detailed information exists.
Return comma separated list only.

Text:
{full_text}
"""

    response = call_llm(prompt)
    names = [x.strip() for x in response.split(",") if len(x.strip()) > 2]
    return list(set(names))

# -----------------------------------------------------------------------------
# Get Full Details
# -----------------------------------------------------------------------------
def get_full_details(name, entity_type):

    full_text = fetch_combined_context()

    prompt = f"""
Provide complete detailed information about {entity_type} named {name}.
Use only provided text.
If insufficient data, return NOTHING.

Text:
{full_text}
"""

    return call_llm(prompt)

# -----------------------------------------------------------------------------
# Hybrid Search (PDF only for vector search)
# -----------------------------------------------------------------------------
def call_search(query):

    search_sql = f"""
        WITH query_vec AS (
            SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                '{EMBED_MODEL}',
                ?
            ) AS emb
        ),
        vector_results AS (
            SELECT
                c.CHUNK_TEXT,
                VECTOR_COSINE_SIMILARITY(c.EMBEDDING, q.emb) AS SCORE
            FROM {PDF_TABLE} c
            CROSS JOIN query_vec q
            WHERE c.EMBEDDING IS NOT NULL
        )
        SELECT *
        FROM vector_results
        WHERE SCORE >= {SIMILARITY_THRESHOLD}
        ORDER BY SCORE DESC
        LIMIT {MAX_CHUNKS}
    """

    return session.sql(search_sql, params=[query]).to_pandas()

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
# ADMIN / OWNER ENTITY VIEW
# -----------------------------------------------------------------------------
if st.session_state.app_role in ["admin", "owner"]:

    st.sidebar.markdown("---")
    sidebar_tab = st.sidebar.radio(
        "Select Category",
        ["Patients Details", "Doctor Details"]
    )

    if sidebar_tab == "Patients Details":
        patient_names = extract_entities("patient")

        for name in patient_names:
            if st.sidebar.button(name, key=f"patient_{name}"):

                details = get_full_details(name, "patient")

                if details.strip():
                    st.session_state.messages = []
                    st.session_state.messages.append(
                        {"role": "user", "content": f"Show complete details of patient {name}"}
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": details}
                    )

    if sidebar_tab == "Doctor Details":
        doctor_names = extract_entities("doctor")

        for name in doctor_names:
            if st.sidebar.button(name, key=f"doctor_{name}"):

                details = get_full_details(name, "doctor")

                if details.strip():
                    st.session_state.messages = []
                    st.session_state.messages.append(
                        {"role": "user", "content": f"Show complete details of doctor {name}"}
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": details}
                    )

# -----------------------------------------------------------------------------
# MAIN CHAT
# -----------------------------------------------------------------------------
st.title("📄 PDF + Structured Data Chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask about PDFs or structured tables")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):

        full_text = fetch_combined_context()

        full_prompt = f"""
Use ONLY provided data.
If answer not present, say:
"Information not found in documents."

Data:
{full_text}

Question:
{prompt}

Answer:
"""

        answer = call_llm(full_prompt)
        st.write(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
