import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# App Configuration
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
    sql = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER
    """
    result = session.sql(sql, params=[MODEL_NAME, prompt]).collect()
    return result[0]["ANSWER"]

# -----------------------------------------------------------------------------
# Masking
# -----------------------------------------------------------------------------
def mask_answer(answer_text):
    masking_prompt = f"""
Mask ALL PII and PHI.
Replace sensitive values with: XXXXXX
Return only masked text.
Do NOT invent or add any new information. Only mask what's present.

Text:
{answer_text}
"""
    return call_llm(masking_prompt)

# -----------------------------------------------------------------------------
# Hybrid Search
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
                c.SOURCE_FILE,
                c.PAGE_NUM,
                VECTOR_COSINE_SIMILARITY(c.EMBEDDING, q.emb) AS SCORE
            FROM AI_POC_DB.PII_PHI_POC.DOCS_CHUNKS_NEW c
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
# Fetch All Chunks
# -----------------------------------------------------------------------------
def fetch_all_chunks():
    return session.sql("""
        SELECT CHUNK_TEXT
        FROM AI_POC_DB.PII_PHI_POC.DOCS_CHUNKS_NEW
    """).to_pandas()

# -----------------------------------------------------------------------------
# Extract Names
# -----------------------------------------------------------------------------
def extract_entities(entity_type):
    df = fetch_all_chunks()
    full_text = "\n".join(df["CHUNK_TEXT"].tolist())

    prompt = f"""
Extract unique {entity_type} names from the text.

Only return names where complete detailed information exists.
Return comma separated list only.
Do NOT invent any names. Only use what's present in the text.

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
    df = fetch_all_chunks()
    full_text = "\n".join(df["CHUNK_TEXT"].tolist())

    prompt = f"""
Provide complete detailed information about {entity_type} named {name}.
Use ONLY the provided text.
If insufficient data, return NOTHING.
Do NOT make up any information or hallucinate details.

Text:
{full_text}
"""

    return call_llm(prompt)

# -----------------------------------------------------------------------------
# LOGIN SCREEN
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
# ADMIN / OWNER ENTITY VIEW (ChatGPT Style)
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
# MAIN CHAT APPLICATION
# -----------------------------------------------------------------------------
st.title("📄 PDF Chatbot on Snowflake")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask about your PDFs")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            try:
                chunks_df = call_search(prompt)

                if chunks_df.empty:
                    answer = "Information not found in documents."
                else:
                    context_text = "\n\n".join(
                        [
                            f"[File: {row.SOURCE_FILE} | Page: {row.PAGE_NUM}]\n{row.CHUNK_TEXT}"
                            for _, row in chunks_df.iterrows()
                        ]
                    )

                    full_prompt = f"""
Use ONLY context provided below.
If answer cannot be found in context, reply exactly:
"Information not found in documents."
Do NOT invent or hallucinate any details.

Context:
{context_text}

Question:
{prompt}

Answer:
"""
                    answer = call_llm(full_prompt)

                    if st.session_state.app_role not in ["admin", "owner"]:
                        answer = mask_answer(answer)

                st.write(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
