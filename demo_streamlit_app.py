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
STAGE_NAME = "AI_POC_DB.PII_PHI_POC.PHI_PII_POC_STAGE1"

# -----------------------------------------------------------------------------
# Session State Initialization
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
def call_llm(model_name, prompt):
    sql = """SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER"""
    row = session.sql(sql, params=[model_name, prompt]).collect()[0]
    return row["ANSWER"]

# -----------------------------------------------------------------------------
# Extract All Entity Names Using LLM
# -----------------------------------------------------------------------------
def extract_entities(entity_type):

    prompt = f"""
You are a medical document analyzer.

Extract ALL unique {entity_type} names across the provided text.
Return ONLY a clean comma-separated list.
No explanation.

Text:
(Use ALL document chunks from DOCS_CHUNKS table)

Output:
"""

    sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'llama3.1-70b',
            CONCAT(
                '{prompt}',
                LISTAGG(CHUNK_TEXT, '\n\n') 
                FROM DOCS_CHUNKS
            )
        ) AS ANSWER
    """

    # Simpler approach (safe)
    all_chunks = session.sql("SELECT CHUNK_TEXT FROM DOCS_CHUNKS").to_pandas()
    full_text = "\n\n".join(all_chunks["CHUNK_TEXT"].tolist())

    final_prompt = f"""
You are a medical document analyzer.

Extract ALL unique {entity_type} names across the text below.
Return ONLY comma-separated list.
No explanation.

Text:
{full_text}

Output:
"""

    response = call_llm("llama3.1-70b", final_prompt)

    entities = [e.strip() for e in response.split(",") if e.strip()]
    return sorted(list(set(entities)))

# -----------------------------------------------------------------------------
# Generate Answer For Selected Entity
# -----------------------------------------------------------------------------
def generate_entity_answer(name, entity_type):

    context_df = session.sql("SELECT CHUNK_TEXT FROM DOCS_CHUNKS").to_pandas()
    context_text = "\n\n---\n\n".join(context_df["CHUNK_TEXT"].tolist())

    prompt = f"""
You are a medical document assistant.

Provide complete details about {entity_type} "{name}".
Use ONLY the context below.
Do not hallucinate.

Context:
{context_text}

Answer:
"""

    return call_llm("llama3.1-70b", prompt)

# -----------------------------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:

    st.title("🔐 Chatbot Login")

    with st.form("login_form"):
        login_user = st.text_input("Username")
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
# ADMIN / OWNER TABS
# -----------------------------------------------------------------------------
if st.session_state.app_role in ["admin", "owner"]:

    st.sidebar.markdown("## 🔍 Data Explorer")

    main_tab = st.sidebar.radio(
        "Select Category",
        ["Patient Details", "Doctor Details", "Hospital Details"]
    )

    if main_tab == "Patient Details":
        entities = extract_entities("patient")

    elif main_tab == "Doctor Details":
        entities = extract_entities("doctor")

    else:
        entities = extract_entities("hospital")

    selected_entity = st.sidebar.selectbox(
        f"Select {main_tab[:-8]}",
        ["-- Select --"] + entities
    )

    if selected_entity != "-- Select --":

        answer = generate_entity_answer(selected_entity, main_tab[:-8])

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

# -----------------------------------------------------------------------------
# Main Chat Window
# -----------------------------------------------------------------------------
st.title("📄 PDF Chatbot on Snowflake")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask a question about the PDFs")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            context_df = session.sql("SELECT CHUNK_TEXT FROM DOCS_CHUNKS").to_pandas()
            context_text = "\n\n---\n\n".join(context_df["CHUNK_TEXT"].tolist())

            full_prompt = f"""
Answer the question using ONLY context below.
Do not hallucinate.

Context:
{context_text}

Question:
{prompt}

Answer:
"""

            answer = call_llm("llama3.1-70b", full_prompt)

            st.write(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
