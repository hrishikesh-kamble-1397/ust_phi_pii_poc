import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="PDF/Database Chatbot", page_icon="📄", layout="wide")

session = get_active_session()

MODEL_NAME = "mistral-large2"
EMBED_MODEL = "snowflake-arctic-embed-m"
SIMILARITY_THRESHOLD = 0.35
MAX_CHUNKS = 50

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "app_role" not in st.session_state:
    st.session_state.app_role = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your PDFs or database."}
    ]
if "mode" not in st.session_state:
    st.session_state.mode = "PDF"


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


def call_llm(prompt):
    sql = """ SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER """
    result = session.sql(sql, params=[MODEL_NAME, prompt]).collect()
    return result[0]["ANSWER"]


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


def fetch_all_chunks():
    return session.sql("""
        SELECT CHUNK_TEXT
        FROM AI_POC_DB.PII_PHI_POC.DOCS_CHUNKS_NEW
    """).to_pandas()


def get_db_rows(user_prompt: str):
    notes_col = "EHR_NOTES" if st.session_state.app_role in ["admin", "owner"] else "NOTES_REDACTED"
    safe_prompt = user_prompt.replace("'", "''")
    sql = f"""
        SELECT
            PATIENT_ID,
            PATIENT_NAME,
            PATIENT_ADDRESS,
            HP_DETAILS,
            {notes_col} AS NOTES
        FROM AI_POC_DB.PII_PHI_POC.POC_EHR_NOTES_PHI_REDACTED_OPT
        WHERE CONTAINS(LOWER({notes_col}), LOWER('{safe_prompt}'))
           OR CONTAINS(LOWER(HP_DETAILS), LOWER('{safe_prompt}'))
        LIMIT 50
    """
    return session.sql(sql).to_pandas()


def get_db_answer(user_prompt: str):
    rows_df = get_db_rows(user_prompt)
    if rows_df.empty:
        return "Information not found in database."

    context_lines = []
    for _, row in rows_df.iterrows():
        line = (
            f"PATIENT_ID: {row.PATIENT_ID} | "
            f"NAME: {row.PATIENT_NAME} | "
            f"ADDRESS: {row.PATIENT_ADDRESS} | "
            f"HP_DETAILS: {row.HP_DETAILS} | "
            f"NOTES: {row.NOTES}"
        )
        context_lines.append(line)

    context_text = "\n".join(context_lines)

    prompt = f"""
You are a clinical data assistant.
Use ONLY the patient records provided in the context below.
If the answer cannot be found in the context, reply exactly:
"Information not found in database."
Do NOT invent or hallucinate any details.

Context (each line is one row from the table):
{context_text}

User Question:
{user_prompt}

Answer:
"""
    answer = call_llm(prompt)
    if st.session_state.app_role not in ["admin", "owner"]:
        answer = mask_answer(answer)
    return answer


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

st.sidebar.success("Authenticated")
st.sidebar.write("User:", st.session_state.username)
st.sidebar.write("Role:", st.session_state.app_role.upper())
st.sidebar.markdown("---")
st.session_state.mode = st.sidebar.radio("Select Mode", ["PDF", "Database"])
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

st.title("📄 PDF / Database Chatbot on Snowflake")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask a question")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                if st.session_state.mode == "PDF":
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

                elif st.session_state.mode == "Database":
                    answer = get_db_answer(prompt)
                else:
                    answer = "Invalid mode selected."

                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
