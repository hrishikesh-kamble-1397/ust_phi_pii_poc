import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# App config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

# -----------------------------------------------------------------------------
# Snowflake Session
# -----------------------------------------------------------------------------
session = get_active_session()

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

if "app_role" not in st.session_state:
    st.session_state.app_role = None

# -----------------------------------------------------------------------------
# Authenticate User
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
# Generic LLM Call
# -----------------------------------------------------------------------------
def call_llm(model_name: str, prompt: str) -> str:

    sql = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER
    """

    row = session.sql(sql, params=[model_name, prompt]).collect()[0]
    return row["ANSWER"]

# -----------------------------------------------------------------------------
# Mask Final Answer Using LLM (NOT CHUNKS)
# -----------------------------------------------------------------------------
def mask_answer_with_llm(answer_text: str) -> str:

    masking_prompt = f"""
You are a healthcare privacy engine.

Mask ALL PII and PHI in the text below.

Rules:
- Replace sensitive values with exactly "XXXXXX"
- Keep the sentence readable
- Do NOT remove non-sensitive info
- Do NOT explain
- Return only masked text

Text:
{answer_text}

Masked Output:
"""

    return call_llm("llama3.1-70b", masking_prompt)

# -----------------------------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:

    st.title("🔐 Chatbot Login")
    st.caption("Authenticate to access PDF Chatbot")

    with st.form("login_form"):
        login_user = st.text_input("Username")
        login_password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:

        if not login_user.strip() or not login_password.strip():
            st.warning("Please enter username and password.")
            st.stop()

        role = authenticate_user(login_user, login_password)

        if not role:
            st.error("❌ Invalid username or password.")
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
st.sidebar.write("👤 User:", st.session_state.username)
st.sidebar.write("🛡️ App Role:", st.session_state.app_role.upper())

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
st.title("📄 PDF Chatbot on Snowflake")

top_k = 10
model = "llama3.1-70b"

# -----------------------------------------------------------------------------
# Vector Search
# -----------------------------------------------------------------------------
def call_search(query: str, k: int) -> pd.DataFrame:

    search_sql = f"""
        WITH query_vec AS (
            SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                'snowflake-arctic-embed-m',
                ?
            ) AS emb
        )
        SELECT
            c.CHUNK_TEXT,
            c.SOURCE_FILE,
            VECTOR_COSINE_SIMILARITY(c.EMBEDDING, q.emb) AS SCORE
        FROM DOCS_CHUNKS c
        CROSS JOIN query_vec q
        ORDER BY SCORE DESC
        LIMIT {k}
    """

    return session.sql(search_sql, params=[query]).to_pandas()

# -----------------------------------------------------------------------------
# Chat History
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your PDFs."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------------------------------------------------------
# Chat Input
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Type your question about the PDFs"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                chunks_df = call_search(prompt, top_k)

                if chunks_df.empty:
                    answer = "No relevant content found in documents."
                else:
                    context_text = "\n\n---\n\n".join(
                        chunks_df["CHUNK_TEXT"].tolist()
                    )

                    system_prompt = """
You are a medical document assistant.

Answer the question using ONLY the context below.
Be precise.
Do not hallucinate.
"""

                    full_prompt = f"""
{system_prompt}

Context:
{context_text}

Question:
{prompt}

Answer:
"""

                    # 1️⃣ Generate FULL Answer using RAW data
                    answer = call_llm(model, full_prompt)

                    # 2️⃣ If NOT admin → Mask final answer
                    if st.session_state.app_role not in ["admin", "owner"]:
                        answer = mask_answer_with_llm(answer)

                st.write(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                err_msg = f"Error: {e}"
                st.error(err_msg)
                st.session_state.messages.append(
