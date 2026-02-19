import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session
#--Update--
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

    sql = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER
    """

    row = session.sql(sql, params=[model_name, prompt]).collect()[0]
    return row["ANSWER"]

# -----------------------------------------------------------------------------
# Mask Final Answer (Only for Non-Admin)
# -----------------------------------------------------------------------------
def mask_answer_with_llm(answer_text):

    masking_prompt = f"""
You are a healthcare data privacy engine.

Mask ALL PII and PHI in the text below.

Rules:
- Replace sensitive values with exactly "XXXXXX"
- Keep text readable
- Do not explain anything
- Return only masked text

Text:
{answer_text}

Masked Output:
"""

    return call_llm("llama3.1-70b", masking_prompt)

# -----------------------------------------------------------------------------
# Generate Presigned URL
# -----------------------------------------------------------------------------
def get_presigned_url(file_name):

    sql = f"""
        SELECT GET_PRESIGNED_URL(
            @{STAGE_NAME},
            '{file_name}',
            3600
        ) AS URL
    """

    result = session.sql(sql).collect()
    return result[0]["URL"]

# -----------------------------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:

    st.title("🔐 Chatbot Login")

    with st.form("login_form"):
        login_user = st.text_input("Username",placeholder="e.g. Vedant")
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
SIMILARITY_THRESHOLD = 0.65  # adjust if needed

# -----------------------------------------------------------------------------
# Vector Search
# -----------------------------------------------------------------------------
def call_search(query, k):

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
        FROM AI_POC_DB.PII_PHI_POC.DOCS_CHUNKS c
        CROSS JOIN query_vec q
        ORDER BY SCORE DESC
        LIMIT {k}
    """

    return session.sql(search_sql, params=[query]).to_pandas()

# -----------------------------------------------------------------------------
# Render Chat History
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------------------------------------------------------
# Chat Input
# -----------------------------------------------------------------------------
prompt = st.chat_input("Type your question about the PDFs")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                chunks_df = call_search(prompt, top_k)

                if chunks_df.empty:
                    answer = "No relevant content found in documents."
                    st.write(answer)

                else:
                    # -----------------------------
                    # Generate Answer
                    # -----------------------------
                    context_text = "\n\n---\n\n".join(
                        chunks_df["CHUNK_TEXT"].tolist()
                    )

                    full_prompt = f"""
You are a medical document assistant.

Answer using ONLY the context below.
Be precise.
Do not hallucinate.

Context:
{context_text}

Question:
{prompt}

Answer:
"""

                    answer = call_llm(model, full_prompt)

                    # Mask for non-admin
                    if st.session_state.app_role not in ["admin", "owner"]:
                        answer = mask_answer_with_llm(answer)

                    st.write(answer)

                    # -----------------------------
                    # Admin: Show Most Relevant PDF
                    # -----------------------------
                    if st.session_state.app_role in ["admin", "owner"]:

                        # Calculate average similarity per file
                        file_scores = (
                            chunks_df
                            .groupby("SOURCE_FILE")["SCORE"]
                            .mean()
                            .reset_index()
                            .sort_values("SCORE", ascending=False)
                        )

                        best_file = file_scores.iloc[0]["SOURCE_FILE"]
                        best_score = file_scores.iloc[0]["SCORE"]

                        if best_score >= SIMILARITY_THRESHOLD:

                            st.markdown("### 📥 Most Relevant PDF")

                            url = get_presigned_url(best_file)

                            st.link_button(
                                f"Download {best_file} (Score: {best_score:.3f})",
                                url
                            )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                err_msg = f"Error: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err_msg}
                )
