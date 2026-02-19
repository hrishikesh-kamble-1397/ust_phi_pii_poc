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
try:
    session = get_active_session()
except Exception as e:
    st.error(f"Snowflake session error: {str(e)}")
    st.stop()

MODEL_NAME = "llama3.1-70b"
EMBED_MODEL = "snowflake-arctic-embed-m"
SIMILARITY_THRESHOLD = 0.70
TOP_K = 5

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

    try:
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

    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return None

# -----------------------------------------------------------------------------
# LLM Call
# -----------------------------------------------------------------------------
def call_llm(prompt):

    try:
        sql = """SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER"""
        row = session.sql(sql, params=[MODEL_NAME, prompt]).collect()[0]
        return row["ANSWER"]

    except Exception as e:
        return f"LLM error: {str(e)}"

# -----------------------------------------------------------------------------
# Vector Search
# -----------------------------------------------------------------------------
def vector_search(query):

    try:
        sql = f"""
            WITH query_vec AS (
                SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                    '{EMBED_MODEL}',
                    ?
                ) AS emb
            )
            SELECT
                CHUNK_TEXT,
                SOURCE_FILE,
                VECTOR_COSINE_SIMILARITY(EMBEDDING, q.emb) AS SCORE
            FROM DOCS_CHUNKS, query_vec q
            ORDER BY SCORE DESC
            LIMIT {TOP_K}
        """

        df = session.sql(sql, params=[query]).to_pandas()

        if df.empty:
            return None, 0

        best_score = df.iloc[0]["SCORE"]

        if best_score < SIMILARITY_THRESHOLD:
            return None, best_score

        context = "\n\n---\n\n".join(df["CHUNK_TEXT"].tolist())

        return context, best_score

    except Exception as e:
        st.error(f"Vector search error: {str(e)}")
        return None, 0

# -----------------------------------------------------------------------------
# Extract Candidate Entities
# -----------------------------------------------------------------------------
def extract_entities(entity_type):

    try:
        chunks = session.sql("SELECT CHUNK_TEXT FROM DOCS_CHUNKS").to_pandas()

        if chunks.empty:
            return []

        full_text = "\n\n".join(chunks["CHUNK_TEXT"].tolist())

        prompt = f"""
Extract ALL unique {entity_type} names from text below.
Return ONLY comma-separated list.
No explanation.

Text:
{full_text}

Output:
"""

        response = call_llm(prompt)

        candidates = [x.strip() for x in response.split(",") if x.strip()]

        return sorted(list(set(candidates)))

    except Exception as e:
        st.error(f"Entity extraction error: {str(e)}")
        return []

# -----------------------------------------------------------------------------
# Validate Entities with Vector Similarity
# -----------------------------------------------------------------------------
def get_valid_entities(entity_type):

    valid_entities = []
    candidates = extract_entities(entity_type)

    for name in candidates:
        _, score = vector_search(name)
        if score >= SIMILARITY_THRESHOLD:
            valid_entities.append(name)

    return valid_entities

# -----------------------------------------------------------------------------
# Generate Entity Details
# -----------------------------------------------------------------------------
def generate_entity_details(name, entity_type):

    context, score = vector_search(name)

    if not context:
        return f"No strong contextual match found for {name}."

    prompt = f"""
You are a medical document assistant.

Provide complete details about {entity_type} "{name}".
Use ONLY context below.
Do not hallucinate.

Context:
{context}

Answer:
"""

    return call_llm(prompt)

# -----------------------------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:

    st.title("🔐 Chatbot Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:

        if not username.strip() or not password.strip():
            st.warning("Please enter username and password.")
            st.stop()

        role = authenticate_user(username, password)

        if not role:
            st.error("Invalid username or password.")
            st.stop()

        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.app_role = role
        st.rerun()

    st.stop()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.success("Authenticated")
st.sidebar.write("👤 User:", st.session_state.username)
st.sidebar.write("🛡️ Role:", st.session_state.app_role.upper())

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# ADMIN / OWNER ENTITY EXPLORER
# -----------------------------------------------------------------------------
if st.session_state.app_role in ["admin", "owner"]:

    st.sidebar.markdown("## 🔎 Data Explorer")

    category = st.sidebar.radio(
        "Select Category",
        ["Patient", "Doctor", "Hospital"]
    )

    if st.sidebar.button("Load Entities"):

        with st.sidebar.spinner("Analyzing documents..."):

            entities = get_valid_entities(category)

            if entities:
                st.session_state["entities"] = entities
                st.session_state["entity_category"] = category
            else:
                st.sidebar.warning("No high-confidence entities found.")

    if "entities" in st.session_state:

        st.sidebar.markdown("### Available Entities")

        for name in st.session_state["entities"]:

            if st.sidebar.button(name, key=f"{category}_{name}"):

                answer = generate_entity_details(
                    name,
                    st.session_state["entity_category"]
                )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

# -----------------------------------------------------------------------------
# MAIN CHAT WINDOW
# -----------------------------------------------------------------------------
st.title("📄 PDF Chatbot on Snowflake")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about your PDFs")

if user_input:

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            context, score = vector_search(user_input)

            if not context:
                answer = "No relevant content found."
            else:
                prompt = f"""
Answer question using ONLY context below.
Do not hallucinate.

Context:
{context}

Question:
{user_input}

Answer:
"""
                answer = call_llm(prompt)

            st.write(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
