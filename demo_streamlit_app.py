import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# App Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

session = get_active_session()

SIMILARITY_THRESHOLD = 0.70
MODEL_NAME = "llama3.1-70b"

# -----------------------------------------------------------------------------
# Session State Init
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your PDFs."}
    ]

# -----------------------------------------------------------------------------
# LLM Call
# -----------------------------------------------------------------------------
def call_llm(prompt):
    sql = """SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER"""
    row = session.sql(sql, params=[MODEL_NAME, prompt]).collect()[0]
    return row["ANSWER"]

# -----------------------------------------------------------------------------
# Vector Search for Entity
# -----------------------------------------------------------------------------
def search_entity_context(query):

    sql = f"""
        WITH query_vec AS (
            SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                'snowflake-arctic-embed-m',
                ?
            ) AS emb
        )
        SELECT
            CHUNK_TEXT,
            VECTOR_COSINE_SIMILARITY(EMBEDDING, q.emb) AS SCORE
        FROM DOCS_CHUNKS, query_vec q
        ORDER BY SCORE DESC
        LIMIT 5
    """

    df = session.sql(sql, params=[query]).to_pandas()

    if df.empty:
        return None, 0

    best_score = df.iloc[0]["SCORE"]

    if best_score < SIMILARITY_THRESHOLD:
        return None, best_score

    context = "\n\n---\n\n".join(df["CHUNK_TEXT"].tolist())

    return context, best_score

# -----------------------------------------------------------------------------
# Extract and Filter Valid Entities
# -----------------------------------------------------------------------------
def get_valid_entities(entity_type):

    # Step 1: Extract names using LLM
    all_chunks = session.sql("SELECT CHUNK_TEXT FROM DOCS_CHUNKS").to_pandas()
    full_text = "\n\n".join(all_chunks["CHUNK_TEXT"].tolist())

    prompt = f"""
Extract ALL unique {entity_type} names from text below.
Return ONLY comma separated list.
No explanation.

Text:
{full_text}

Output:
"""

    response = call_llm(prompt)

    candidates = [x.strip() for x in response.split(",") if x.strip()]

    valid_entities = []

    # Step 2: Validate using Vector Similarity
    for name in candidates:
        _, score = search_entity_context(name)
        if score >= SIMILARITY_THRESHOLD:
            valid_entities.append(name)

    return sorted(list(set(valid_entities)))

# -----------------------------------------------------------------------------
# Generate Entity Details
# -----------------------------------------------------------------------------
def generate_entity_details(name, entity_type):

    context, score = search_entity_context(name)

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
# MAIN APP
# -----------------------------------------------------------------------------
st.sidebar.success("Authenticated")

# -----------------------------------------------------------------------------
# ADMIN SIDEBAR ENTITY PANEL
# -----------------------------------------------------------------------------
if st.session_state.get("app_role") in ["admin", "owner"]:

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

    # Show buttons instead of dropdown
    if "entities" in st.session_state:

        st.sidebar.markdown("### Available Entities")

        for name in st.session_state["entities"]:

            if st.sidebar.button(name):

                answer = generate_entity_details(
                    name,
                    st.session_state["entity_category"]
                )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

# -----------------------------------------------------------------------------
# Main Chat Window
# -----------------------------------------------------------------------------
st.title("📄 PDF Chatbot")

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

            context, score = search_entity_context(user_input)

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
