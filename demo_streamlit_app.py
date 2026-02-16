import re
import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# App config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 PDF Chatbot on Snowflake")

# Obtain active Snowflake session (works in Snowsight / Snowflake-hosted Streamlit)
session = get_active_session()

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Settings")
model = st.sidebar.selectbox(
    "LLM model",
    options=[
        # Keep the one(s) that exist in your account; adjust if needed
        "mistral-large",
        "llama3.1-70b",
        "mixtral-8x7b"
    ],
    index=0
)
top_k = st.sidebar.slider("Top‑K chunks", min_value=1, max_value=10, value=5, step=1)

show_sources = st.sidebar.checkbox("Show sources", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("PII/PHI Extractor")
default_stage_file = "sample_hospitalization_claim1.pdf"
stage_file_name = st.sidebar.text_input(
    "Stage file name (file must be uploaded in the stage)",
    value=default_stage_file,
    help="The file must already be uploaded to the specified stage."
)
run_proc = st.sidebar.button("Run PII/PHI Parse & Classify")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def call_search(query: str, k: int) -> pd.DataFrame:
    search_sql = """
       SELECT
          CHUNK_TEXT,
          SOURCE_FILE,
          SCORE
        FROM SNOWFLAKE.CORTEX.SEARCH(
          SERVICE => 'pdf_search_svc',
          QUERY   => ?,
          TOP_K   => ?
        )
        ORDER BY SCORE DESC
    """
    return session.sql(search_sql, params=[query, k]).to_pandas()

def call_llm(model_name: str, prompt: str) -> str:
    """
    Calls Snowflake Cortex COMPLETE to generate the answer.
    """
    llm_sql = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER
    """
    row = session.sql(llm_sql, params=[model_name, prompt]).collect()[0]
    return row["ANSWER"]

def sanitize_suffix(file_name: str) -> str:
    """
    Mirrors the stored procedure's suffix normalization:
    - remove .pdf
    - replace non-alphanumeric with underscores
    - uppercase
    """
    # We'll perform similar transform in SQL to avoid mismatches,
    # but provide a best-effort local preview.
    base = re.sub(r"\.pdf$", "", file_name, flags=re.IGNORECASE)
    base = re.sub(r"[^A-Za-z0-9]+", "_", base)
    return base.upper()

def call_pii_phi_proc(file_name: str) -> str:
    """
    Calls your stored procedure and returns the procedure's message.
    """
    sql = "CALL AI_POC_DB.PII_PHI_POC.SP_PARSE_EXTRACT_CLASSIFY(?)"
    row = session.sql(sql, params=[file_name]).collect()[0]
    return list(row.asDict().values())[0]

def load_output_table(file_name: str) -> pd.DataFrame:
    """
    Loads the output table produced by the procedure, reconstructing the FQN
    from the same suffix logic (safe if proc returns constant naming pattern).
    """
    suffix = sanitize_suffix(file_name)
    out_fqn = f"AI_POC_DB.PII_PHI_POC.OUTPUT_{suffix}"
    return session.table(out_fqn).to_pandas()

# -----------------------------------------------------------------------------
# Chat history
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your PDFs."}
    ]

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------------------------------------------------------
# Optional: run the PII/PHI procedure from the sidebar
# -----------------------------------------------------------------------------
if run_proc:
    with st.spinner("Running PII/PHI parse & classify stored procedure..."):
        try:
            proc_msg = call_pii_phi_proc(stage_file_name)
            st.sidebar.success(proc_msg)

            # Try to load and preview the output
            try:
                out_df = load_output_table(stage_file_name)
                st.subheader("PII/PHI Output Table Preview")
                st.dataframe(out_df, use_container_width=True, height=320)
            except Exception as e:
                st.warning(
                    f"Procedure succeeded, but could not load output table yet. "
                    f"If this persists, check naming and permissions.\n\nDetails: {e}"
                )
        except Exception as e:
            st.sidebar.error(f"Procedure failed: {e}")

# -----------------------------------------------------------------------------
# Chat input and response
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Type your question about the PDFs"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 1) Retrieve top-k relevant chunks
                chunks_df = call_search(prompt, top_k)

                if chunks_df.empty:
                    answer = (
                        "I couldn't find anything relevant in the PDFs. "
                        "Try rephrasing or broadening your query."
                    )
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    # Build context blocks
                    context_blocks = []
                    for _, row in chunks_df.iterrows():
                        src = row["SOURCE_FILE"]
                        txt = row["CHUNK_TEXT"]
                        score = row["SCORE"]
                        context_blocks.append(
                            f"File: {src}\nScore: {score:.4f}\nContent:\n{txt}\n"
                        )
                    context_text = "\n\n---\n\n".join(context_blocks)

                    system_prompt = (
                        "You are a helpful assistant that answers questions using only the provided PDF excerpts.\n"
                        "If the answer is not contained in the context, say you don’t know.\n"
                        "Cite the file names you used."
                    )

                    full_prompt = f"""{system_prompt}

Context:
{context_text}

Question: {prompt}
Answer:"""

                    # 2) Call LLM for the grounded answer
                    answer = call_llm(model, full_prompt)

                    # Output answer
                    st.write(answer)

                    # Optional: show sources used
                    if show_sources:
                        with st.expander("Show retrieved sources"):
                            st.dataframe(
                                chunks_df[["SOURCE_FILE", "SCORE", "CHUNK_TEXT"]],
                                use_container_width=True,
                                height=300
                            )

                    st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                err_msg = f"Something went wrong while answering: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
