import re
import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

session = get_active_session()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 PDF Chatbot on Snowflake")

# Constants (adjust if needed)
DB = "AI_POC_DB"
SCHEMA = "PII_PHI_POC"
STAGE = f"{DB}.{SCHEMA}.PHI_PII_POC_STAGE1"          # must be DIRECTORY = TRUE
PAGES_TBL = f"{DB}.{SCHEMA}.PDF_PAGES"
CHUNKS_TBL = f"{DB}.{SCHEMA}.DOCS_CHUNKS"
SEARCH_SVC = "pdf_search_svc"
MASKING_POLICY = f"{DB}.{SCHEMA}.PII_PHI_MASK"
ROW_POLICY     = f"{DB}.{SCHEMA}.PII_PHI_ROW_POLICY"
DEFAULT_WAREHOUSE = st.secrets.get("snowflake", {}).get("ANURAG_WH", None)

PRIV_ROLE = "PHI_FULL_ACCESS"   # privileged role name
PII_ROLE  = "PII_VIEWER"        # mid-tier role (optional, adjust if unused)

# Sidebar controls
st.sidebar.header("Settings")
model = st.sidebar.selectbox("LLM model", ["llama3.1-70b", "mistral-large", "mixtral-8x7b"], index=0)
top_k = st.sidebar.slider("Top‑K chunks", 1, 12, 5, 1)
show_sources = st.sidebar.checkbox("Show retrieved sources", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("PII/PHI Stored Procedure")
stage_file_name = st.sidebar.text_input(
    "Stage file name (must exist on your PDF stage)",
    value="sample_hospitalization_claim1.pdf",
)
run_proc_btn = st.sidebar.button("Run PII/PHI Parse & Classify")

# Helpers
def run_sql(sql: str, params=None):
    return session.sql(sql, params=params).collect()

def run_sql_df(sql: str, params=None) -> pd.DataFrame:
    return session.sql(sql, params=params).to_pandas()

def call_search(query: str, k: int) -> pd.DataFrame:
    sql = f"""
        SELECT
          CHUNK_TEXT,
          SOURCE_FILE,
          SCORE
        FROM SNOWFLAKE.CORTEX.SEARCH(
          SERVICE => '{SEARCH_SVC}',
          QUERY   => %s,
          TOP_K   => {k}
        )
        ORDER BY SCORE DESC
    """
    return session.sql(sql, params=[query]).to_pandas()

def call_llm(model_name: str, prompt: str) -> str:
    row = session.sql("SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS ANSWER", params=[model_name, prompt]).collect()[0]
    return row["ANSWER"]

def sanitize_suffix(file_name: str) -> str:
    base = re.sub(r"\.pdf$", "", file_name, flags=re.IGNORECASE)
    base = re.sub(r"[^A-Za-z0-9]+", "_", base)
    return base.upper()

def has_role(role: str) -> bool:
    """Check RBAC context using IS_ROLE_IN_SESSION (Snowflake context function)."""
    try:
        row = session.sql("SELECT IS_ROLE_IN_SESSION(%s) AS ok", params=[role]).collect()[0]
        return bool(row["OK"])
    except Exception:
        return False

def call_pii_phi_proc(file_name: str) -> str:
    row = session.sql("CALL AI_POC_DB.PII_PHI_POC.SP_PARSE_EXTRACT_CLASSIFY(%s)", params=[file_name]).collect()[0]
    return list(row.asDict().values())[0]  # first column contains return string

def extract_output_table_name(proc_return_msg: str) -> str | None:
    marker = "Output table = "
    i = proc_return_msg.find(marker)
    if i == -1:
        return None
    return proc_return_msg[i + len(marker):].strip()

def attach_policies_to_output_table(out_fqn: str):
    """Attach masking + row access policies. Masking uses USING (CLASSIFICATION)."""
    try:
        # Mask COLUMN_VALUE based on CLASSIFICATION and session roles
        session.sql(
            f"ALTER TABLE {out_fqn} MODIFY COLUMN COLUMN_VALUE "
            f"SET MASKING POLICY {MASKING_POLICY} USING (CLASSIFICATION)"
        ).collect()
    except Exception as e:
        st.warning(f"Could not attach masking policy to {out_fqn}: {e}")

    try:
        # Filter rows via row policy (hide PHI/PII for non-privileged depending on your logic)
        session.sql(
            f"ALTER TABLE {out_fqn} ADD ROW ACCESS POLICY {ROW_POLICY} ON (CLASSIFICATION)"
        ).collect()
    except Exception as e:
        # If policy already attached, harmless to skip
        if "already has a row access policy" not in str(e):
            st.warning(f"Could not attach row access policy to {out_fqn}: {e}")

def load_output_df(out_fqn: str) -> pd.DataFrame:
    # Querying directly picks up masking/row policies at runtime (enforced by Snowflake)
    return session.table(out_fqn).to_pandas()

def redact_text_for_unprivileged(text: str) -> str:
    """Lightweight client-side PII redaction for chat context (defense in depth)."""
    # emails
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
    # phone-like
    text = re.sub(r"\+?\d[\d\s().-]{7,}\d", "[REDACTED_PHONE]", text)
    # SSN-like (US example); adjust for locale as needed
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", text)
    # ID-like long sequences
    text = re.sub(r"\b\d{9,}\b", "[REDACTED_ID]", text)
    return text

# Tabs
tab_chat, tab_pii = st.tabs(["💬 PDF Chatbot", "🧪 PII/PHI Output"])

# Chat tab: RAG with RBAC‑aware redaction
with tab_chat:
    st.caption(
        "Retrieves relevant chunks via Cortex Search and answers with a model. "
        "If your role lacks PHI/PII clearance, we redact likely PII before the model sees it."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me anything about your PDFs."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    privileged = has_role(PRIV_ROLE) or has_role(PII_ROLE)

    if prompt := st.chat_input("Type your question about the PDFs"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chunks_df = call_search(prompt, top_k)
                    if chunks_df.empty:
                        answer = "I couldn't find anything relevant in the PDFs. Try rephrasing or broadening your query."
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        # Build grounded context
                        blocks = []
                        for _, row in chunks_df.iterrows():
                            src = row["SOURCE_FILE"]
                            txt = row["CHUNK_TEXT"]
                            score = row["SCORE"]
                            blocks.append(f"File: {src}\nScore: {score:.4f}\nContent:\n{txt}\n")
                        context_text = "\n\n---\n\n".join(blocks)

                        # RBAC-aware redaction before sending to LLM (defense-in-depth)
                        if not privileged:
                            context_text = redact_text_for_unprivileged(context_text)

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

                        answer = call_llm(model, full_prompt)
                        st.write(answer)

                        if show_sources:
                            with st.expander("Show retrieved sources"):
                                st.dataframe(
                                    chunks_df[["SOURCE_FILE", "SCORE", "CHUNK_TEXT"]],
                                    use_container_width=True, height=320
                                )

                        st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err = f"Something went wrong while answering: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

# PII/PHI tab: run stored procedure, then attach masking/row policies, then preview
with tab_pii:
    st.caption(
        "Runs the stored procedure to parse → extract → classify. "
        "Then auto‑attaches masking + row access policies so sensitive data obeys RBAC."
    )

    if run_proc_btn:
        with st.spinner("Running stored procedure..."):
            try:
                proc_msg = call_pii_phi_proc(stage_file_name)
                st.success(proc_msg)

                # Find output table
                out_tbl = extract_output_table_name(proc_msg)
                if not out_tbl:
                    # Fallback: reconstruct from suffix
                    suffix = sanitize_suffix(stage_file_name)
                    out_tbl = f"{DB}.{SCHEMA}.OUTPUT_{suffix}"

                # Attach policies
                attach_policies_to_output_table(out_tbl)

                # Load preview (Snowflake enforces masking/row filters when we SELECT)
                out_df = load_output_df(out_tbl)
                st.subheader("PII/PHI Output Preview (RBAC‑enforced)")
                st.dataframe(out_df, use_container_width=True, height=380)

                with st.expander("Summary counts"):
                    try:
                        summary = (
                            out_df["CLASSIFICATION"]
                            .value_counts(dropna=False)
                            .rename_axis("CLASSIFICATION")
                            .reset_index(name="COUNT")
                        )
                        st.table(summary)
                    except Exception:
                        st.write("Summary unavailable.")
            except Exception as e:
                st.error(f"Procedure failed: {e}")
