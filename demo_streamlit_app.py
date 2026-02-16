import streamlit as st
from snowflake.snowpark.context import get_active_session

session = get_active_session()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot on Snowflake")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your PDFs."}
    ]

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if prompt := st.chat_input("Type your question about the PDFs"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Retrieval: query Cortex Search for relevant chunks
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1) Get top-k relevant chunks
            search_sql = """
                SELECT
                  chunk_text,
                  source_file,
                  SCORE
                FROM SNOWFLAKE.CORTEX.SEARCH(
                  SERVICE => 'pdf_search_svc',
                  QUERY   => ?,
                  TOP_K   => 5
                )
                ORDER BY SCORE DESC;
            """
            chunks_df = session.sql(search_sql, params=[prompt]).to_pandas()

            if chunks_df.empty:
                answer = "I couldn't find anything relevant in the PDFs."
            else:
                context_blocks = []
                for _, row in chunks_df.iterrows():
                    context_blocks.append(
                        f"File: {row['SOURCE_FILE']}\nContent:\n{row['CHUNK_TEXT']}\n"
                    )
                context_text = "\n\n---\n\n".join(context_blocks)

                system_prompt = """
You are a helpful assistant that answers questions using only the provided PDF excerpts.
If the answer is not contained in the context, say you don’t know.
Include which file(s) you used if possible.
"""

                full_prompt = f"""{system_prompt}

Context:
{context_text}

Question: {prompt}
Answer:"""

                # 2) Call Cortex LLM to answer
                llm_sql = """
                    SELECT SNOWFLAKE.CORTEX.COMPLETE(
                      'mistral-large',
                      %s
                    ) AS ANSWER;
                """
                answer_row = session.sql(llm_sql, params=[full_prompt]).collect()[0]
                answer = answer_row["ANSWER"]

            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
