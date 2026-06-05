import streamlit as st

from main import (
    PDF_DIR,
    INDEX_DIR,
    load_pdfs,
    split_documents,
    build_vectorstore,
    load_vectorstore,
    build_rag_chain,
)

st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("RAG Q&A — Chat with your PDFs")


@st.cache_resource
def init_chain():
    if not INDEX_DIR.exists():
        return None
    vs = load_vectorstore()
    return build_rag_chain(vs)


if "chain" not in st.session_state:
    st.session_state.chain = init_chain()
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Ingestion")
    if st.button("Re-index PDFs"):
        with st.spinner("Loading and indexing PDFs..."):
            docs = load_pdfs(PDF_DIR)
            chunks = split_documents(docs)
            build_vectorstore(chunks)
        st.session_state.chain = init_chain()
        st.success("Index updated!")
        st.rerun()

    st.divider()
    st.markdown(f"**PDF directory:** `{PDF_DIR}`")
    st.markdown(f"**Index:** `{INDEX_DIR.relative_to(INDEX_DIR.parent.parent)}`")
    st.markdown(f"**Model:** `llama3.2 (Ollama)`")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your PDFs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chain = st.session_state.chain
    if chain is None:
        with st.chat_message("assistant"):
            st.error("No index found. Click **Re-index PDFs** in the sidebar first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke({"input": prompt})
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
