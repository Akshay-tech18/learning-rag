from pathlib import Path
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


PDF_DIR = Path(__file__).parent / "data" / "pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
INDEX_DIR = Path(__file__).parent / "faiss_index"

SYSTEM_PROMPT = (
    "You are an assistant that answers questions based on the provided context. "
    "Use only the given context to answer. If the context doesn't contain enough "
    "information, say 'I cannot find this information in the provided documents.' "
    "Keep your answers concise and informative."
    "\n\nContext:\n{context}"
)


def load_pdfs(pdf_dir: Path) -> list:
    all_docs = []
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s)")

    for pdf_file in pdf_files:
        print(f"  Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = pdf_file.name
            doc.metadata["file_type"] = "pdf"
        all_docs.extend(docs)
        print(f"    -> {len(docs)} pages")

    print(f"Total documents: {len(all_docs)}")
    return all_docs


def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: list) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"FAISS index saved to {INDEX_DIR}")
    return vectorstore


def load_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_rag_chain(vectorstore: FAISS):
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return (
        {"context": itemgetter("input") | retriever | format_docs, "input": itemgetter("input")}
        | prompt
        | llm
        | StrOutputParser()
    )


def ingest():
    print("=== Ingestion ===")
    docs = load_pdfs(PDF_DIR)
    chunks = split_documents(docs)
    build_vectorstore(chunks)


def ask(chain, question: str):
    answer = chain.invoke({"input": question})
    print(f"\nAnswer: {answer}")


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
        return

    if not INDEX_DIR.exists():
        print("No index found. Run `python main.py ingest` first.")
        return

    print("Loading vectorstore...")
    vectorstore = load_vectorstore()
    chain = build_rag_chain(vectorstore)

    print("Ready! Type your questions (or 'quit' to exit).")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        ask(chain, question)


if __name__ == "__main__":
    main()
