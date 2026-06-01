"""
RAG pipeline for HerRight.
Loads documents, builds FAISS vector store, and returns a retrieval chain.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

DOCS_PATH = "data/docs"
VECTOR_STORE_PATH = "data/vectorstore"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_embeddings = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache embedding model — runs once per session."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings

def load_documents(docs_path: str) -> list:
    """Load all PDFs and text files from the docs directory."""
    documents = []
    for filename in os.listdir(docs_path):
        filepath = os.path.join(docs_path, filename)
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            continue
        documents.extend(loader.load())
    print(f"Loaded {len(documents)} document chunks from {docs_path}")
    return documents


def build_vector_store(documents: list) -> FAISS:
    """Split documents and build FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,   # legal text needs more context per chunk
        chunk_overlap=100  # more overlap so clauses don't get cut off
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    vector_store = FAISS.from_documents(chunks, get_embeddings())
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"Vector store saved to {VECTOR_STORE_PATH}")
    return vector_store


def load_vector_store() -> FAISS:
    """Load existing FAISS vector store from disk."""
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )


def build_rag_chain(vector_store: FAISS):
    """Build and return the retrieval chain using Gemini."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are HerRight, a compassionate assistant helping women in India "
         "understand their legal rights, safety options, and support services. "
         "Use only the context below to answer. "
         "If the answer is not in the context, say: I dont have specific information on this, "
         "but you can call the Women Helpline at 181 for immediate support. "
         "Always respond in the same language the user writes in. "
         "Keep answers clear, simple, and actionable.\n\nContext:\n{context}"
        ),
        ("human", "{input}")
    ])

    combine_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": 4}),
        combine_chain
    )
    return retrieval_chain


def get_rag_chain():
    """
    Main entry point.
    Builds vector store if it doesn't exist, then returns the RAG chain.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        print("Building vector store for the first time...")
        documents = load_documents(DOCS_PATH)
        vector_store = build_vector_store(documents)
    else:
        print("Loading existing vector store...")
        vector_store = load_vector_store()

    return build_rag_chain(vector_store)