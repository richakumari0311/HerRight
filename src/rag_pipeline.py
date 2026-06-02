"""
RAG pipeline for HerRight.
Loads documents, builds FAISS vector store, and returns a history-aware retrieval chain.
Built with LCEL - no deprecated langchain.chains imports.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import BaseMessage

load_dotenv()

DOCS_PATH = "data/docs"
VECTOR_STORE_PATH = "data/vectorstore"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"Vector store saved to {VECTOR_STORE_PATH}")
    return vector_store


def load_vector_store() -> FAISS:
    """Load existing FAISS vector store from disk."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def format_docs(docs: list) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(messages: list[BaseMessage]) -> str:
    """Format chat history into a readable string for the prompt."""
    if not messages:
        return ""
    formatted = []
    for msg in messages:
        role = "User" if msg.type == "human" else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)


def build_rag_chain(vector_store: FAISS):
    """
    Build a history-aware RAG chain using pure LCEL.
    Returns a callable that accepts {input, chat_history} and returns {answer, context}.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Step 1 — rephrase user question considering chat history
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the conversation history and the latest user question, "
         "reformulate the question to be standalone and self-contained. "
         "Do not answer it. Just reformulate if needed, otherwise return as is. "
         "Chat history:\n{chat_history}"
        ),
        ("human", "{input}")
    ])

    rephrase_chain = rephrase_prompt | llm | StrOutputParser()

    # Step 2 — answer using retrieved context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are HerRight, a compassionate assistant helping women in India "
        "understand their legal rights, safety options, and support services. "
        "Use only the context below to answer. "
        "If the answer is not in the context, say: I don't have specific information "
        "on this, but you can call the Women Helpline at 181 for immediate support. "
        "Always respond in the same language the user writes in. "
        "If the user asks for audio or a different language, just respond in that language as text — "
        "audio is handled automatically by the app. "
        "Keep answers clear, simple, and actionable.\n\nContext:\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    def retrieve_with_history(inputs: dict) -> dict:
        """Rephrase question using history, then retrieve docs."""
        chat_history_str = format_chat_history(inputs.get("chat_history", []))
        rephrased = rephrase_chain.invoke({
            "input": inputs["input"],
            "chat_history": chat_history_str
        })
        docs = retriever.invoke(rephrased)
        return {
            "context": docs,
            "context_str": format_docs(docs),
            "input": inputs["input"],
            "chat_history": inputs.get("chat_history", [])
        }

    def generate_answer(inputs: dict) -> dict:
        """Generate answer from context and return with source docs."""
        answer = (qa_prompt | llm | StrOutputParser()).invoke({
            "context": inputs["context_str"],
            "input": inputs["input"],
            "chat_history": inputs["chat_history"]
        })
        return {
            "answer": answer,
            "context": inputs["context"]
        }

    return RunnableLambda(retrieve_with_history) | RunnableLambda(generate_answer)


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