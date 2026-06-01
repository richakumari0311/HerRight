"""Quick test to verify RAG pipeline end to end."""

import sys
sys.path.append(".")

from src.rag_pipeline import get_rag_chain

chain = get_rag_chain()
response = chain.invoke({"input": "What are my rights if my husband is violent?"})
print(response["answer"])