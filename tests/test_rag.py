# tests/test_rag.py

from rag.query_engine import RAGEngine

def test_rag():
    rag = RAGEngine(k=2)
    question = "Quel est le montant total ?"

    result = rag.query(question)

    assert "answer" in result
    assert "sources" in result

    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
