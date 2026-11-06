from rag import Chunk, Retriever, split_pages, answer_question


def test_split_preserves_source_and_page():
    chunks = split_pages([Chunk("paper.pdf", 3, "alpha beta gamma " * 40)], chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.source == "paper.pdf" and c.page == 3 for c in chunks)


def test_retrieval_finds_relevant_passage():
    r = Retriever([Chunk("a.pdf", 1, "Neural networks classify images."), Chunk("b.pdf", 2, "Rivers flow into oceans.")])
    assert r.search("image classification neural networks")[0][0].source == "a.pdf"


def test_no_api_key_returns_guidance():
    assert "API key" in answer_question("What?", [(Chunk("a.pdf", 1, "Text"), 0.5)], "")


def test_empty_results():
    assert "couldn't find" in answer_question("What?", [], "")
