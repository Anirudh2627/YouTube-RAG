from app.retrieval.bm25 import BM25Index, tokenize


DOCS = [
    ("d1", "the transformer uses multi head attention with queries keys and values"),
    ("d2", "recurrent neural networks process sequences one token at a time"),
    ("d3", "attention allows every token to attend to every other token directly"),
    ("d4", "cooking pasta requires boiling water salt and tomato sauce"),
]


def build():
    idx = BM25Index()
    idx.build(DOCS)
    return idx


def test_tokenize_stopwords():
    toks = tokenize("What is the learning rate for BERT?")
    assert "the" not in toks and "is" not in toks
    assert "learning" in toks and "rate" in toks and "bert" in toks


def test_exact_term_query_ranks_relevant_first():
    idx = build()
    res = idx.search("pasta tomato sauce", top_k=4)
    assert res[0][0] == "d4"


def test_semantic_word_query():
    idx = build()
    res = idx.search("attention token", top_k=2)
    # d3 contains BOTH terms (twice each) → must rank first; the second slot
    # goes to a single-term doc (d1 or d2 depending on length-normalized tf)
    assert res[0][0] == "d3"
    assert res[1][0] in {"d1", "d2"}


def test_learning_rate_specific():
    idx = BM25Index()
    idx.build(DOCS + [("d5", "i recommend a learning rate of two times ten to the minus five")])
    res = idx.search("learning rate recommend", top_k=1)
    assert res[0][0] == "d5"


def test_scores_positive_and_sorted():
    idx = build()
    res = idx.search("attention", top_k=10)
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_no_match_returns_empty():
    idx = build()
    assert idx.search("zzz qqq", top_k=5) == []


def test_allowed_subset_filter():
    idx = build()
    res = idx.search("token attention", top_k=5, allowed={"d2"})
    assert all(cid == "d2" for cid, _ in res)


def test_len():
    assert len(build()) == 4
