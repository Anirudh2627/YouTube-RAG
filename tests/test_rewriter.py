from app.generation.base import Message
from app.generation.rewriter import QueryRewriter, needs_rewrite
from app.models.schemas import ConversationTurn


def hist():
    return [
        ConversationTurn(role="user", content="What is the attention mechanism?"),
        ConversationTurn(role="assistant", content="Attention lets tokens interact..."),
    ]


def test_needs_rewrite_detection():
    assert needs_rewrite("Why is it useful?", hist())
    assert needs_rewrite("how does that work", hist())
    assert not needs_rewrite("What learning rate is recommended for BERT fine-tuning?", hist())
    assert not needs_rewrite("Why is it useful?", [])       # no history


def test_heuristic_rewrite_includes_topic():
    rw = QueryRewriter(llm=None, enabled=True)
    out, method = rw.rewrite("Why is it useful?", hist())
    assert method == "heuristic"
    assert "attention mechanism" in out.lower()


def test_passthrough_when_standalone():
    rw = QueryRewriter(llm=None, enabled=True)
    q = "What learning rate does the speaker recommend?"
    out, method = rw.rewrite(q, hist())
    assert out == q and method is None


def test_disabled_rewriter():
    rw = QueryRewriter(llm=None, enabled=False)
    out, method = rw.rewrite("Why is it useful?", hist())
    assert out == "Why is it useful?" and method is None


class FakeLLM:
    model = "fake"
    name = "fake"

    def __init__(self, response="Why is the attention mechanism useful?"):
        self.response = response
        self.calls = []

    def complete(self, messages, temperature=None, max_tokens=None):
        self.calls.append(messages)
        return self.response


def test_llm_rewrite_used_when_available():
    llm = FakeLLM()
    rw = QueryRewriter(llm=llm, enabled=True)
    out, method = rw.rewrite("Why is it useful?", hist())
    assert method == "llm"
    assert out == "Why is the attention mechanism useful?"
    # prompt contains the history and the new question
    joined = str(llm.calls[0])
    assert "attention mechanism" in joined and "Why is it useful?" in joined


def test_llm_rewrite_failure_falls_back():
    class BrokenLLM(FakeLLM):
        def complete(self, *a, **k):
            raise RuntimeError("boom")
    rw = QueryRewriter(llm=BrokenLLM(), enabled=True)
    out, method = rw.rewrite("Why is it useful?", hist())
    assert method == "heuristic"


def test_llm_rewrite_garbage_guard():
    llm = FakeLLM(response="x")  # too short → guard rejects
    rw = QueryRewriter(llm=llm, enabled=True)
    out, method = rw.rewrite("Why is it useful?", hist())
    assert out == "Why is it useful?"   # passthrough, method still 'llm'


def test_message_helpers():
    assert Message.user("hi") == {"role": "user", "content": "hi"}
    assert Message.system("s")["role"] == "system"
