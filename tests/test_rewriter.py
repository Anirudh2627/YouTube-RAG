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
    assert not needs_rewrite(
        "What learning rate is recommended for BERT fine-tuning?",
        hist(),
    )
    assert not needs_rewrite("Why is it useful?", [])


def test_heuristic_rewrite_includes_topic():
    rw = QueryRewriter(enabled=True)

    out, method = rw.rewrite("Why is it useful?", hist())

    assert method == "heuristic"
    assert "attention mechanism" in out.lower()


def test_passthrough_when_standalone():
    rw = QueryRewriter(enabled=True)

    q = "What learning rate does the speaker recommend?"

    out, method = rw.rewrite(q, hist())

    assert out == q
    assert method is None


def test_disabled_rewriter():
    rw = QueryRewriter(enabled=False)

    out, method = rw.rewrite("Why is it useful?", hist())

    assert out == "Why is it useful?"
    assert method is None


def test_heuristic_rewrite_uses_recent_history():
    history = [
        ConversationTurn(role="user", content="What is gradient descent?"),
        ConversationTurn(role="assistant", content="It is an optimization algorithm."),
        ConversationTurn(role="user", content="What is a learning rate?"),
        ConversationTurn(role="assistant", content="It controls the step size."),
    ]

    rw = QueryRewriter(enabled=True, max_history_turns=4)

    out, method = rw.rewrite("Why is it important?", history)

    assert method == "heuristic"
    assert "learning rate" in out.lower()


def test_heuristic_rewrite_truncates_long_topic():
    long_question = (
        "Can you explain how gradient descent works in neural networks "
        "including learning rates momentum regularization and convergence?"
    )

    history = [
        ConversationTurn(role="user", content=long_question),
        ConversationTurn(role="assistant", content="Explanation..."),
    ]

    rw = QueryRewriter(enabled=True)

    out, method = rw.rewrite("Why?", history)

    assert method == "heuristic"

    topic = long_question.rstrip("?").split()
    expected_topic = " ".join(topic[:20])

    assert expected_topic in out


def test_message_helpers():
    assert Message.user("hi") == {"role": "user", "content": "hi"}
    assert Message.system("s")["role"] == "system"