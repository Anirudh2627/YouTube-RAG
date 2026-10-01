"""RAG engine behavior: grounding, refusal, off-topic, follow-ups, debug."""
from app.generation.prompts import REFUSAL_SENTENCE
from app.generation.mock import OFF_TOPIC


def test_grounded_answer_with_citations(demo_container, demo):
    resp = demo_container.engine.answer(
        "What learning rate does the speaker recommend for fine-tuning BERT?",
        video_id=demo.video_id, debug=True)
    assert resp.answer
    assert resp.sources, "must return sources"
    assert resp.conversation_id
    # at least one source overlaps the gold lr segment (410–440s)
    assert any(s.start_time < 442 and s.end_time > 408 for s in resp.sources)
    # markdown contains clickable timestamp links
    assert "youtube.com/watch?v=" in resp.answer_markdown
    assert "&t=" in resp.answer_markdown
    assert resp.debug is not None
    assert resp.debug.original_query.startswith("What learning rate")


def test_unanswerable_question_refuses(demo_container, demo):
    resp = demo_container.engine.answer(
        "What does the speaker say about underwater basket weaving?",
        video_id=demo.video_id)
    low = resp.answer.lower()
    assert (REFUSAL_SENTENCE.lower() in low) or ("unrelated" in low)


def test_offtopic_question(demo_container, demo):
    resp = demo_container.engine.answer("Plan my wedding menu please",
                                        video_id=demo.video_id)
    assert resp.answer  # mock returns off-topic or refusal; must not fabricate
    assert "wedding" not in resp.answer.lower()


def test_followup_uses_conversation(demo_container, demo):
    r1 = demo_container.engine.answer("What is the attention mechanism?",
                                      video_id=demo.video_id)
    r2 = demo_container.engine.answer("Why is it better than recurrence?",
                                      video_id=demo.video_id,
                                      conversation_id=r1.conversation_id,
                                      debug=True)
    assert r2.conversation_id == r1.conversation_id
    # heuristic rewriter must have contextualized the follow-up
    assert r2.debug.rewritten_query is not None
    assert "attention" in r2.debug.rewritten_query.lower()


def test_cited_indices_valid(demo_container, demo):
    resp = demo_container.engine.answer("How many attention heads does the Transformer use?",
                                        video_id=demo.video_id)
    for i in resp.cited_indices:
        assert 0 <= i < len(resp.sources)


def test_debug_trace_contents(demo_container, demo):
    resp = demo_container.engine.answer("Why is warmup needed?", video_id=demo.video_id,
                                        debug=True)
    d = resp.debug
    assert d.final_context and "[C1]" in d.final_context
    assert d.llm_model
    assert "total" in d.timings_ms
    assert d.candidates  # stage-1 list recorded


def test_cross_video_scope(demo_container, demo):
    # without video_id, retrieval spans the whole collection
    resp = demo_container.engine.answer("What is the attention mechanism?")
    assert resp.video_ids
