from app.evaluation.judges import HeuristicJudge, JudgeInput
from app.evaluation.metrics import GoldItem

CONTEXT = [("The transformer uses eight attention heads with dimension sixty-four.",
            260.0, 284.0, "v")]
GOLD = [GoldItem(video_id="v", start=260.0, end=284.0)]


def ji(answer, cited=((260.0, 284.0, "v"),), expected="eight attention heads dimension sixty four",
       refusal=False, context=CONTEXT, gold=GOLD):
    return JudgeInput(question="How many heads?", expected_answer=expected,
                      expected_refusal=refusal, gold=list(gold), answer=answer,
                      context_chunks=list(context),
                      cited_spans=[tuple(c) for c in cited])


def test_good_answer_scores_high():
    s = HeuristicJudge().score(ji("The transformer uses eight attention heads with dimension sixty-four."))
    assert s.answer_correctness > 0.7
    assert s.faithfulness == 1.0          # verbatim from context
    assert s.citation_accuracy == 1.0     # cited span == gold span
    assert s.refused is False


def test_unfaithful_answer_penalized():
    s = HeuristicJudge().score(ji(
        "The model uses eight attention heads and was trained on fourteen GPUs with batch size 8192."))
    assert s.faithfulness < 1.0           # second sentence not in context


def test_missing_citation_penalized():
    s = HeuristicJudge().score(ji(
        "The transformer uses eight attention heads with dimension sixty-four.", cited=()))
    assert s.citation_accuracy == 0.0


def test_wrong_citation_penalized():
    s = HeuristicJudge().score(ji(
        "The transformer uses eight attention heads with dimension sixty-four.",
        cited=((0.0, 10.0, "v"),)))
    assert s.citation_accuracy < 1.0


def test_expected_refusal_correct():
    s = HeuristicJudge().score(ji(
        "I couldn't find enough information about that in the video.",
        refusal=True, expected="", gold=[]))
    assert s.refusal_correct is True
    assert s.answer_correctness == 1.0


def test_expected_refusal_but_answered():
    s = HeuristicJudge().score(ji(
        "Sure! Here is a full answer about something else entirely.",
        refusal=True, expected="", gold=[]))
    assert s.refusal_correct is False
    assert s.answer_correctness == 0.0


def test_refused_answerable_question():
    s = HeuristicJudge().score(ji(
        "I couldn't find enough information about that in the video.", refusal=False))
    assert s.refusal_correct is None      # answerable → not a refusal judgment
    assert s.refused is True
    assert s.answer_correctness == 0.0


def test_context_relevance():
    s = HeuristicJudge().score(ji(
        "eight attention heads", context=[("cooking pasta with tomato", 0, 5, "v")],
        cited=((0, 5, "v"),), gold=[GoldItem(video_id="v", start=0, end=5)]))
    assert s.context_relevance == 0.0     # pasta context irrelevant to "heads" question
