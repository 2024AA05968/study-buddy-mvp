from app.core.orchestrator import Orchestrator

def test_start_session_generates_warmup_question():
    orch = Orchestrator()
    state = orch.start_session("Photosynthesis is how plants make food.")
    assert state.warmup_question is not None
    assert "what do you think this lesson is mainly about" in state.warmup_question.lower()

def test_start_session_generates_5_concept_questions():
    orch = Orchestrator()
    state = orch.start_session(
        "Photosynthesis is how plants make food. It needs sunlight, water, and carbon dioxide."
    )
    assert len(state.concept_questions) == 5
    assert all(isinstance(q, str) and q.strip() for q in state.concept_questions)

from app.core.orchestrator import Orchestrator


def test_respond_to_answer_returns_evalresult():
    orch = Orchestrator()
    state = orch.start_session("Photosynthesis is how plants make food. It needs sunlight.")
    question = state.concept_questions[0]

    result = orch.respond_to_answer(state, question, "Plants make food")

    assert result.verdict in ("correct", "partial", "needs_help")
    assert isinstance(result.encouragement, str) and result.encouragement.strip()
    # correction can be empty for 'correct', so only check type
    assert isinstance(result.gentle_correction, str)

def test_wrapup_generates_summary_text():
    orch = Orchestrator()
    state = orch.start_session(
        "Photosynthesis is how plants make food. It needs sunlight, water, and carbon dioxide."
    )

    summary = orch.wrapup(state)

    assert len(state.asked_questions) == 6  # 1 warm-up + 5 concept questions
    assert isinstance(summary, str) and summary.strip()
    assert "great job" in summary.lower()