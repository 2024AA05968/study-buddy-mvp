# app/streamlit_app.py

from __future__ import annotations

import streamlit as st

from app.core.orchestrator import Orchestrator, SessionState


st.set_page_config(page_title="Study Buddy MVP", page_icon="📚", layout="centered")


def _init_state() -> None:
    if "orch" not in st.session_state:
        st.session_state.orch = Orchestrator()

    if "session" not in st.session_state:
        st.session_state.session = None  # type: SessionState | None

    if "questions" not in st.session_state:
        st.session_state.questions = []  # list[str]

    if "q_index" not in st.session_state:
        st.session_state.q_index = 0

    if "answer" not in st.session_state:
        st.session_state.answer = ""

    if "feedback" not in st.session_state:
        st.session_state.feedback = ""

    if "started" not in st.session_state:
        st.session_state.started = False


def _reset_session() -> None:
    st.session_state.session = None
    st.session_state.questions = []
    st.session_state.q_index = 0
    st.session_state.answer = ""
    st.session_state.feedback = ""
    st.session_state.started = False


def _on_next() -> None:
    """
    Move to the next question and clear the answer + feedback safely.
    IMPORTANT: This runs before widgets are re-instantiated on rerun,
    so it is safe to reset session_state['answer'] here.
    """
    st.session_state.q_index += 1
    st.session_state.answer = ""
    st.session_state.feedback = ""


_init_state()

st.title("📚 Study Buddy (MVP)")
st.caption("Paste a lesson → warm-up → 5 questions → gentle feedback → wrap-up (with retry)")

with st.expander("⚙️ Controls", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reset session", use_container_width=True):
            _reset_session()
            st.rerun()
    with col_b:
        st.write("Mode: Local (Mock LLM)")

st.divider()

# -------------------------
# Step 1: Lesson input
# -------------------------
if not st.session_state.started:
    st.subheader("1) Paste today’s lesson")

    lesson_text = st.text_area(
        "Lesson text",
        height=220,
        placeholder="Paste the lesson here (even rough notes are fine)...",
    )

    start_clicked = st.button("Start revision", type="primary", use_container_width=True)

    if start_clicked:
        if not lesson_text.strip():
            st.warning("Please paste some lesson text to begin.")
        else:
            session = st.session_state.orch.start_session(lesson_text.strip())

            # UI flow includes warm-up + 5 concept questions (total 6)
            questions = []
            if session.warmup_question:
                questions.append(session.warmup_question)
            questions.extend(session.concept_questions)

            st.session_state.session = session
            st.session_state.questions = questions
            st.session_state.q_index = 0
            st.session_state.answer = ""
            st.session_state.feedback = ""
            st.session_state.started = True
            st.rerun()

# -------------------------
# Step 2: Q&A flow (with retry)
# -------------------------
else:
    session: SessionState | None = st.session_state.session
    questions: list[str] = st.session_state.questions
    q_index: int = st.session_state.q_index

    if session is None or not questions:
        st.error("Session not initialized. Please reset and start again.")
        st.stop()

    # Finished all questions => wrap-up
    if q_index >= len(questions):
        st.subheader("✅ Wrap-up")
        summary = st.session_state.orch.wrapup(session)
        st.write(summary)

        st.success("Session complete! 🎉")
        if st.button("Start a new session", type="primary"):
            _reset_session()
            st.rerun()
        st.stop()

    is_warmup = (q_index == 0)
    st.subheader(f"{'Warm-up' if is_warmup else 'Question'} {q_index + 1} of {len(questions)}")
    st.write(f"🤖 **{questions[q_index]}**")

    # Keep the input ALWAYS enabled so retry is possible.
    st.text_input(
        "Your answer",
        key="answer",
        placeholder="Type your answer here... (You can retry after feedback)",
    )

    col_submit, col_next = st.columns([1, 1])

    with col_submit:
        submit_clicked = st.button(
            "Submit answer",
            type="primary",
            use_container_width=True,
        )

    with col_next:
        next_clicked = st.button(
            "Next",
            use_container_width=True,
            # Next becomes available only after feedback exists
            disabled=not bool(st.session_state.feedback),
            on_click=_on_next,
        )

    # Compute / update feedback on submit (can be done multiple times = retry)
    if submit_clicked:
        answer = (st.session_state.answer or "").strip()
        if not answer:
            st.warning("Please type a short answer before submitting.")
        else:
            if is_warmup:
                # Warm-up: no evaluation; just encouragement.
                st.session_state.feedback = "Nice! 😊 Thanks for sharing. Now let’s go to the next questions."
            else:
                result = st.session_state.orch.respond_to_answer(
                    session,
                    questions[q_index],
                    answer,
                )
                parts = [result.encouragement]
                if result.gentle_correction:
                    parts.append(result.gentle_correction)
                if result.retry_suggestion:
                    parts.append(result.retry_suggestion)

                # Show feedback; user can edit answer and hit Submit again to retry.
                st.session_state.feedback = "\n\n".join(parts)

            st.rerun()

    # Show feedback (if any)
    if st.session_state.feedback:
        st.info(st.session_state.feedback)

    # Note: No manual state updates for Next here.
    # Next uses the callback to safely clear answer/feedback and advance.