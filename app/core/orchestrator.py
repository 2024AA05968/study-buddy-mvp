# app/core/orchestrator.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

from app.providers.llm_mock import MockLLMProvider, EvalResult


@dataclass
class SessionState:
    lesson_text: str
    warmup_question: Optional[str] = None
    concept_questions: List[str] = field(default_factory=list)
    asked_questions: List[str] = field(default_factory=list)


class Orchestrator:
    """
    Controls the revision session flow.

    Current MVP coverage:
    - Start session: generate 1 warm-up + 5 concept questions
    - Respond to one answer: return EvalResult (correct/partial/needs_help)
    - Wrap-up: generate summary based on lesson + asked questions
    """

    def __init__(self, llm: Optional[MockLLMProvider] = None) -> None:
        self.llm = llm or MockLLMProvider()

    def start_session(self, lesson_text: str) -> SessionState:
        """
        Creates a session state and generates:
        - 1 warm-up question
        - 5 concept questions
        Also tracks asked questions for wrap-up generation.
        """
        state = SessionState(lesson_text=lesson_text)

        state.warmup_question = self.llm.generate_warmup_question(lesson_text)
        state.concept_questions = self.llm.generate_concept_questions(lesson_text, n=5)

        # Track the questions asked in this session (warm-up + concept questions)
        warmup = [state.warmup_question] if state.warmup_question else []
        state.asked_questions = warmup + state.concept_questions

        return state

    def respond_to_answer(self, state: SessionState, question: str, answer: str) -> EvalResult:
        """
        Evaluates a student's answer for a given question using the provider.
        MVP step: one question -> one evaluation result.
        """
        return self.llm.evaluate_answer(state.lesson_text, question, answer)

    def wrapup(self, state: SessionState) -> str:
        """
        Generates the end-of-session summary based on lesson text and asked questions.
        """
        return self.llm.generate_wrapup(state.lesson_text, state.asked_questions)