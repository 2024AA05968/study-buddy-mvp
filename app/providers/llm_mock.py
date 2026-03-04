# app/providers/llm_mock.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class EvalResult:
    verdict: str  # "correct" | "partial" | "needs_help"
    encouragement: str
    gentle_correction: str
    retry_suggestion: str


class MockLLMProvider:
    """
    A deterministic, offline 'LLM' used to test the MVP flow without any API calls.
    It generates simple questions from lesson text and evaluates answers using keyword overlap.
    """

    def __init__(self) -> None:
        # Fixed, kid-friendly phrases to keep tone consistent and predictable
        self._encouragement = [
            "Nice effort! 😊",
            "Good try! 👍",
            "Well done for explaining! 🌟",
            "Thanks for sharing your thoughts! 🙂",
        ]

    # ---------- Public API (what the rest of the app will call) ----------

    def generate_warmup_question(self, lesson_text: str) -> str:
        topic = self._extract_topic_hint(lesson_text)
        if topic:
            return f"Let’s start easy 😊 What do you think this lesson is mainly about: {topic}?"
        return "Let’s start easy 😊 What do you think this lesson is mainly about?"

    def generate_concept_questions(self, lesson_text: str, n: int = 5) -> List[str]:
        concepts = self._extract_concepts(lesson_text, limit=max(n, 8))
        questions: List[str] = []

        # Create questions that encourage explanation, not exam-style recall.
        for c in concepts[:n]:
            questions.append(f"Can you explain “{c}” in your own words?")

        # If not enough concepts found, fall back to generic questions
        while len(questions) < n:
            idx = len(questions) + 1
            questions.append(f"Can you tell me one important point from this lesson? (Question {idx})")

        return questions

    def evaluate_answer(self, lesson_text: str, question: str, answer: str) -> EvalResult:
        """
        Simple heuristic:
        - Extract keywords from lesson + question
        - Measure overlap with student's answer
        - Decide verdict based on overlap ratio
        """
        answer = (answer or "").strip()
        if not answer:
            return EvalResult(
                verdict="needs_help",
                encouragement="It’s okay 😊 Take your time.",
                gentle_correction="Try saying what you remember in a simple way.",
                retry_suggestion="Want to try again with one short sentence?",
            )

        # Keywords from lesson + question (important words only)
        lesson_kw = self._keywords(lesson_text)
        q_kw = self._keywords(question)
        target_kw = list(dict.fromkeys(q_kw + lesson_kw))  # preserve order, unique

        ans_kw = set(self._keywords(answer))

        # Score: how many target keywords appear in the answer
        if not target_kw:
            overlap_ratio = 0.0
        else:
            overlap = sum(1 for w in target_kw[:20] if w in ans_kw)  # cap to avoid huge lessons
            overlap_ratio = overlap / max(1, min(20, len(target_kw)))

        # Decide verdict thresholds (tunable later)
        if overlap_ratio >= 0.35:
            verdict = "correct"
        elif overlap_ratio >= 0.15:
            verdict = "partial"
        else:
            verdict = "needs_help"

        # Build gentle feedback based on verdict
        if verdict == "correct":
            return EvalResult(
                verdict="correct",
                encouragement="Great job! You explained it clearly 👍",
                gentle_correction="",
                retry_suggestion="Ready for the next question?",
            )

        if verdict == "partial":
            return EvalResult(
                verdict="partial",
                encouragement="Good thinking! 😊 You’ve got part of it.",
                gentle_correction="One small thing to add: try including one more key idea from the lesson.",
                retry_suggestion="Want to try saying it once more in your own words?",
            )

        # needs_help
        hint = self._build_hint(lesson_text, question)
        return EvalResult(
            verdict="needs_help",
            encouragement="Nice attempt 😊 Let’s think about it together.",
            gentle_correction=hint,
            retry_suggestion="Want to try again with the hint in mind?",
        )

    def generate_wrapup(self, lesson_text: str, asked_questions: List[str]) -> str:
        concepts = self._extract_concepts(lesson_text, limit=6)
        bullets = concepts[:4] if concepts else ["the main idea", "important points", "how things work", "one example"]

        lines = ["🌟 Great job today!", "Here’s what you covered:"]
        for b in bullets:
            lines.append(f"• {b}")

        lines.append("You did well by trying and explaining in your own words. See you next time! 😊")
        return "\n".join(lines)

    # ---------- Helpers (internal) ----------

    def _extract_topic_hint(self, text: str) -> str:
        # Use first non-empty line as a topic hint if it looks like a heading
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return ""
        first = lines[0]
        # Keep it short and clean
        return first[:60]

    def _extract_concepts(self, text: str, limit: int = 8) -> List[str]:
        """
        Extract 'concept-like' phrases from lesson text.
        Strategy:
        - Prefer lines that look like headings/bullets
        - Otherwise, extract frequent non-trivial words
        """
        text = text or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        concepts: List[str] = []

        # 1) Heading/bullet-like lines
        for ln in lines:
            if ln.startswith(("-", "*", "•")):
                item = ln.lstrip("-*•").strip()
                if 3 <= len(item) <= 80:
                    concepts.append(item)
            # heading heuristic: short line with title-like casing
            elif 3 <= len(ln) <= 60 and (ln.isupper() or ln.istitle()):
                concepts.append(ln)

        # 2) If still not enough, use top frequent keywords
        if len(concepts) < limit:
            kws = self._keywords(text)
            freq: Dict[str, int] = {}
            for w in kws:
                freq[w] = freq.get(w, 0) + 1
            common = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
            for w, _ in common:
                if len(concepts) >= limit:
                    break
                # Turn single keywords into concept phrases (simple)
                concepts.append(w)

        # Deduplicate while preserving order
        concepts = list(dict.fromkeys(concepts))
        return concepts[:limit]

    def _keywords(self, text: str) -> List[str]:
        """
        Extract lowercased keywords (simple tokenization + stopwords).
        """
        text = (text or "").lower()
        tokens = re.findall(r"[a-zA-Z]{3,}", text)

        stop = {
            "the", "and", "for", "with", "that", "this", "from", "they", "their", "there",
            "what", "when", "where", "which", "your", "about", "into", "than", "then",
            "have", "has", "had", "will", "would", "could", "should", "can", "may",
            "are", "was", "were", "been", "being", "also", "because", "using",
            "lesson", "today", "question", "explain", "words",
        }
        return [t for t in tokens if t not in stop]

    def _build_hint(self, lesson_text: str, question: str) -> str:
        # Provide a gentle hint using a short excerpt from the lesson
        lines = [ln.strip() for ln in (lesson_text or "").splitlines() if ln.strip()]
        if not lines:
            return "Try looking at the lesson again and pick one simple idea to explain."

        # Use the first line that shares a keyword with the question
        qk = set(self._keywords(question))
        for ln in lines[:25]:
            if qk.intersection(self._keywords(ln)):
                return f"Hint: Think about this line from the lesson → “{ln[:120]}”"

        # Fallback: first meaningful line
        return f"Hint: Look at this part of the lesson → “{lines[0][:120]}”"