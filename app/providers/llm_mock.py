# app/providers/llm_mock.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class EvalResult:
    verdict: str  # "correct" | "partial" | "needs_help"
    encouragement: str
    gentle_correction: str
    retry_suggestion: str


class MockLLMProvider:
    """
    Deterministic, offline 'LLM' used to test MVP flow without API calls.

    Improvements:
    - Extracts simple 2-word phrases (e.g., "Carbon Dioxide") for better concept questions.
    - Avoids weak single-word concepts via stopword filtering and fallback logic.
    - Does NOT form phrases across punctuation (fixes "Food Plants", "Sunlight Water").
    - Allows "sunlight" and "water" as single-word concepts (so we can reliably get 5).
    """

    # Stopwords used for token filtering in general
    _STOP_BASE = {
        # common function words
        "the", "and", "for", "with", "that", "this", "from", "they", "their", "there",
        "what", "when", "where", "which", "your", "about", "into", "than", "then",
        "have", "has", "had", "will", "would", "could", "should", "can", "may",
        "are", "was", "were", "been", "being", "also", "because",

        # app/meta words
        "lesson", "today", "question", "explain", "words",
    }

    # Extra stopwords used ONLY when selecting single-word concepts (to avoid boring questions)
    # NOTE: sunlight and water have been REMOVED here so they can be used as concepts.
    _STOP_SINGLE_EXTRA = {
        # filler verbs / generic words
        "how", "make", "made", "making", "process", "mainly", "needs", "need",
        "using", "use", "used",

        # weak concept words
        "own", "happens", "happen", "happening", "green",

        # overly common topic words (good for phrases, not great as single words)
        "plants", "plant", "food", "carbon", "dioxide",
    }

    def __init__(self) -> None:
        self._encouragement = [
            "Nice effort! 😊",
            "Good try! 👍",
            "Well done for explaining! 🌟",
            "Thanks for sharing your thoughts! 🙂",
        ]

    # ---------- Public API ----------

    def generate_warmup_question(self, lesson_text: str) -> str:
        topic = self._extract_topic_hint(lesson_text)
        if topic:
            return f"Let’s start easy 😊 What do you think this lesson is mainly about: {topic}?"
        return "Let’s start easy 😊 What do you think this lesson is mainly about?"

    def generate_concept_questions(self, lesson_text: str, n: int = 5) -> List[str]:
        concepts = self._extract_concepts(lesson_text, limit=max(n, 10))
        questions: List[str] = []

        for c in concepts[:n]:
            questions.append(f"Can you explain “{c}” in your own words?")

        while len(questions) < n:
            idx = len(questions) + 1
            questions.append(f"Can you tell me one important point from this lesson? (Question {idx})")

        return questions

    def evaluate_answer(self, lesson_text: str, question: str, answer: str) -> EvalResult:
        answer = (answer or "").strip()
        if not answer:
            return EvalResult(
                verdict="needs_help",
                encouragement="It’s okay 😊 Take your time.",
                gentle_correction="Try saying what you remember in a simple way.",
                retry_suggestion="Want to try again with one short sentence?",
            )

        lesson_kw = self._keywords(lesson_text)
        q_kw = self._keywords(question)
        target_kw = list(dict.fromkeys(q_kw + lesson_kw))  # unique, preserve order

        ans_kw = set(self._keywords(answer))

        if not target_kw:
            overlap_ratio = 0.0
        else:
            overlap = sum(1 for w in target_kw[:20] if w in ans_kw)
            overlap_ratio = overlap / max(1, min(20, len(target_kw)))

        if overlap_ratio >= 0.35:
            verdict = "correct"
        elif overlap_ratio >= 0.15:
            verdict = "partial"
        else:
            verdict = "needs_help"

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

    # ---------- Helpers ----------

    def _extract_topic_hint(self, text: str) -> str:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return ""
        return lines[0][:60]

    def _extract_concepts(self, text: str, limit: int = 10) -> List[str]:
        """
        Extract concept-like items in this order:
        1) Bullet/heading lines
        2) Frequent 2-word phrases (bigrams) without crossing punctuation
        3) Frequent single keywords (filtered more aggressively)
        """
        text = text or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        concepts: List[str] = []

        # 1) Bullet-like items or headings
        for ln in lines:
            if ln.startswith(("-", "*", "•")):
                item = ln.lstrip("-*•").strip()
                if 3 <= len(item) <= 80:
                    concepts.append(item)
            elif 3 <= len(ln) <= 60 and (ln.isupper() or ln.istitle()):
                concepts.append(ln)

        # 2) Two-word phrase extraction (bigrams) (no punctuation crossing)
        if len(concepts) < limit:
            phrase_freq = self._phrase_frequencies(text)
            for phrase, _ in phrase_freq:
                if len(concepts) >= limit:
                    break
                concepts.append(phrase.title())

        # 3) Fallback to frequent single keywords (more aggressive filtering)
        if len(concepts) < limit:
            kws = self._keywords(text)
            freq: Dict[str, int] = {}
            for w in kws:
                freq[w] = freq.get(w, 0) + 1

            common = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
            for w, _ in common:
                if len(concepts) >= limit:
                    break
                concepts.append(w)

        return list(dict.fromkeys(concepts))[:limit]

    def _keywords(self, text: str) -> List[str]:
        """
        Single-word keywords for evaluation + single-word concept fallback.
        Uses BASE + SINGLE_EXTRA stopwords to avoid boring concept words.
        """
        text = (text or "").lower()
        tokens = re.findall(r"[a-zA-Z]{3,}", text)
        stop = self._STOP_BASE | self._STOP_SINGLE_EXTRA
        return [t for t in tokens if t not in stop]

    def _phrase_frequencies(self, text: str) -> List[Tuple[str, int]]:
        """
        Build frequencies of simple 2-word phrases (bigrams) WITHOUT crossing punctuation.

        Fixes:
        - No phrases across sentence boundaries or separators like commas/semicolons
          so we avoid "Food Plants" (across ".") and "Sunlight Water" (across ",").
        """
        text = (text or "").lower()

        # Split on sentence boundaries and common separators.
        chunks = re.split(r"[.!?;:,\n]+", text)

        # Phrase stopwords are lighter than single-word stopwords.
        # We allow topic words like "carbon dioxide" to form a phrase,
        # even if "carbon" and "dioxide" are blocked as single-word concepts.
        stop_phrase = self._STOP_BASE | {
            "how", "make", "made", "making", "using", "use", "used",
            "mainly", "process", "needs", "need",
            "own", "happens", "happen", "happening",
        }

        freq: Dict[str, int] = {}

        for chunk in chunks:
            words = re.findall(r"[a-zA-Z]{3,}", chunk)
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                if w1 in stop_phrase or w2 in stop_phrase:
                    continue
                if w1 == w2:
                    continue
                phrase = f"{w1} {w2}"
                freq[phrase] = freq.get(phrase, 0) + 1

        return sorted(freq.items(), key=lambda x: (-x[1], x[0]))

    def _build_hint(self, lesson_text: str, question: str) -> str:
        lines = [ln.strip() for ln in (lesson_text or "").splitlines() if ln.strip()]
        if not lines:
            return "Try looking at the lesson again and pick one simple idea to explain."

        qk = set(self._keywords(question))
        for ln in lines[:25]:
            if qk.intersection(self._keywords(ln)):
                return f"Hint: Think about this line from the lesson → “{ln[:120]}”"

        return f"Hint: Look at this part of the lesson → “{lines[0][:120]}”"