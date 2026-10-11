"""Semantic judging of free-text answers.

Scoring long-document QA by string comparison is broken in a way that is easy to
demonstrate: `Operating Activities` loses to `Operations activities`, and
`1,358,000` loses to `1358000`. MMLongBench-Doc-V2 replaced its string metric
with an LLM judge for exactly this reason, and this harness follows that
approach so that a wrong answer means a *retrieval or reading* failure rather
than a formatting one.

The judge receives only the question, the reference answer and the candidate
response. It never sees which system produced the response, so it cannot be
biased by the comparison being made.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

SYSTEM_PROMPT = """You are a strict grader for a long-document question-answering benchmark.

You are given a question, a reference answer, and a candidate response.

Decide whether the candidate response means the same thing as the reference \
answer. Grade meaning, not form:

- Ignore differences in capitalisation, punctuation, spacing and number formatting.
  "1,358,000", "1358000" and "1.358M" are the same answer.
- Accept a paraphrase that carries the same fact. "National Atmospheric Research
  Laboratory" matches "the National Atmospheric Research Laboratory (NARL)".
- Accept a longer response if it contains the reference answer and does not
  contradict it.
- Reject a response that is empty, evasive, hedged without committing ("it may
  be...", "the document does not clearly state"), or that states a different fact.
- Reject a response that contains the right words but the wrong relation, and
  reject extra claims that contradict the reference.

Answer with exactly one word: YES or NO. Do not explain."""


@dataclass
class Judge:
    """A model call used only for grading. Same judge for every system."""

    model: str
    temperature: float = 0.0
    timeout_s: int = 120
    max_retries: int = 3

    def build_messages(self, question: str, reference: str, response: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Reference answer:\n{reference}\n\n"
                    f"Candidate response:\n{response}\n\n"
                    "Does the candidate response mean the same as the reference answer? "
                    "Answer YES or NO."
                ),
            },
        ]

    @staticmethod
    def parse_verdict(text: str) -> bool | None:
        """Read a YES/NO verdict from the opening token.

        The judge is asked for a single word. A response that opens with
        anything else has not committed to a verdict and is reported as
        unscored rather than guessed at. The failure mode worth protecting
        against is a hedge — "it may be YES, though the context is unclear" —
        being silently read as a pass, which would inflate every accuracy
        figure in the report. The price of that strictness is that a chatty
        judge shows up as unscored, which is visible in the run summary.

        Only the opening token is examined, so a verdict followed by a short
        justification ("YES — the figures match") still counts.
        """
        if not text or not text.strip():
            return None
        first = re.sub(r"[^A-Za-z]", "", text.strip().split()[0]).upper()
        if first == "YES":
            return True
        if first == "NO":
            return False
        return None

    def grade(self, question: str, reference: str, response: str) -> tuple[bool | None, dict]:
        """Return (verdict, usage). Raises on persistent transport failure."""
        import litellm  # imported lazily so scoring tests need no network stack

        last: Exception | None = None
        for _ in range(self.max_retries):
            try:
                completion = litellm.completion(
                    model=self.model,
                    messages=self.build_messages(question, reference, response),
                    temperature=self.temperature,
                    timeout=self.timeout_s,
                )
                text = completion.choices[0].message.content or ""
                usage = getattr(completion, "usage", None)
                return self.parse_verdict(text), {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                }
            except Exception as exc:  # noqa: BLE001 - retried, then surfaced
                last = exc
        raise RuntimeError(f"judge failed after {self.max_retries} attempts: {last}")


def default_judge() -> Judge:
    """Judge model from the environment so a run stays reproducible and free locally."""
    return Judge(model=os.environ.get("JUDGE_MODEL", "ollama/qwen2.5:7b-instruct"))
