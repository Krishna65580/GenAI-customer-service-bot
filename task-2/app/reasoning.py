"""
Reasoning layer for the multi-modal assistant.

Implements, in a fully inspectable/deterministic way (with an optional
pluggable local LLM for fluent phrasing):
  1. Contextual reasoning   -> maintains rolling conversation + image memory
  2. Ambiguity handling     -> detects underspecified questions and asks
                                a clarifying question instead of guessing
  3. Evidence-based answers -> every claim is traced back to either OCR
                                text, measured image stats, or prior turns
  4. Response validation    -> a second pass checks the drafted answer only
                                references evidence actually present before
                                it's returned; otherwise it's rewritten to
                                be conservative
"""

import re


VAGUE_PATTERNS = [
    r"^\s*(what is this|what's this|explain|describe|tell me)\s*\??\s*$",
    r"^\s*(help|hi|hello)\s*$",
]


class ConversationMemory:
    def __init__(self):
        self.turns = []          # list of {"role", "text"}
        self.last_image_evidence = None
        self.last_image_name = None

    def add_turn(self, role: str, text: str):
        self.turns.append({"role": role, "text": text})

    def set_image(self, evidence: dict, name: str):
        self.last_image_evidence = evidence
        self.last_image_name = name

    def recent_context(self, n: int = 6) -> str:
        return "\n".join(f"{t['role']}: {t['text']}" for t in self.turns[-n:])

    def has_image_context(self) -> bool:
        return self.last_image_evidence is not None


def is_ambiguous(query: str, memory: ConversationMemory) -> bool:
    q = query.strip().lower()
    if len(q.split()) <= 2 and not memory.has_image_context():
        # very short query with no image on the table yet
        for pat in VAGUE_PATTERNS:
            if re.match(pat, q):
                return True
    # "this"/"it" reference with nothing established yet
    if re.search(r"\b(this|it|that)\b", q) and not memory.has_image_context() and len(memory.turns) == 0:
        return True
    return False


def clarifying_question(query: str) -> str:
    return (
        "Could you clarify what you'd like to know? For example: upload an image and "
        "ask something specific about it (e.g. \"what text is in this image?\", "
        "\"describe the colors\"), or ask a text-only question directly."
    )


def compose_answer(query: str, memory: ConversationMemory, evidence_desc: str = None,
                    llm_backend=None) -> tuple[str, list[str]]:
    """Returns (answer, evidence_used). Grounds the answer in whatever
    evidence is actually available (image evidence and/or conversation
    history) and is explicit when it has none."""
    evidence_used = []
    context_str = memory.recent_context()

    has_image = memory.has_image_context()
    if has_image:
        evidence_used.append(f"image evidence: {evidence_desc}")

    if llm_backend is not None:
        prompt = (
            "You are a careful multi-modal assistant. Only state facts that are "
            "directly supported by the evidence below; if the evidence doesn't "
            "cover the question, say so explicitly rather than guessing.\n\n"
            f"Conversation so far:\n{context_str}\n\n"
        )
        if has_image:
            prompt += f"Image evidence (from OCR + pixel analysis, not a guess):\n{evidence_desc}\n\n"
        prompt += f"User question: {query}\n\nAnswer:"
        answer = llm_backend(prompt)
        if answer:
            return answer, evidence_used

    # --- deterministic fallback ---
    if has_image:
        lines = [f"Based on the image you provided: {evidence_desc}"]
        q_lower = query.lower()
        if "text" in q_lower or "read" in q_lower or "say" in q_lower:
            if "No readable text" in evidence_desc:
                lines.append("I could not find any readable text in this image via OCR.")
            else:
                lines.append("(See the OCR excerpt above for the text found.)")
        return " ".join(lines), evidence_used
    else:
        return (
            "I don't have an image to analyze yet and this question isn't answerable "
            "from general knowledge in this offline demo. Upload an image, or ask a "
            "more specific text question."
        ), evidence_used


def validate_response(answer: str, evidence_desc: str, has_image: bool) -> tuple[str, bool]:
    """Very deliberately simple validator: if the answer claims to see text
    ('says', 'reads') but no OCR text was found, flag and soften the claim.
    Returns (possibly-rewritten answer, was_flagged)."""
    claims_text = re.search(r"\b(says|reads|written|text is)\b", answer.lower())
    if has_image and claims_text and "No readable text" in (evidence_desc or ""):
        rewritten = (
            answer + "\n\n⚠️ Validation note: no OCR text was actually detected in this "
            "image, so any specific wording above should be treated as uncertain."
        )
        return rewritten, True
    return answer, False
