"""
Conversation context management for cross-lingual continuity.

The key requirement this satisfies: "preserving context, intent, and
conversational continuity throughout language switches." The trick is that
`current_intent` is stored as a language-agnostic key (e.g. "refund"), not
as text in any particular language -- so switching from English to Hindi
mid-conversation doesn't lose the thread, because the topic was never
tied to a language in the first place.
"""

import re

# Very short / pronoun-referencing follow-ups, in all 4 languages -- used to
# detect "this is a continuation of the previous topic" rather than a new
# standalone question.
FOLLOWUP_PATTERNS = [
    r"^\s*(and|also|what about|how about)\b",
    r"^\s*(y|también|qué tal|y qué hay de)\b",
    r"^\s*(et|aussi|qu'en est-il de)\b",
    r"^\s*(और|भी)\b",
]


class ConversationContext:
    def __init__(self):
        self.turns = []            # [{role, text, lang, intent}]
        self.current_intent = None
        self.language_history = []  # detected language per user turn

    def add_user_turn(self, text: str, lang: str, intent: str):
        self.turns.append({"role": "user", "text": text, "lang": lang, "intent": intent})
        self.language_history.append(lang)
        if intent:
            self.current_intent = intent

    def add_assistant_turn(self, text: str, lang: str):
        self.turns.append({"role": "assistant", "text": text, "lang": lang, "intent": None})

    def is_followup(self, text: str) -> bool:
        t = text.strip().lower()
        if len(t.split()) <= 4:
            for pat in FOLLOWUP_PATTERNS:
                if re.match(pat, t):
                    return True
        return False

    def languages_used(self):
        return sorted(set(self.language_history))

    def had_language_switch(self) -> bool:
        return len(self.languages_used()) > 1
