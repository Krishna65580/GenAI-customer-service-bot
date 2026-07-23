"""
Orchestrates the multilingual chatbot's decision-making for a single turn:
  1. detect language (handling mixed-language input)
  2. try to match an intent using cross-lingual keyword matching
  3. if no intent matches but this looks like a follow-up, reuse the
     conversation's current topic (context retention across turns AND
     across language switches)
  4. if still nothing, ask a clarifying question in the user's language
     (ambiguity handling) instead of guessing
  5. respond in the detected language, using the SAME underlying intent
     logic regardless of which language was used (response consistency)
"""

from language_id import detect_mixed_language, LANGUAGE_NAMES
from intent_matcher import best_intent, get_response
from conversation_context import ConversationContext


def handle_turn(text: str, kb: dict, context: ConversationContext) -> dict:
    lang_info = detect_mixed_language(text)
    lang = lang_info["dominant"] if lang_info["dominant"] in kb["languages"] else "en"

    intent = best_intent(text, kb)

    used_context = False
    if intent is None and context.is_followup(text) and context.current_intent:
        # Ambiguous short follow-up ("and how long?") -- resolve using the
        # conversation's existing topic rather than asking again.
        intent = context.current_intent
        used_context = True

    if intent is None:
        # Genuinely ambiguous with no prior topic to fall back on --
        # ask for clarification instead of guessing.
        response = kb["clarifying_question"].get(lang, kb["clarifying_question"]["en"])
        context.add_user_turn(text, lang, intent=None)
        context.add_assistant_turn(response, lang)
        return {
            "lang": lang, "lang_name": LANGUAGE_NAMES.get(lang, lang),
            "intent": None, "used_context": False, "is_mixed": lang_info["is_mixed"],
            "response": response, "clarifying": True,
        }

    response = get_response(intent, lang, kb)
    context.add_user_turn(text, lang, intent=intent)
    context.add_assistant_turn(response, lang)

    return {
        "lang": lang, "lang_name": LANGUAGE_NAMES.get(lang, lang),
        "intent": intent, "used_context": used_context, "is_mixed": lang_info["is_mixed"],
        "response": response, "clarifying": False,
        "chunk_languages": [c["lang"] for c in lang_info["chunks"]] if lang_info["is_mixed"] else None,
    }
