"""
Sentiment-adaptive response engine for a customer-support chatbot.

Combines a lightweight intent classifier (keyword-based, since this is a
demo without a labeled intent-classification training set) with the
sentiment label from sentiment.py to select a response tone AND content
that's appropriate to both *what* the customer is asking and *how* they
feel about it -- the same question gets a different opening depending on
whether the customer sounds angry, happy, or neutral.
"""

import re

INTENT_KEYWORDS = {
    "order_status": ["order", "shipment", "delivery", "tracking", "package", "shipped", "where is my"],
    "refund": ["refund", "money back", "cancel my order", "return this", "want my money"],
    "complaint": ["broken", "defective", "not working", "doesn't work", "terrible", "worst",
                  "disappointed", "unacceptable", "faulty", "damaged"],
    "praise": ["thank you", "thanks", "great job", "awesome", "love this", "excellent", "amazing", "appreciate"],
    "product_question": ["how do i", "how to", "what is", "does it", "can i", "compatible", "which"],
}


def detect_intent(text: str) -> str:
    t = text.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return intent
    return "general"


INTENT_BODIES = {
    "order_status": "Let me help you track that down. Could you share your order number so I can "
                     "look up the current shipping status?",
    "refund": "I can help start a refund for you. To process it I'll need your order number and the "
              "reason for the return -- once I have that I can get it moving right away.",
    "complaint": "I'm sorry to hear the product isn't working as expected. Could you tell me a bit "
                 "more about what's happening? I'd like to get this fixed for you, whether that's a "
                 "replacement, repair, or refund.",
    "praise": "That means a lot -- thank you for letting us know! Is there anything else I can help "
              "you with today?",
    "product_question": "Happy to help with that. Could you give me a bit more detail on exactly what "
                         "you're trying to do, so I point you to the right answer?",
    "general": "I'm here to help -- could you tell me a bit more about what you need?",
}

# Tone prefixes layered on top of the intent body, based on detected sentiment + intensity
TONE_PREFIXES = {
    ("negative", "strong"): "I can hear how frustrating this has been, and I'm really sorry. ",
    ("negative", "mild"): "Sorry for the trouble -- let's get this sorted out. ",
    ("positive", "strong"): "So glad to hear that! ",
    ("positive", "mild"): "Glad things are going well! ",
    ("neutral", "none"): "",
    ("neutral", "mild"): "",
}


def needs_escalation(sentiment: dict, intent: str) -> bool:
    return sentiment["label"] == "negative" and sentiment["intensity"] == "strong" and intent in (
        "complaint", "refund"
    )


def compose_response(text: str, sentiment: dict, intent: str) -> tuple[str, bool]:
    prefix = TONE_PREFIXES.get((sentiment["label"], sentiment["intensity"]), "")
    body = INTENT_BODIES.get(intent, INTENT_BODIES["general"])
    escalate = needs_escalation(sentiment, intent)

    response = prefix + body
    if escalate:
        response += (
            "\n\nGiven how significant this issue is, I'm also going to loop in a human specialist "
            "so this gets resolved as quickly as possible."
        )
    return response, escalate
