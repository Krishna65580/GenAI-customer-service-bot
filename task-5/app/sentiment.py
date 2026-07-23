"""
Sentiment detection for the customer-support chatbot.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) -- a
lexicon-based analyzer tuned for short, informal text like chat messages
and social media, which is exactly the register customer support messages
tend to use ("ugh this is SO frustrating!!", "great, thanks :)"). It ships
its lexicon inside the pip package, so it runs fully offline with no model
download required.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Thresholds recommended by the VADER authors for compound score -> label
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


def detect_sentiment(text: str) -> dict:
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= POS_THRESHOLD:
        label = "positive"
    elif compound <= NEG_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    # Rough intensity bucket, used to pick how strongly to adapt the tone
    if abs(compound) >= 0.6:
        intensity = "strong"
    elif abs(compound) >= 0.05:
        intensity = "mild"
    else:
        intensity = "none"

    return {
        "label": label,
        "compound": compound,
        "pos": scores["pos"],
        "neg": scores["neg"],
        "neu": scores["neu"],
        "intensity": intensity,
    }
