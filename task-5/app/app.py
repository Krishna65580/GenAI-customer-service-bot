import os
import sys
import json

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sentiment import detect_sentiment
from response_engine import detect_intent, compose_response

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

st.set_page_config(page_title="Sentiment-Aware Support Chatbot", page_icon="💬", layout="wide")

SENTIMENT_COLOR = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
SENTIMENT_EMOJI = {"positive": "🙂", "neutral": "😐", "negative": "🙁"}

if "history" not in st.session_state:
    st.session_state.history = []          # [{role, text, sentiment, intent}]
if "sentiment_log" not in st.session_state:
    st.session_state.sentiment_log = []    # per-turn compound scores, for the trend chart
if "escalations" not in st.session_state:
    st.session_state.escalations = 0

page = st.sidebar.radio("Navigate", ["💬 Chat", "📊 Sentiment Analytics", "🧪 Accuracy Evaluation"])

st.sidebar.markdown("---")
st.sidebar.caption("Sentiment engine: VADER (lexicon-based, fully offline)")
st.sidebar.metric("Messages analyzed", len(st.session_state.sentiment_log))
st.sidebar.metric("Escalations triggered", st.session_state.escalations)
if st.sidebar.button("🔄 Reset conversation"):
    st.session_state.history = []
    st.session_state.sentiment_log = []
    st.session_state.escalations = 0
    st.rerun()

# ==================== Chat Page ====================
if page == "💬 Chat":
    st.title("💬 Sentiment-Aware Customer Support Chatbot")
    st.caption(
        "Detects whether each message is positive, negative, or neutral, and adapts both the "
        "tone and content of the response accordingly -- e.g. an angry complaint gets an "
        "apology + escalation offer, while a happy message gets a warm, brief reply."
    )

    if not st.session_state.history:
        st.session_state.history.append({
            "role": "assistant", "text": "Hi, welcome to support! How can I help you today?",
            "sentiment": None, "intent": None,
        })

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg["sentiment"]:
                s = msg["sentiment"]
                st.caption(
                    f"{SENTIMENT_EMOJI[s['label']]} detected sentiment: **{s['label']}** "
                    f"(compound {s['compound']:.2f}) · intent: `{msg['intent']}`"
                )

    if prompt := st.chat_input("Type a customer message..."):
        sentiment = detect_sentiment(prompt)
        intent = detect_intent(prompt)
        st.session_state.history.append({
            "role": "user", "text": prompt, "sentiment": sentiment, "intent": intent
        })
        st.session_state.sentiment_log.append({
            "turn": len(st.session_state.sentiment_log) + 1,
            "text": prompt, "label": sentiment["label"], "compound": sentiment["compound"],
        })

        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(
                f"{SENTIMENT_EMOJI[sentiment['label']]} detected sentiment: **{sentiment['label']}** "
                f"(compound {sentiment['compound']:.2f}) · intent: `{intent}`"
            )

        response, escalated = compose_response(prompt, sentiment, intent)
        if escalated:
            st.session_state.escalations += 1

        with st.chat_message("assistant"):
            st.markdown(response)
            if escalated:
                st.warning("🚨 Escalated to a human agent due to strong negative sentiment.")

        st.session_state.history.append({
            "role": "assistant", "text": response, "sentiment": None, "intent": None
        })

    st.markdown("---")
    st.caption("Try: *\"This is the third time my order has been late, I'm furious!\"*, "
               "*\"Thank you, that fixed it!\"*, or *\"Where is my order?\"*")

# ==================== Analytics Page ====================
elif page == "📊 Sentiment Analytics":
    st.title("📊 Sentiment Analytics")
    st.caption("Tracks sentiment across the conversation as a proxy for customer satisfaction impact.")

    if not st.session_state.sentiment_log:
        st.info("No messages analyzed yet — chat with the bot first.")
    else:
        df = pd.DataFrame(st.session_state.sentiment_log)

        col1, col2, col3 = st.columns(3)
        counts = df["label"].value_counts()
        col1.metric("😊 Positive", int(counts.get("positive", 0)))
        col2.metric("😐 Neutral", int(counts.get("neutral", 0)))
        col3.metric("🙁 Negative", int(counts.get("negative", 0)))

        st.markdown("### Sentiment trend across the conversation")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = [SENTIMENT_COLOR[l] for l in df["label"]]
        ax.bar(df["turn"], df["compound"], color=colors)
        ax.axhline(0, color="#333", linewidth=0.8)
        ax.set_xlabel("Message #")
        ax.set_ylabel("Compound sentiment score")
        ax.set_ylim(-1, 1)
        st.pyplot(fig)

        st.markdown("### Estimated customer satisfaction trend")
        # Simple proxy: rolling average compound score, mapped to a 0-100 "satisfaction" score
        df["satisfaction_proxy"] = ((df["compound"].expanding().mean() + 1) / 2 * 100).round(1)
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(df["turn"], df["satisfaction_proxy"], marker="o", color="#3498db")
        ax2.set_ylim(0, 100)
        ax2.set_xlabel("Message #")
        ax2.set_ylabel("Satisfaction proxy (0-100)")
        st.pyplot(fig2)
        st.caption(
            "Satisfaction proxy = running average of sentiment compound scores, rescaled to 0-100. "
            "This is a simple, transparent stand-in for a real CSAT metric — in production this "
            "would be validated against actual post-chat satisfaction survey scores."
        )

        st.markdown("### Message log")
        st.dataframe(df[["turn", "text", "label", "compound"]], use_container_width=True)

# ==================== Evaluation Page ====================
else:
    st.title("🧪 Accuracy Evaluation")
    st.caption(
        "Runs the sentiment detector against a small hand-labeled evaluation set to measure "
        "detection accuracy — one of the task's evaluation criteria."
    )

    with open(os.path.join(DATA_DIR, "eval_set.json")) as f:
        eval_set = json.load(f)

    if st.button("▶️ Run evaluation"):
        rows = []
        correct = 0
        for item in eval_set:
            pred = detect_sentiment(item["text"])["label"]
            is_correct = pred == item["label"]
            correct += is_correct
            rows.append({
                "text": item["text"], "true_label": item["label"],
                "predicted_label": pred, "correct": "✅" if is_correct else "❌",
            })
        acc = correct / len(eval_set)
        st.metric("Accuracy on labeled eval set", f"{acc*100:.1f}%", f"{correct}/{len(eval_set)} correct")

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        st.markdown("### Confusion breakdown")
        conf = pd.crosstab(df["true_label"], df["predicted_label"])
        st.dataframe(conf)

        wrong = df[df["correct"] == "❌"]
        if not wrong.empty:
            st.markdown("### Misclassified examples")
            for _, r in wrong.iterrows():
                st.write(f"- *\"{r['text']}\"* — true: `{r['true_label']}`, predicted: `{r['predicted_label']}`")
    else:
        st.info(f"Evaluation set has {len(eval_set)} labeled customer messages "
                f"(positive / negative / neutral). Click **Run evaluation** to score the detector.")
        st.dataframe(pd.DataFrame(eval_set), use_container_width=True)
