# Sentiment-Aware Customer Support Chatbot

Built to satisfy:
> "Integrate sentiment analysis into the chatbot to detect and respond
> appropriately to customer emotions during interactions. Expected Outcome:
> A chatbot that can recognize and address positive, negative, or neutral
> sentiments in user messages. Evaluation Criteria: Accuracy of sentiment
> detection, appropriateness of responses to different sentiments, impact on
> customer satisfaction."

## How it works
- **Sentiment detection** (`app/sentiment.py`) — uses VADER, a lexicon-based
  sentiment analyzer tuned for short, informal text (chat messages, not
  formal writing). It ships its lexicon inside the pip package, so it runs
  fully offline with no model download required. Classifies each message
  as positive / negative / neutral, with an intensity level (none/mild/strong).
- **Intent detection** (`app/response_engine.py`) — lightweight keyword
  classifier for common support intents: order status, refund, complaint,
  praise, product question, general.
- **Sentiment-adaptive responses** — the same intent produces a different
  response depending on detected sentiment: a furious complaint gets an
  apology and an offer to escalate to a human agent; a happy message gets a
  warm, brief reply; a neutral question gets a straightforward answer.
- **Escalation logic** — strong negative sentiment + complaint/refund intent
  automatically triggers a simulated hand-off to a human agent.

## Meeting the evaluation criteria
| Criterion | How it's addressed |
|---|---|
| Accuracy of sentiment detection | **Accuracy Evaluation** tab runs the detector against a 20-message hand-labeled test set (`data/eval_set.json`) and reports accuracy + a confusion table + misclassified examples |
| Appropriateness of responses | **Chat** tab shows tone + content adapting live to each message's detected sentiment and intent, with the detected sentiment/intent shown inline for transparency |
| Impact on customer satisfaction | **Sentiment Analytics** tab tracks the sentiment trend across a conversation and derives a simple, transparent "satisfaction proxy" (0-100) from the running average sentiment score |

## Demo flow (see screenshots/)
1. `01_chat_initial.png` — fresh chat
2. `02_negative_escalation.png` — an angry refund complaint triggers an empathetic response + automatic escalation
3. `03_positive_response.png` — a thank-you message gets a warm, brief reply
4. `04_neutral_response.png` — a plain question gets a straightforward, neutral-toned answer
5. `05_analytics.png` — sentiment trend chart and satisfaction proxy across the conversation so far
6. `06_accuracy_eval.png` — accuracy score + per-message results on the labeled evaluation set

## Run it
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Limitations / next steps
- VADER is lexicon-based, not a trained classifier — it's fast and fully
  offline but can miss sarcasm, negation nuance, or domain-specific phrasing.
  Swapping in a fine-tuned transformer (e.g. a DistilBERT sentiment model)
  would improve accuracy at the cost of needing model weights.
- Intent detection is keyword-based for this demo; a real deployment would
  use a trained intent classifier or an LLM-based classifier.
- The "satisfaction proxy" is a transparent stand-in metric, not a validated
  CSAT score — in production it should be correlated against real post-chat
  survey results before being trusted as a satisfaction measure.
