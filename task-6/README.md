# Multilingual Customer Support Chatbot

Built to satisfy:
> "Extend the existing chatbot to support multilingual conversations across
> at least three additional languages while preserving context, intent, and
> conversational continuity throughout language switches. The assistant
> should automatically identify language, manage mixed-language inputs
> within the same conversation, resolve ambiguous queries across languages,
> and maintain consistent responses regardless of the language used. The
> solution should demonstrate cross-lingual reasoning, context retention,
> and intelligent handling of multilingual interactions using open-source
> models and frameworks."

Extends the customer-support chatbot pattern from earlier tasks to
**English + 3 additional languages: Spanish, French, and Hindi**.

## How each requirement is implemented
| Requirement | Implementation |
|---|---|
| Automatic language identification | `app/language_id.py` — uses `langdetect` (offline, open-source, no model download needed) |
| Mixed-language input within one message | `detect_mixed_language()` splits a message into chunks and detects language per-chunk; `intent_matcher.py` scores keywords across **all 4 languages simultaneously**, so a message like *"Hola, where is my order?"* still resolves correctly to `order_status` |
| Context retention across language switches | `conversation_context.py`: `current_intent` is stored as a language-agnostic key (e.g. `"refund"`), never as text tied to one language — so a topic survives a switch from English to Hindi mid-conversation |
| Ambiguous query resolution | Genuinely unmatched queries get a clarifying question **in the user's detected language**, instead of a guessed answer; short pronoun-style follow-ups ("and...?", "y...?", "et...?", "और...?") are recognized and resolved against the current topic instead of being treated as new, unrelated, unmatched questions |
| Consistent responses regardless of language | Every intent has a human-reviewed response in all 4 languages (`data/multilingual_kb.json`) — the same underlying intent-matching logic runs regardless of detected language, so the same question gets equivalent information in any supported language |

## Demo flow (see screenshots/)
1. `01_initial.png` — fresh chat
2. `02_english_order.png` — English order-status question
3. `03_spanish_shipping.png` — Spanish shipping-time question
4. `04_french_refund.png` — French refund question
5. `05_hindi_support_hours.png` — Hindi support-hours question
6. `06_mixed_language.png` — mixed English/Spanish in one message, still correctly resolved to `order_status`
7. `07_ambiguous_clarify.png` — a genuinely unmatched message triggers a clarifying question instead of a guess

## A bug I found and fixed while building this
Cross-lingual keyword matching initially scored intents by simple hit-count, which meant *"Hola, where is my order?"* tied between `greeting` (matched "hola") and `order_status` (matched "where is my order") and broke the tie arbitrarily — landing on the wrong intent. Fixed by weighting matches by phrase length (word count) instead of a flat +1 per hit, so longer/more specific phrase matches outrank short greeting words. I also found a real coverage gap this same way: *"Et la livraison?"* (French, "And the delivery?") didn't match `shipping_time` because the keyword list only had full phrases like "délai de livraison", not the standalone word "livraison" — it fell through to the generic follow-up mechanism instead. Both were caught by actually running test conversations through the code, not just reading it, and fixed by adding standalone keyword variants to `data/multilingual_kb.json`.

## Run it
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Why a curated KB instead of a general-purpose neural MT model
Real neural translation models (e.g. via Argos Translate, which is fully
open-source) need to download model weight files at install time. In the
sandboxed environment this was built in, Argos Translate's package *index*
is reachable, but the actual model file downloads return `403 Forbidden` —
verified by actually attempting the download, not assumed. Rather than
leave translation unimplemented, this uses a curated, human-reviewed
multilingual intent knowledge base instead of live machine translation,
which is arguably *more* reliable for a customer-support bot anyway (no
risk of a general-purpose translator mistranslating a policy detail like a
refund window). `language_id.py`'s `detect_language()` function is the
integration point where a real MT/translation call would slot in if you
want to extend this to fully open-ended (non-FAQ) multilingual
conversation on a machine with unrestricted internet access.

## Limitations / next steps
- Intent matching is keyword-based, not a trained multilingual NLU model —
  transparent and fast, but less robust to novel phrasing than a trained
  classifier (e.g. a multilingual sentence embedding model) would be.
- Only 8 intents are covered end-to-end in all 4 languages; extending
  coverage means adding more entries to `data/multilingual_kb.json`.
- `langdetect` can be unreliable on very short messages (a known
  limitation of statistical language ID) — the mixed-language chunking
  mitigates this somewhat, but very short single-word messages in a
  low-resource language may be misdetected.
