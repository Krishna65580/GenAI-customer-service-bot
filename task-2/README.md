# Multi-Modal AI Assistant (Text + Image Reasoning)

Built to satisfy:
> "Develop a multi-modal AI assistant capable of understanding and reasoning
> over both text and image inputs. The assistant should analyze visual
> content, extract relevant information, maintain conversational context
> across multiple interactions, and generate evidence-based responses...
> demonstrate contextual reasoning, ambiguity handling, response validation,
> and intelligent decision-making rather than simple model inference."

## Why classical CV instead of a vision-language model
This was built in a sandboxed environment with no internet access to
download vision-language model weights (e.g. LLaVA, CLIP-based captioners)
or call a cloud vision API. Rather than fake image understanding, the app
extracts **real, verifiable visual evidence** — OCR text, color/brightness
statistics, and a complexity proxy — and grounds every response in that
actual evidence. `app/llm_backend.py` auto-detects a local Ollama LLM (which
can be multi-modal, e.g. `llama3.2-vision`, or text-only) to phrase answers
more fluently around the same measured evidence, if one is available.

## How each requirement is implemented
| Requirement | Implementation |
|---|---|
| Analyze visual content | `vision_utils.analyze_image()`: OCR (pytesseract), color profile, brightness/contrast, edge-density complexity |
| Extract relevant information | `describe_evidence()` turns raw measurements into a factual evidence string used as the only source of truth for image claims |
| Maintain conversational context | `reasoning.ConversationMemory`: rolling turn history + "active image" state persisted across turns in `st.session_state` |
| Evidence-based responses | `compose_answer()` only asserts what's in the evidence description or conversation history |
| Contextual reasoning | Answers reference the currently attached image and prior turns, not just the latest message in isolation |
| Ambiguity handling | `is_ambiguous()` intercepts vague queries ("what is this", "explain") when there's no image/context to resolve them, and asks a clarifying question instead of guessing |
| Response validation | `validate_response()` re-checks the drafted answer against the evidence (e.g. catches claims about image text when OCR found none) and flags/softens unsupported claims |
| Intelligent decision-making | The app branches between three paths — clarify / image-grounded answer / text-only answer — based on what's actually available, rather than always doing one fixed thing |

## Demo flow (see screenshots/)
1. `01_initial.png` — fresh chat, no image attached yet.
2. `02_ambiguity_handling.png` — asking "what is this" with no image attached triggers a clarifying question instead of a guess.
3. `03_image_uploaded.png` — a sample image (generated text-on-background test image) is uploaded via the sidebar.
4. `04_image_attached_evidence.png` — after attaching, the extracted evidence (OCR text, brightness, dominant color, etc.) is shown in the chat and the sidebar's "raw evidence" expander.
5. `05_evidence_grounded_answer.png` — asking "what text is in this image?" returns an answer grounded specifically in the OCR result.

## Run it
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
Tesseract OCR must be installed on the system (`sudo apt install tesseract-ocr` on Debian/Ubuntu, or the Windows installer from the Tesseract project) for text extraction to work; without it, the app still runs and gracefully reports "no text detected."

## Enabling a real (multi-modal) open-source LLM
1. Install [Ollama](https://ollama.com/download)
2. `ollama pull llama3.2-vision` (or any text model for non-visual phrasing)
3. `ollama serve`
4. Launch the app — `llm_backend.py` auto-detects Ollama and routes answer
   composition through it, still constrained to the extracted evidence.
