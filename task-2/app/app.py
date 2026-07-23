"""
Task 2: Multi-Modal AI Assistant
"""
import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
# API Setup 
GOOGLE_API_KEY = st.secrets.get("AQ.Ab8RN6LbOBGQ0ilvONi05iLc2OtnxQdMvDio4-W6-XEncMioRw", os.getenv("AQ.Ab8RN6LbOBGQ0ilvONi05iLc2OtnxQdMvDio4-W6-XEncMioRw", ""))
if not GOOGLE_API_KEY:
    st.error(
        "No Gemini API key found. Add GOOGLE_API_KEY to your Streamlit Cloud "
        "'Secrets' (Manage app > Settings > Secrets) or to a local .env file."
    )
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)
# Models 
#gemini-1.5-flash has been fully shut down by Google as of 2026.
#gemini-flash-latest is an auto-updating alias that currently points to
#Gemini 3.5 Flash (GA) - it will keep working as Google upgrades models
#behind the scenes, so you won't need to touch this again for a while.
vision_model = genai.GenerativeModel("gemini-flash-latest")
text_model   = genai.GenerativeModel("gemini-flash-latest")
#Page Config 
st.set_page_config(
    page_title="Multi-Modal AI Assistant",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 Multi-Modal AI Assistant")
st.caption("Understands both text and images — with full conversation memory")
#Session State 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role", "content", "has_image"}
if "last_image" not in st.session_state:
    st.session_state.last_image = None
#Sidebar 
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Assistant Mode", ["Customer Service", "General Assistant", "Image Analyst"])
    confidence_check = st.toggle("Enable Response Validation", value=True)
    st.divider()
    st.header("📁 Upload Image")
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.session_state.last_image = image
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.last_image = None
        st.rerun()
#Helper: Build system prompt based on mode 
def get_system_prompt(mode):
    prompts = {
        "Customer Service": (
            "You are a helpful customer service assistant. "
            "Be polite, concise, and solution-focused. "
            "If unsure, acknowledge ambiguity and ask for clarification."
        ),
        "General Assistant": (
            "You are a smart general-purpose AI assistant. "
            "Reason carefully, handle ambiguous questions by asking clarifying questions, "
            "and always provide evidence-based responses."
        ),
        "Image Analyst": (
            "You are an expert image analyst. "
            "Analyze images thoroughly, extract all relevant information, "
            "describe visual content in detail, and reason about what you see."
        ),
    }
    return prompts.get(mode, prompts["General Assistant"])
#Helper: Validate response for quality 
def validate_response(response_text):
    issues = []
    if len(response_text) < 20:
        issues.append("Response seems too short.")
    vague_phrases = ["I don't know", "I'm not sure", "Cannot determine"]
    for phrase in vague_phrases:
        if phrase.lower() in response_text.lower():
            issues.append(f"Response contains vague phrase: '{phrase}'")
    return issues
#Helper: Build conversation context 
def build_context():
    context = ""
    for msg in st.session_state.chat_history[-6:]:   # last 3 turns
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['content']}\n"
    return context
#Helper: Detect ambiguity in user input 
def is_ambiguous(text):
    ambiguous_words = ["it", "this", "that", "they", "them", "those", "these"]
    words = text.lower().split()
    short_and_vague = len(words) <= 4
    contains_ambiguous = any(w in ambiguous_words for w in words)
    return short_and_vague and contains_ambiguous
#Helper: Generate response 
def generate_response(user_input, image=None):
    system_prompt = get_system_prompt(mode)
    context = build_context()
    # Ambiguity check
    if is_ambiguous(user_input) and not image:
        return (
            "I want to make sure I understand your question correctly. "
            "Could you provide more details? For example, what specifically are you referring to?"
        ), []
    # Build full prompt
    full_prompt = f"""{system_prompt}
Conversation so far:
{context}
User's new message: {user_input}
Instructions:
- Maintain context from the conversation history above.
- If the question is ambiguous, ask for clarification.
- Provide a well-reasoned, evidence-based response.
- Be specific and helpful.
Response:"""
    try:
        if image:
            response = vision_model.generate_content([full_prompt, image])
        else:
            response = text_model.generate_content(full_prompt)
        response_text = response.text
        # Validate response
        issues = validate_response(response_text) if confidence_check else []
        return response_text, issues
    except Exception as e:
        return f"Error generating response: {str(e)}", []
# Display Chat History 
st.subheader("💬 Conversation")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("has_image"):
            st.caption("📎 Image was attached to this message")
# Chat Input 
user_input = st.chat_input("Ask anything — with or without an image...")
if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)
        if st.session_state.last_image and uploaded_file:
            st.caption("📎 Image attached")
    # Add to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "has_image": st.session_state.last_image is not None
    })
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text, issues = generate_response(
                user_input,
                image=st.session_state.last_image if uploaded_file else None
            )
        st.write(response_text)
        # Show validation warnings
        if issues:
            with st.expander("⚠️ Response Validation Notes"):
                for issue in issues:
                    st.warning(issue)
        # Clear image after use
        if uploaded_file:
            st.session_state.last_image = None
    # Add assistant response to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_text,
        "has_image": False
    })
# Footer
st.divider()
st.caption(f"Mode: **{mode}** | Messages in memory: **{len(st.session_state.chat_history)}** | Response Validation: **{'On' if confidence_check else 'Off'}**")
