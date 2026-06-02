"""
HerRight - Multilingual legal rights assistant for women in India.
Supports text and voice input/output via Sarvam AI.
"""

import streamlit as st
import sys
sys.path.append(".")

from langchain_core.messages import HumanMessage, AIMessage
from src.rag_pipeline import get_rag_chain
from src.voice_agent import speech_to_text, text_to_speech

st.set_page_config(
    page_title="HerRight",
    page_icon="⚖",
    layout="centered"
)

st.markdown("""
<style>
    .emergency-banner {
        background-color: #c0392b;
        padding: 12px 16px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 8px;
    }
    div[data-testid="stChatMessage"] {
        padding: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

LANGUAGE_OPTIONS = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Bengali": "bn-IN",
    "Kannada": "kn-IN",
    "Marathi": "mr-IN",
}

SOURCE_LINKS = {
    "domestic_violence_act.pdf": (
        "Domestic Violence Act",
        "https://cdn.ncw.gov.in/wp-content/uploads/2023/06/TheProtectionofWomenfromDomesticViolenceAct2005_0.pdf"
    ),
    "pocso_act.pdf": (
        "POCSO Act",
        "https://www.indiacode.nic.in/bitstream/123456789/2079/1/201332.pdf"
    ),
    "ncw_laws_relating_to_women.pdf": (
        "NCW Laws Relating to Women",
        "https://cdn.ncw.gov.in/wp-content/uploads/2023/01/Booklet-Laws-relating-to-Women_0.pdf"
    ),
    "dowry_prohibition_act.pdf": (
        "Dowry Prohibition Act",
        "https://cdn.ncw.gov.in/wp-content/uploads/2023/06/THEDOWRYPROHIBITIONACT1961_0.pdf"
    ),
    "sexual_harassment_workplace_act.pdf": (
        "Sexual Harassment at Workplace Act",
        "https://cdn.ncw.gov.in/wp-content/uploads/2023/06/SexualHarassmentofWomenatWorkPlaceAct2013_0.pdf"
    ),
    "criminal_law_amendment_act.pdf": (
        "Criminal Law Amendment Act",
        "https://cdn.ncw.gov.in/wp-content/uploads/2023/06/The_Criminal_Law_Amendment_Act_2013_0.pdf"
    ),
    "mission_shakti_guidelines.pdf": (
        "Mission Shakti Guidelines",
        "https://missionshakti.wcd.gov.in/public/documents/whatsnew/Mission_Shakti_Guidelines.pdf"
    ),
    "one_stop_centre_scheme.pdf": (
        "One Stop Centre Scheme",
        "https://www.spniwcd.wcd.gov.in/uploads/pdf/1710073241_7VcC8VHeZ2.pdf"
    ),
    "ncw_annual_report.pdf": (
        "NCW Annual Report 2023-24",
        "https://cdn.ncw.gov.in/wp-content/uploads/2025/03/NCWAnnualReport20232024Eng.pdf"
    ),
    "helplines_contacts.txt": (
        "Official Helplines",
        "https://wcd.gov.in/"
    ),
    "cybercrime_reporting_steps.txt": (
        "Cyber Crime Portal",
        "https://cybercrime.gov.in/"
    ),
}

STARTER_QUESTIONS = [
    "What are my rights if my husband is violent?",
    "How do I report cyber harassment or misuse of my photos?",
    "मुझे मुफ्त वकील कैसे मिलेगा?",
    "One Stop Centre क्या है और यह कैसे मदद कर सकता है?",
]


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_chain():
    """Load and cache the RAG chain."""
    return get_rag_chain()


def render_sources(sources: list):
    """Render clickable source links below an answer."""
    if not sources:
        return
    links = []
    for filename in sources:
        if filename in SOURCE_LINKS:
            display_name, url = SOURCE_LINKS[filename]
            links.append(f"[{display_name}]({url})")
    if links:
        st.caption("Sources: " + " | ".join(links))


def process_input(chain, user_input: str, language_code: str):
    """
    Handle user input: build history, invoke chain, store result.
    Audio is rendered in the chat history loop via pending_audio.
    """
    st.session_state.messages.append({"role": "user", "content": user_input})

    chat_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    try:
        response = chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
    except Exception:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I am experiencing high demand right now. Please try again in a minute, or call 181 for immediate support.",
            "sources": []
        })
        return

    answer = response["answer"]

    context_docs = response.get("context", [])
    seen = set()
    sources = []
    for doc in context_docs:
        filename = doc.metadata.get("source", "").split("/")[-1]
        if filename and filename not in seen:
            seen.add(filename)
            sources.append(filename)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.session_state["pending_audio"] = (answer, language_code)

chain = load_chain()

# --- Sidebar ---
with st.sidebar:
    st.title("HerRight")
    st.caption("Know your rights. Raise your voice. Stay safe.")
    st.markdown("---")

    selected_language = st.selectbox(
        "Language / भाषा",
        options=list(LANGUAGE_OPTIONS.keys())
    )
    language_code = LANGUAGE_OPTIONS[selected_language]

    st.markdown("---")
    st.markdown("**Emergency Helplines**")
    st.markdown("""
- **181** — Women Helpline
- **112** — National Emergency
- **1091** — Women Police Helpline
- **1098** — Child Helpline
- **[cybercrime.gov.in](https://cybercrime.gov.in)** — Cyber Crime
- **[ncwapps.nic.in](https://ncwapps.nic.in)** — NCW Complaint
    """)

    st.markdown("---")
    st.markdown("**About HerRight**")
    st.caption(
        "HerRight is a free AI assistant that helps women in India "
        "understand their legal rights and safety options in their own language."
    )

    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("last_audio", None)
        st.session_state.pop("pending_audio", None)
        st.rerun()

# --- Main Area ---
st.markdown("""
<div class='emergency-banner'>
Emergency? Call 181 (Women Helpline) or 112 (National Emergency) right now.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Starter questions — only on empty chat
if not st.session_state.messages:
    st.markdown("**Not sure where to start? Try asking:**")
    cols = st.columns(2)
    for i, question in enumerate(STARTER_QUESTIONS):
        if cols[i % 2].button(question, use_container_width=True):
            st.session_state["starter_input"] = question
            st.rerun()

# Handle starter question before rendering history
if "starter_input" in st.session_state:
    question = st.session_state.pop("starter_input")
    process_input(chain, question, language_code)
    st.rerun()

# --- Chat history ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if i == len(st.session_state.messages) - 1 and "pending_audio" in st.session_state:
                answer_text, lang_code = st.session_state.pop("pending_audio")
                with st.spinner("Generating audio..."):
                    try:
                        audio_bytes = text_to_speech(answer_text, lang_code)
                        st.audio(audio_bytes, format="audio/wav")
                    except Exception:
                        st.caption("Audio unavailable for this response.")
            if message.get("sources"):
                render_sources(message["sources"])

# --- Input tabs ---
tab_text, tab_voice = st.tabs(["Type your question", "Speak your question"])

with tab_text:
    user_input = st.text_area(
        "Your question",
        placeholder="Type your question here...",
        height=100,
        label_visibility="collapsed"
    )
    if st.button("Send", use_container_width=True) and user_input.strip():
        process_input(chain, user_input, language_code)
        st.rerun()

with tab_voice:
    audio_input = st.audio_input(
        "Record your question",
        label_visibility="collapsed"
    )
    if audio_input is not None and (
        st.session_state.get("last_audio") != audio_input.getvalue()
    ):
        st.session_state["last_audio"] = audio_input.getvalue()
        with st.spinner("Transcribing..."):
            transcribed = speech_to_text(audio_input.getvalue(), language_code)
        if not transcribed or not transcribed.strip():
            st.warning("Could not transcribe. Please try speaking again.")
        else:
            st.success(f"You said: {transcribed}")
            process_input(chain, transcribed, language_code)
            st.rerun()

# --- Disclaimer ---
st.markdown("---")
st.caption(
    "HerRight provides legal information, not legal advice. "
    "For emergencies, call 181 immediately. "
    "Information is based on official Indian government sources."
)