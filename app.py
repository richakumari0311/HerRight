"""
HerRight - Multilingual legal rights assistant for women in India.
Supports text and voice input/output via Sarvam AI.
"""

import streamlit as st
import sys
sys.path.append(".")

from src.rag_pipeline import get_rag_chain
from src.voice_agent import speech_to_text, text_to_speech

st.set_page_config(
    page_title="HerRight",
    page_icon="⚖",
    layout="centered"
)

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
        "Protection of Women from Domestic Violence Act",
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
        "Sexual Harassment of Women at Workplace Act",
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
        "Cyber Crime Reporting Portal",
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


def handle_query(chain, user_input: str) -> tuple[str, list]:
    """
    Run the RAG chain and return answer with source filenames.

    Returns:
        Tuple of (answer string, list of source filenames)
    """
    if not user_input or not user_input.strip():
        return "I did not receive a question. Please try again.", []

    response = chain.invoke({"input": user_input})
    answer = response["answer"]

    # create_retrieval_chain returns context, not source_documents
    context_docs = response.get("context", [])
    seen = set()
    sources = []
    for doc in context_docs:
        filename = doc.metadata.get("source", "").split("/")[-1]
        if filename and filename not in seen:
            seen.add(filename)
            sources.append(filename)

    return answer, sources


def render_sources(sources: list):
    """Render source links below an answer."""
    if not sources:
        return
    links = []
    for filename in sources:
        if filename in SOURCE_LINKS:
            display_name, url = SOURCE_LINKS[filename]
            links.append(f"[{display_name}]({url})")
    if links:
        st.caption("Sources: " + " | ".join(links))


def render_answer(answer: str, sources: list, language_code: str):
    """Render answer text, audio, and sources."""
    st.markdown(answer)
    with st.spinner("Generating audio response..."):
        try:
            audio_response = text_to_speech(answer, language_code)
            st.audio(audio_response, format="audio/wav")
        except Exception:
            pass
    render_sources(sources)


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
- **181** - Women Helpline
- **112** - National Emergency
- **1091** - Women Police Helpline
- **1098** - Child Helpline
- **cybercrime.gov.in** - Cyber Crime
- **ncwapps.nic.in** - NCW Complaint
    """)

    st.markdown("---")
    st.markdown("**About HerRight**")
    st.markdown(
        "HerRight is a free AI assistant that helps women in India "
        "understand their legal rights and safety options in their own language."
    )

    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("last_audio", None)
        st.rerun()

# --- Main Area ---
st.markdown("""
<div style='background-color:#c0392b;padding:12px;border-radius:8px;color:white;font-weight:bold;'>
Emergency? Call 181 (Women Helpline) or 112 (National Emergency) right now.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Starter questions shown only when chat is empty
if not st.session_state.get("messages"):
    st.markdown("**Not sure where to start? Try asking:**")
    cols = st.columns(2)
    for i, question in enumerate(STARTER_QUESTIONS):
        if cols[i % 2].button(question, use_container_width=True):
            st.session_state.messages = []
            st.session_state["starter_input"] = question
            st.rerun()
    st.markdown("---")

# Chat history
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])

# Handle starter question click
if "starter_input" in st.session_state:
    user_input = st.session_state.pop("starter_input")
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        answer, sources = handle_query(chain, user_input)
        render_answer(answer, sources, language_code)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# --- Input tabs ---
st.markdown("---")
tab_text, tab_voice = st.tabs(["Type your question", "Speak your question"])

with tab_text:
    user_input = st.text_area(
        "Your question",
        placeholder="Type your question here...",
        height=100,
        label_visibility="collapsed"
    )
    if st.button("Send", use_container_width=True) and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            answer, sources = handle_query(chain, user_input)
            render_answer(answer, sources, language_code)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        st.rerun()

with tab_voice:
    audio_input = st.audio_input("Record your questio", label_visibility="collapsed")

    if audio_input is not None and (
        st.session_state.get("last_audio") != audio_input.getvalue()
    ):
        st.session_state["last_audio"] = audio_input.getvalue()
        audio_bytes = audio_input.getvalue()

        with st.spinner("Transcribing..."):
            transcribed = speech_to_text(audio_bytes, language_code)

        if not transcribed or not transcribed.strip():
            st.warning("Could not transcribe audio. Please try speaking again.")
        else:
            st.success(f"You said: {transcribed}")
            st.session_state.messages.append({"role": "user", "content": transcribed})

            with st.chat_message("assistant"):
                answer, sources = handle_query(chain, transcribed)
                render_answer(answer, sources, language_code)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

# --- Disclaimer ---
st.markdown("---")
st.caption(
    "HerRight provides legal information, not legal advice. "
    "For emergencies, call 181 immediately. "
    "Information is based on official Indian government sources."
)