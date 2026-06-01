# HerRight

A multilingual AI assistant that helps women in India understand their legal rights,
safety options, and available support services — in their own language, including voice.

---

## The Problem

Millions of women in India, especially in tier 2 and tier 3 cities, face domestic
violence, workplace harassment, cyber crime, and other rights violations every day.
The information they need exists — in government acts, NCW guidelines, scheme documents
— but it is written in legal language, available only in English, and impossible to
navigate without a lawyer.

HerRight bridges that gap.

---

## What It Does

- Answers questions about legal rights in plain, simple language
- Supports voice input and voice output in Hindi and other Indian languages
- Cites official government sources for every answer
- Provides emergency helpline numbers at all times
- Works in 7 Indian languages: English, Hindi, Tamil, Telugu, Bengali, Kannada, Marathi

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 2.5 Flash (Google AI) |
| RAG Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Speech to Text | Sarvam AI (saarika:v2.5) |
| Text to Speech | Sarvam AI (bulbul:v3) |
| UI | Streamlit |

---

## Knowledge Base

HerRight is grounded in official Indian government documents:

| Document | Source |
|---|---|
| Protection of Women from Domestic Violence Act, 2005 | [NCW](https://cdn.ncw.gov.in/wp-content/uploads/2023/06/TheProtectionofWomenfromDomesticViolenceAct2005_0.pdf) |
| POCSO Act, 2012 | [India Code](https://www.indiacode.nic.in/bitstream/123456789/2079/1/201332.pdf) |
| Dowry Prohibition Act, 1961 | [NCW](https://cdn.ncw.gov.in/wp-content/uploads/2023/06/THEDOWRYPROHIBITIONACT1961_0.pdf) |
| Sexual Harassment of Women at Workplace Act, 2013 | [NCW](https://cdn.ncw.gov.in/wp-content/uploads/2023/06/SexualHarassmentofWomenatWorkPlaceAct2013_0.pdf) |
| Criminal Law Amendment Act, 2013 | [NCW](https://cdn.ncw.gov.in/wp-content/uploads/2023/06/The_Criminal_Law_Amendment_Act_2013_0.pdf) |
| NCW Laws Relating to Women (booklet) | [NCW](https://cdn.ncw.gov.in/wp-content/uploads/2023/01/Booklet-Laws-relating-to-Women_0.pdf) |
| Mission Shakti Guidelines | [WCD](https://missionshakti.wcd.gov.in/public/documents/whatsnew/Mission_Shakti_Guidelines.pdf) |
| One Stop Centre Scheme | [WCD](https://www.spniwcd.wcd.gov.in/uploads/pdf/1710073241_7VcC8VHeZ2.pdf) |
| NCW Annual Report 2023-24 | [NCW](https://cdn.ncw.gov.in/wp-content/uploads/2025/03/NCWAnnualReport20232024Eng.pdf) |
| Official Helplines & Contacts | [WCD](https://wcd.gov.in/) |
| Cyber Crime Reporting Steps | [Cyber Crime Portal](https://cybercrime.gov.in/) |
---

## Project Structure

```
HerRight/
├── app.py                  # Streamlit UI
├── src/
│   ├── rag_pipeline.py     # Document loading, FAISS vector store, RAG chain
│   └── voice_agent.py      # Sarvam STT and TTS
├── data/
│   ├── docs/
│   │   ├── criminal_law_amendment_act.pdf
│   │   ├── cybercrime_reporting_steps.txt
│   │   ├── domestic_violence_act.pdf
│   │   ├── dowry_prohibition_act.pdf
│   │   ├── helplines_contacts.txt
│   │   ├── mission_shakti_guidelines.pdf
│   │   ├── ncw_annual_report.pdf
│   │   ├── ncw_laws_relating_to_women.pdf
│   │   ├── one_stop_centre_scheme.pdf
│   │   ├── pocso_act.pdf
│   │   └── sexual_harassment_workplace_act.pdf
│   └── vectorstore/        # FAISS index (auto-generated on first run)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/HerRight.git
cd HerRight
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```
GOOGLE_API_KEY=your_gemini_api_key        # aistudio.google.com
SARVAM_API_KEY=your_sarvam_api_key        # dashboard.sarvam.ai
```

Both keys are free to obtain.

### 4. Add knowledge base documents

Place your PDF and text files in `data/docs/`. The vector store is built
automatically on first run.

### 5. Run the app

```bash
streamlit run app.py
```

The vector store will be built on first launch (takes 1-2 minutes).
Subsequent launches load it instantly.

---

## Emergency Helplines

| Helpline | Number / Link |
|---|---|
| Women Helpline | 181 |
| National Emergency | 112 |
| Women Police Helpline | 1091 |
| Child Helpline | 1098 |
| Cyber Crime Portal | cybercrime.gov.in |
| NCW Online Complaint | ncwapps.nic.in |

---

## Disclaimer

HerRight provides legal information, not legal advice. For emergencies, call 181
immediately. All information is sourced from official Indian government documents.