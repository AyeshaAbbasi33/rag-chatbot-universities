# Pakistani University Admissions RAG Chatbot
A Retrieval-Augmented Generation chatbot that answers natural-language questions about university admissions in Pakistan, built with LangChain, FAISS, and Groq's free LLM API. No local model installation required plus it runs entirely on cloud inference.

[Live Demo](https://rag-chatbot-universities-mxpgswcvktoc3bq2qiuhjj.streamlit.app/)

![Status](https://img.shields.io/badge/status-live-success)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)

screenshots
<img width="800" height="400" alt="picture 3" src="https://github.com/user-attachments/assets/fbf5d970-12e9-4d56-93b6-f8abcde10664" />
<img width="800" height="400" alt="picture 2" src="https://github.com/user-attachments/assets/2043fa79-3a5b-437f-a64f-37a04b29534c" />




## What This Project Does

Ask it things like:
- *"What is the minimum percentage required for NUST?"*
- *"Does LUMS accept SAT scores for admission?"*
- *"What is the BSCS fee structure at Riphah?"*
- *"Compare NUST and LUMS admission criteria"*

The chatbot retrieves the most relevant passages from real admission documents for NUST, LUMS, and Riphah International University, then generates a grounded answer instead of hallucinating from general knowledge.

## Why I Built This

Most "ChatGPT wrapper" projects just call an LLM directly and hope for the best. This project demonstrates Retrieval-Augmented Generation (RAG), the technique that makes LLMs reliable for domain-specific, factual question answering. It's the same core architecture used in enterprise AI assistants and customer support bots in production today.

I picked Pakistani university admissions specifically because it's a real problem I understand personally as a current CS student going through this system, and because it forced me to work with messy real-world data: different fee structures, eligibility rules, and formats across universities that don't line up neatly.

## How It Works
Admission Docs (.txt) -> Chunking (LangChain) -> Embeddings (MiniLM-L6-v2)
|
User Question -> Embed Question -> FAISS Similarity Search <- Vector Store
|
Top-K Relevant Chunks
|
Chunks + Question -> Groq LLM (Llama 3.1)
|
Grounded Answer + Source Citation


Admission information for NUST, LUMS, and Riphah is loaded from structured `.txt` files, each tagged with its source university. Documents are split into roughly 600-character overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`, which preserves context across chunk boundaries. Each chunk is converted into a vector using `sentence-transformers/all-MiniLM-L6-v2`, a fast, free, CPU-friendly embedding model, and all embeddings are stored in a FAISS index for similarity search.

When a user asks a question, it's embedded and compared against the FAISS index to find the four most relevant chunks. Those chunks plus the original question are passed to Groq's Llama 3.1 8B Instant model through a prompt template that forces it to answer only from the provided context, which is what prevents hallucination. The app then displays which university's documents were used to generate the answer, with an expandable view of the raw retrieved passages.

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM | Groq API (Llama 3.1 8B Instant) | Free tier, extremely fast inference, no local GPU/storage needed |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Lightweight (~80MB), runs on CPU, no API cost |
| Vector Store | FAISS | Industry-standard, fast similarity search, runs locally |
| Orchestration | LangChain | Standard framework for RAG pipelines |
| Frontend | Streamlit | Fast to build, free hosting on Streamlit Cloud |
| Deployment | Streamlit Community Cloud | Free, zero-config hosting |

## Run It Yourself

Clone the repository:
```bash
git clone https://github.com/AyeshaAbbasi33/rag-chatbot-universities.git
cd rag-chatbot-universities
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card needed, just sign up and create a key.

Run the app:
```bash
streamlit run app.py
```

Paste your Groq API key into the sidebar when the app opens in your browser.

## Project Structure
rag-chatbot-universities/

├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── data/
│   ├── nust_admissions.txt         # NUST admission knowledge base
│   ├── lums_admissions.txt         # LUMS admission knowledge base
│   └── riphah_admissions.txt       # Riphah admission knowledge base

├── screenshots/
  └── picture 1.png                    # App screenshot
    ├── picture 2.png
    └── picture 2.png
├── .gitignore

└── README.md

## What I actually learned doing this

Chunk size matters more than I assumed going in  dumping whole documents into the prompt retrieves worse than smaller overlapping chunks, because the similarity search gets diluted. Getting the model to admit "I don't know" instead of confidently guessing took actual prompt iteration, it doesn't happen by default. And building everything around a free, cloud-hosted LLM instead of a local one taught me that the "proper" RAG setup people show in tutorials (Ollama + local models) isn't actually required to build something real — it's just one option.

## Future Improvements

- Add more universities (FAST-NUCES, GIKI, COMSATS)
- Allow users to upload their own university prospectus PDF
- Add conversational memory so follow-up questions work ("What about its fee?")
- Add a feedback mechanism (thumbs up/down) to track answer quality
- Multilingual support for Urdu queries

## About me

**Ayesha Abbasi**
CS Student @ Riphah International University, Islamabad
[LinkedIn](https://linkedin.com/in/ayesha-abbasi-3215a7321) · [GitHub](https://github.com/AyeshaAbbasi33)

