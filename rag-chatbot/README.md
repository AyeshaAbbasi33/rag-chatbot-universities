#  Pakistani University Admissions RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers natural-language questions about university admissions in Pakistan — built end-to-end with LangChain, FAISS, and Groq's free LLM API. No local model installation required; runs entirely on cloud inference.

**[Live Demo](https://rag-chatbot-universities-mxpgswcvktoc3bq2qiuhjj.streamlit.app/)** *(add your Streamlit Cloud link here after deploying)*

![Status](https://img.shields.io/badge/status-live-success)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)

![App Screenshot](screenshots/picture3.png)




## What This Project Does

Ask questions like:
- *"What is the minimum percentage required for NUST?"*
- *"Does LUMS accept SAT scores for admission?"*
- *"What is the BSCS fee structure at Riphah?"*
- *"Compare NUST and LUMS admission criteria"*

The chatbot retrieves the most relevant passages from real admission documents (NUST, LUMS, and Riphah International University) and generates a grounded, accurate answer — instead of hallucinating from general knowledge.



## Why I Built This

Most "ChatGPT wrapper" projects just call an LLM directly and hope for the best. This project demonstrates **Retrieval-Augmented Generation (RAG)** — the technique that makes LLMs reliable for domain-specific, factual question answering. It's the same core architecture used in enterprise AI assistants and customer support bots in production today.

I chose Pakistani university admissions specifically because:
1. It's a real problem because  thousands of students struggle to find accurate, consolidated admission info every year.
2. I understand the domain personally as a current CS student going through this system.
3. It let me work with real-world messy data (different fee structures, eligibility rules, and formats across universities).



## How It Works (Architecture)

```
Admission Docs (.txt)  →  Chunking (LangChain)  →  Embeddings (MiniLM-L6-v2)
                                                              ↓
User Question  →  Embed Question  →  FAISS Similarity Search  ←  Vector Store
                                              ↓
                                    Top-K Relevant Chunks
                                              ↓
                            Chunks + Question → Groq LLM (Llama 3.1)
                                              ↓
                                  Grounded Answer + Source Citation
```

### The Pipeline, Step by Step

1. **Document Loading** — Admission information for NUST, LUMS, and Riphah is loaded from structured `.txt` files (each tagged with its source university).
2. **Chunking** — Documents are split into ~600-character overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`, preserving context across chunk boundaries.
3. **Embedding** — Each chunk is converted into a vector using `sentence-transformers/all-MiniLM-L6-v2` — a fast, free, CPU-friendly embedding model.
4. **Vector Storage** — All embeddings are stored in a **FAISS** index for fast similarity search.
5. **Retrieval** — When a user asks a question, it's embedded and compared against the FAISS index to find the 4 most relevant chunks.
6. **Generation** — The retrieved chunks + the original question are passed to **Groq's Llama 3.1 8B Instant** model via a custom prompt template that forces the model to answer only from the provided context — preventing hallucination.
7. **Citation** — The app displays which university's documents were used to generate the answer, with an expandable view of the raw retrieved passages.



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

### 1. Clone the repository
```bash
git clone https://github.com/AyeshaAbbasi33/rag-chatbot-universities.git
cd rag-chatbot-universities
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
- Go to [console.groq.com](https://console.groq.com)
- Sign up (free, no credit card)
- Create an API key
- or you can ask few questions for on the owners API Key

### 4. Run the app
```bash
streamlit run app.py
```
Paste your Groq API key into the sidebar when the app opens in your browser.



## Project Structure

```
rag-chatbot-universities/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── data/
│   ├── nust_admissions.txt         # NUST admission knowledge base
│   ├── lums_admissions.txt         # LUMS admission knowledge base
│   └── riphah_admissions.txt       # Riphah admission knowledge base
├── screenshots/
│   └── picture 1.png
    ├── picture 2.png
    └── picture 3.png
 # App screenshot
   ├── .gitignore
   └── README.md
```



## What I Learned Building This

- **RAG fundamentals**: how retrieval grounds an LLM's answers and prevents hallucination on domain-specific facts
- **Chunking strategy matters**: smaller chunks with overlap retrieve more precisely than dumping whole documents
- **Prompt engineering for groundedness**: explicitly instructing the LLM to say "I don't know" when context is insufficient, rather than guessing
- **Cloud-first architecture**: building an LLM app with zero local model downloads — useful for resource-constrained environments
- **Source attribution**: showing users which document an answer came from builds trust and lets them verify



## Future Improvements

- [ ] Add more universities (FAST-NUCES, GIKI, COMSATS)
- [ ] Allow users to upload their own university prospectus PDF
- [ ] Add conversational memory so follow-up questions work ("What about its fee?")
- [ ] Add a feedback mechanism (thumbs up/down) to track answer quality
- [ ] Multilingual support (Urdu queries)



## Author

**Ayesha Abbasi**
CS Student @ Riphah International University, Islamabad
[LinkedIn](https://linkedin.com/in/ayesha-abbasi-3215a7321) · [GitHub](https://github.com/AyeshaAbbasi33)




