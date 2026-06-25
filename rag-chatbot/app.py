"""
Pakistani University Admissions RAG Chatbot
=============================================
A Retrieval-Augmented Generation chatbot that answers questions about
admissions at NUST, LUMS, and Riphah International University using
real document content and the Groq API (cloud LLM - no local install needed).

Author: Ayesha Abbasi
"""

import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Pakistani University Admissions Assistant",
    page_icon=":mortar_board:",
    layout="wide",
)

# --------------------------------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0A192F;
        margin-bottom: 0;
    }
    .sub-header {
        color: #64748B;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    .university-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        background-color: #E0F7FA;
        color: #0284C7;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown('<p class="main-header">Pakistani University Admissions Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ask questions about admissions at NUST, LUMS, and Riphah International University - '
    'answered directly from official admission documents using RAG (Retrieval-Augmented Generation).</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<span class="university-badge">NUST</span>'
    '<span class="university-badge">LUMS</span>'
    '<span class="university-badge">Riphah International University</span>',
    unsafe_allow_html=True
)
st.divider()

# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
# Try to load a pre-configured key from Streamlit secrets (set by the owner).
# Visitors then don't need their own key at all.
try:
    default_api_key = st.secrets["GROQ_API_KEY"]
    key_preconfigured = True
except (KeyError, FileNotFoundError):
    default_api_key = ""
    key_preconfigured = False

with st.sidebar:
    st.header("Settings")

    if key_preconfigured:
        st.success("Running on the owner's API key - no setup needed!")
        api_key = default_api_key
        with st.expander("Use your own Groq API key instead"):
            custom_key = st.text_input(
                "Your Groq API Key",
                type="password",
                help="Get a free key at console.groq.com",
            )
            if custom_key:
                api_key = custom_key
    else:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get a free key at console.groq.com - no credit card needed.",
        )

    st.markdown("---")
    st.subheader("Knowledge Base")
    st.markdown("""
    This chatbot has been pre-loaded with admission information for:
    - **NUST** - Eligibility, NET, merit criteria, fees
    - **LUMS** - Eligibility, LCAT/SAT/ACT, financial aid
    - **Riphah** - BSCS eligibility, fee structure, campuses
    """)

    st.markdown("---")
    st.subheader("Try asking")
    sample_questions = [
        "What is the minimum percentage required for NUST?",
        "Does LUMS accept SAT scores?",
        "What is the fee for BSCS at Riphah?",
        "How do I apply for financial aid at LUMS?",
        "What is the NUST Entry Test (NET)?",
        "Compare admission criteria between NUST and Riphah",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")
    st.caption("Built by Ayesha Abbasi | Powered by LangChain + Groq + FAISS")

# --------------------------------------------------------------------------
# RAG PIPELINE SETUP (cached so it only runs once)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    """Load all university documents, chunk them, and build a FAISS vector store."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    all_docs = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(data_dir, filename), encoding="utf-8")
            docs = loader.load()
            university = filename.replace("_admissions.txt", "").upper()
            for doc in docs:
                doc.metadata["university"] = university
            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


@st.cache_resource(show_spinner=False)
def get_qa_chain(_vectorstore, api_key):
    """Build the RAG chain using Groq's free, fast LLM API (modern LCEL pattern)."""
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
    )

    prompt_template = """You are a helpful assistant answering questions about university admissions in Pakistan.
Use ONLY the context below to answer the question. If the answer isn't in the context, say
"I don't have that information in my knowledge base - please check the official university website."
Always mention which university the information is about. Be specific with numbers, fees, and dates when available.

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# --------------------------------------------------------------------------
# MAIN APP LOGIC
# --------------------------------------------------------------------------
if not api_key:
    st.info("Please enter your free Groq API key in the sidebar to start chatting. "
            "Get one in 2 minutes at **console.groq.com**")
    st.stop()

with st.spinner("Loading knowledge base... (first time only, ~10 seconds)"):
    vectorstore = load_knowledge_base()
    rag_chain, retriever = get_qa_chain(vectorstore, api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can answer your questions about admissions at NUST, LUMS, "
                                          "and Riphah International University. What would you like to know?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Ask about NUST, LUMS, or Riphah admissions...")
question = pending or user_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching admission documents..."):
            try:
                sources = retriever.invoke(question)
                answer = rag_chain.invoke(question)

                st.markdown(answer)

                if sources:
                    unique_unis = list(set(doc.metadata.get("university", "Unknown") for doc in sources))
                    st.caption(f"Sources: {', '.join(unique_unis)}")

                    with st.expander("View retrieved passages"):
                        for i, doc in enumerate(sources, 1):
                            st.markdown(f"**Passage {i} - {doc.metadata.get('university', 'Unknown')}**")
                            st.text(doc.page_content[:300] + "...")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"Something went wrong: {str(e)}. Please check your API key is valid."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.divider()
st.caption(
    "This chatbot provides information for general guidance only based on documents in its knowledge base. "
    "Always verify critical admission details (deadlines, fees, eligibility) on official university websites "
    "before making decisions."
)
