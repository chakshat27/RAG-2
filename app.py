# app.py
import os
import shutil
from pathlib import Path

import streamlit as st
from PyPDF2 import PdfReader

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from InstructorEmbedding import INSTRUCTOR


# 🤗 Hugging Face imports
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain import HuggingFaceHub

# ─── Configuration ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="PDF Q&A – HuggingFace", layout="wide")
st.title("📄 PDF Q&A with Hugging Face")

# 1) Your Hugging Face token (get one free at https://huggingface.co/settings/tokens)
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("❗️ Please set your `HF_TOKEN` in Streamlit Secrets or .env")
    st.stop()

# 2) Embedding model (Instructor XL is free on HF)
embeddings_model = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl",
    model_kwargs={"device": "cpu"},
)

# 3) Text-generation model (Flan-T5-XL via HuggingFaceHub)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-xl",
    model_kwargs={"temperature": 0, "max_length": 512},
    huggingfacehub_api_token=HF_TOKEN,
)

# 4) Prompt template
prompt = PromptTemplate.from_template(
    """You are a helpful assistant. Use the following context to answer the question as accurately as possible.

    Context:
    {context}

    Question:
    {question}

    Answer:"""
)

# ─── Helpers ────────────────────────────────────────────────────────────────────

def save_pdf(uploaded_file):
    save_dir = Path("data")
    save_dir.mkdir(exist_ok=True)
    path = save_dir / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def load_and_split(path: Path):
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def build_faiss_index(docs):
    db_path = Path("vectorstore/faiss_index")
    if db_path.exists():
        shutil.rmtree(db_path)
    vectordb = FAISS.from_documents(docs, embeddings_model)
    vectordb.save_local(str(db_path))
    return vectordb

def make_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="history", input_key="question")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )

# ─── Streamlit App ────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
query = st.text_input("Enter your question about the document")

if st.button("Get Answer"):
    if not uploaded_file or not query:
        st.warning("📌 Please upload a PDF and enter a query.")
    else:
        # 1) Save and split
        pdf_path = save_pdf(uploaded_file)
        docs = load_and_split(pdf_path)

        # 2) Build / load FAISS index
        vectordb = build_faiss_index(docs)

        # 3) Create QA chain and run
        qa = make_qa_chain(vectordb)
        answer = qa.run(query)

        # 4) Display
        st.markdown("### 💬 Answer")
        st.write(answer)

