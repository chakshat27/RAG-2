from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
import requests
from huggingface_hub import configure_http_backend
import streamlit as st
from pathlib import Path
import shutil

# fixing the SSL issue
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

# set GEMINI key
os.environ["GOOGLE_API_KEY"] = " "

# embeddings and model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# prompt
prompt_temp = PromptTemplate.from_template(
    """
    You are an intelligent assistant designed to answer questions based on the content of uploaded PDF documents.
    Please provide accurate and helpful answers to the questions asked, using the context provided from the documents.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """
)

prompt_str = prompt_temp.template
prompt = PromptTemplate(template=prompt_str, input_variables=["history", "context", "question"])

# STREAMLIT UI
st.title("File Upload and Query")
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
query = st.text_input("Enter your query")

# loading data
# def load_data(uploaded_file):
#     save_path = os.path.join("./data/", uploaded_file.name)
#     with open(save_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     loader = PyPDFDirectoryLoader("./data/")
#     docs = loader.load()
#     return docs

import os

def load_data(uploaded_file):
    # Ensure the directory exists
    save_dir = "./data"
    os.makedirs(save_dir, exist_ok=True)

    # Define the save path
    save_path = os.path.join(save_dir, uploaded_file.name)

    # Save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Now load using PyPDFLoader or similar
    loader = PyPDFLoader(save_path)
    return loader.load()


# splitting text
def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=64)
    return text_splitter.split_documents(docs)

# storing in vector DB
def store_VDB(texts):
    global qa_chain  
    global memory    

    # Delete old vectorstore
    db_path = Path("./vectorstores/db_faiss")
    if db_path.exists() and db_path.is_dir():
        shutil.rmtree(db_path)  # Removes old vectorstore folder

    vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings_model)
    vectorstore.save_local(str(db_path))

    # Rebuild memory
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="question"
    )

    # Load new DB
    db = FAISS.load_local(str(db_path), embeddings_model, allow_dangerous_deserialization=True)

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": memory,
        },
    )

    return db

# defining QA chain
def create_chain(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferMemory(memory_key="history", input_key="question"),
        },
    )

# answering query
def query_qa_chain(qa_chain, query):
    response = qa_chain({"query": query})
    return response['result']

# main logic
if st.button("Get Answer"):
    if uploaded_file is not None and query:
        docs = load_data(uploaded_file)
        texts = split_text(docs)

        db_path = Path("./vectorstores/db_faiss")
        db = store_VDB(texts)
        qa_chain = create_chain(db)
        result = query_qa_chain(qa_chain, query)
        st.write("Answer:", result)
    else:
        st.write("Please upload a file and enter a query.")

