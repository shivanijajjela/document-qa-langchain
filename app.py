import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub

# Use local LLM (stub with your choice)
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Setup local embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Simple QA model (replace if needed)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Streamlit UI
st.set_page_config(page_title="📄 Document Q&A App")
st.title("📄 Document Q&A App (Local, No API Keys)")

uploaded_file = st.file_uploader("Upload PDF or DOCX file", type=["pdf", "docx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load documents
    try:
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = Docx2txtLoader(tmp_path)
        docs = loader.load()
    except Exception as e:
        st.error(f"⚠️ Error loading document: {e}")
        st.stop()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    # Create FAISS vectorstore
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
    except Exception as e:
        st.error(f"⚠️ VectorStore creation failed: {e}")
        st.stop()

    st.success("✅ Document processed. You can now ask questions.")

    query = st.text_input("Ask a question based on the document")

    if query:
        try:
            docs = vectorstore.similarity_search(query)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write("💡 Answer:")
            st.success(response)
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")
