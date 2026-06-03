import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile, os

st.set_page_config(page_title="RAG Q&A App", page_icon="📄")
st.title("📄 RAG-Based Document Q&A")
st.markdown("Upload a PDF and ask questions about its content.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    with st.spinner("Processing document..."):
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeddings)

        pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
        llm = HuggingFacePipeline(pipeline=pipe)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    st.success("Document ready! Ask your questions below.")
    question = st.text_input("💬 Ask a question:")
    if question:
        with st.spinner("Generating answer..."):
            answer = qa.run(question)
        st.markdown(f"**Answer:** {answer}")

    os.unlink(tmp_path)
