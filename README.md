# 📄 RAG-Based Document Q&A System

A Retrieval-Augmented Generation (RAG) pipeline that answers questions
from custom documents using LangChain, FAISS, and Flan-T5.

## 🔍 How It Works
1. Upload any PDF document
2. Document is chunked and embedded using Sentence-Transformers
3. FAISS stores the embeddings for fast retrieval
4. Flan-T5 generates answers based on retrieved context

## 📊 Results on SQuAD 2.0 (100 samples)
| Metric | Score |
|--------|-------|
| Exact Match (EM) | 58% |
| F1 Score | 0.71 |

## 🛠️ Tech Stack
- **LangChain** – RAG pipeline orchestration
- **FAISS** – Vector similarity search
- **Sentence-Transformers** – all-MiniLM-L6-v2 embeddings
- **Flan-T5** – Answer generation (runs locally)
- **Streamlit** – Interactive UI

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run rag_app.py
```

## 📁 Project Structure
```
rag-qa-app/
├── rag_app.py        # Streamlit UI + RAG pipeline
├── eval.py           # Evaluation on SQuAD 2.0
├── requirements.txt
└── README.md
```
