# Powered HR Assistant 🧠

A Retrieval-Augmented Generation (RAG) web assistant designed to answer questions about KPMG's Powered HR methodology — including project phase activities, Target Operating Model (TOM) components, and sales process stages — by referencing internal documentation.

Built with:

- 🧠 LangChain (LLM + RetrievalQA)
- 📄 FAISS + BM25 hybrid retriever
- 🔍 OpenAI embeddings & GPT-4o
- 🌐 Flask web front-end
- ☁️ Deployed via [Render.com](https://render.com)

---

## 💡 Features

- Answers questions using only indexed Markdown documents
- Cites document sources
- Supports hybrid search (semantic + keyword)
- Clarifies differences between TOM assets and project delivery activities
- Handles contextual prompts about Powered HR phases and sales stages

---

## 📂 Project Structure
powered-rag/
├── app.py # Flask front-end app
├── scripts/
│ └── build_vector_store.py # Vector DB creation
├── templates/
│ └── index.html # Web interface
├── vectorstore/ # FAISS vector index (auto-generated)
├── data/ # Markdown knowledge base
├── requirements.txt
└── README.md

Developed by Nick Phillips
Assisted by GPT-4 (OpenAI + LangChain stack)