import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # Correct import path
from load_documents import load_and_split_docs

# ── Load environment variables ───────────────────────────────────────
load_dotenv()  # Loads OPENAI_API_KEY from .env

def build_vectorstore():
    # Load and split documents
    documents = load_and_split_docs()

    # Use the same embedding model expected by your app.py
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create vectorstore from documents
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save locally
    vectorstore.save_local("vectorstore")
    print("✅ Vectorstore created and saved.")

if __name__ == "__main__":
    build_vectorstore()

