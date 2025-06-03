import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from load_documents import load_and_split_docs
from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from .env

def build_vectorstore():
    documents = load_and_split_docs()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("vectorstore")
    print("✅ Vectorstore created and saved.")

if __name__ == "__main__":
    build_vectorstore()
