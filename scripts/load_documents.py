from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_docs(data_dir="data"):
    def custom_loader(path):
        return TextLoader(path, encoding="utf-8")

    loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=custom_loader)
    docs = loader.load()

    # Tag documents related to the KPMG sales process
    for doc in docs:
        filename = Path(doc.metadata.get("source", "")).name.lower()
        if filename.startswith("salesprocess-") or "-salesprocess-" in filename:
            doc.metadata["topic"] = "kpmg_sales_process"

    # Improved chunking to preserve stage context
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

if __name__ == "__main__":
    docs = load_and_split_docs()
    print(f"✅ Loaded and split {len(docs)} chunks.")

