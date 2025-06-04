from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_docs(data_dir="data"):
    def custom_loader(path):
        return TextLoader(path, encoding="utf-8")

    loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=custom_loader)
    docs = loader.load()

    # Tag documents with relevant topic metadata
    for doc in docs:
        filename = Path(doc.metadata.get("source", "")).name.lower()

        if filename.startswith("salesprocess-") or "-salesprocess-" in filename:
            doc.metadata["topic"] = "kpmg_sales_process"
        elif filename.startswith("poweredphase-"):
            doc.metadata["topic"] = "powered_phase_delivery"
        elif filename == "poweredhr-glossary-testing.md":
            doc.metadata["topic"] = "powered_testing_glossary"
        elif "tom" in filename:
            doc.metadata["topic"] = "powered_tom_assets"
        elif "vision" in filename or "validate" in filename:
            doc.metadata["topic"] = "powered_methodology_structure"
        else:
            doc.metadata["topic"] = "general"

    # Use conservative chunking to preserve semantic context
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

if __name__ == "__main__":
    docs = load_and_split_docs()
    print(f"✅ Loaded and split {len(docs)} chunks.")
