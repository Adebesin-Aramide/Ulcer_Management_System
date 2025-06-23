# build_index.py

import os
import pickle
from glob import glob
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load .env (if present)
load_dotenv()

def load_txt_docs(path: str):
    """
    Load .txt files from a folder or a single file.
    Returns a list of LangChain Documents.
    """
    docs = []
    if os.path.isdir(path):
        for fn in glob(os.path.join(path, "*.txt")):
            with open(fn, encoding="utf-8") as f:
                text = f.read()
            docs.append(Document(page_content=text,
                                 metadata={"source": os.path.basename(fn)}))
    elif os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(page_content=text,
                             metadata={"source": os.path.basename(path)}))
    else:
        raise FileNotFoundError(f"No file or directory at {path}")
    return docs

def main():
    # Point this at your ulcer knowledge base
    data_path = "ulcer.txt"   # or a folder "ulcer_kb/"
    raw_docs = load_txt_docs(data_path)

    # Split into ~500-token chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)

    # Embed locally with a small Sentence-Transformer
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index = FAISS.from_documents(chunks, embeddings)

    # Persist the FAISS index
    with open("ulcer_faiss_index.pkl", "wb") as f:
        pickle.dump(faiss_index, f)

    print("✅ FAISS index built and saved to ulcer_faiss_index.pkl")

if __name__ == "__main__":
    main()
