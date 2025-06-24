import os

from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)


index_name = "ulcer-index" 

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

#creating embeddings by converting the splitted chunks of text into a format the the AI model can understand
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

with open("data/ulcer.txt", encoding="utf-8") as f:
    text = f.read()

document = Document(page_content=text, metadata={"source": "data/ulcer.txt"})

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents([document])
# Adding documents to the vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
vector_store.add_documents(chunks)