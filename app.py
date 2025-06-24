# app.py
import os
import sys
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from pinecone import Pinecone
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
import warnings

warnings.filterwarnings("ignore")

# ─── Load env ───────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "ulcer-index"

if not HF_TOKEN or not PINECONE_API_KEY or not PINECONE_ENV:
    print("❌ Please set all required tokens in your .env")
    sys.exit(1)

# ─── Init Mistral Inference ────────────────
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# ─── Init Pinecone Vector Store ─────────────
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"❌ Pinecone index `{INDEX_NAME}` not found.")
    sys.exit(1)

index = pc.Index(INDEX_NAME)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# ─── FastAPI Setup ─────────────────────────
app = FastAPI(title="Ulcer RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request/Response Models ───────────────
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

# ─── Prompt Formatter ──────────────────────
def build_prompt(docs: List[Document], question: str) -> str:
    if docs:
        lines = [d.page_content.replace("\n", " ").strip() for d in docs[:3]]
    else:
        lines = []
    ctx = "\n".join(f"- {line}" for line in lines) if lines else "No relevant context found"

    return (
        "<s>[INST] <<SYS>>\n"
        "You are a medical assistant specialized in gastric ulcers. "
        "Answer the user's question using ONLY the provided context. "
        "If the answer isn't in the context, say: \"I don't know, please consult a healthcare professional.\"\n"
        "<</SYS>>\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {question} [/INST]"
    )

# ─── Inference with Mistral ────────────────
import requests

def generate_answer(prompt: str) -> str:
    response = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.1}}
    )
    result = response.json()
    return result[0]["generated_text"].split("[/INST]")[-1].strip() if isinstance(result, list) else result


# ─── Root Route ────────────────────────────
@app.get("/")
def root():
    return {"message": "Ulcer RAG Chatbot API is running."}

# ─── Main Chat Endpoint ────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        docs = retriever.invoke(req.question)
        prompt = build_prompt(docs, req.question)

        print(f"\n--- PROMPT ---\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}")
        answer = generate_answer(prompt)
        print(f"\n--- ANSWER ---\n{answer}\n")

        return ChatResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
