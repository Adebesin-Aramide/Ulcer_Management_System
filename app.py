# app.py
import os
import pickle
import sys
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    print("❌ Please set HUGGINGFACEHUB_API_TOKEN in your .env")
    sys.exit(1)

# Initialize InferenceClient
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# Load FAISS index
INDEX_PATH = "ulcer_faiss_index.pkl"
if not os.path.exists(INDEX_PATH):
    print(f"❌ FAISS index not found at {INDEX_PATH}. Run `build_index.py` first.")
    sys.exit(1)

with open(INDEX_PATH, "rb") as f:
    faiss_index = pickle.load(f)
retriever = faiss_index.as_retriever(search_kwargs={"k": 4})

# FastAPI app
app = FastAPI(title="Ulcer RAG Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

def build_prompt(docs: List[Document], question: str) -> str:
    # Include up to top 3 documents as context bullets
    if docs:  # Check if docs is not empty
        lines = [d.page_content.replace("\n", " ").strip() for d in docs[:3]]
    else:  # Handle empty docs case
        lines = []
    ctx = "\n".join(f"- {line}" for line in lines) if lines else "No relevant context found"
    
    # Properly formatted prompt string
    return (
        "<s>[INST] <<SYS>>\n"
        "You are a medical assistant specialized in gastric ulcers. "
        "Answer the user's question using ONLY the provided context. "
        "If the answer isn't in the context, say: \"I don't know, please consult a healthcare professional.\"\n"
        "<</SYS>>\n\n"
        "Context:\n"
        f"{ctx}\n\n"
        f"Question: {question} [/INST]"
    )

def generate_answer(prompt: str) -> str:
    # Generate the response
    output = client.text_generation(
        prompt,
        max_new_tokens=512,
        temperature=0.1,
        stop_sequences=["</s>"]
    )
    
    # Extract just the assistant's response
    if "[/INST]" in output:
        return output.split("[/INST]")[-1].strip()
    return output.strip()

@app.get("/")
def root():
    return {"message": "Ulcer RAG Chatbot API is running."}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        # 1) retrieve relevant docs
        docs = retriever.invoke(req.question)
        
        # 2) build prompt and generate answer
        prompt = build_prompt(docs, req.question)
        
        # Print prompt for debugging (first 500 characters)
        print(f"\n--- PROMPT ---\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}")
        
        answer = generate_answer(prompt)
        
        # Print answer for debugging
        print(f"\n--- ANSWER ---\n{answer}\n")
        
        return ChatResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")