import os
import pickle
import sys
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# 1) Load your HF Hub token
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    print("❌ Please set HUGGINGFACEHUB_API_TOKEN in your .env")
    sys.exit(1)

# 2) Initialize the HF inference client for your instruct model
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# 3) Path to your FAISS index
INDEX_PATH = "ulcer_faiss_index.pkl"

def ensure_index():
    if not os.path.exists(INDEX_PATH):
        print("❌ FAISS index not found. Run `python build_index.py` first.")
        sys.exit(1)

def load_retriever():
    with open(INDEX_PATH, "rb") as f:
        faiss_index = pickle.load(f)
    return faiss_index.as_retriever(search_kwargs={"k": 4})

def build_prompt(docs: list[Document], question: str) -> str:
    # Include only top 3 docs
    lines = [d.page_content.replace("\n", " ") for d in docs[:3]]
    ctx = "\n".join(f"- {line}" for line in lines)
    
    # Use Mistral's required prompt format
    return (
        "<s>[INST] You are a friendly medical assistant specialized in gastric ulcers. "
        "Answer the user's question using ONLY the provided context. "
        "Please follow these guidelines when answering:\n"
        "  • Be clear and concise.\n"
        "  • Be precise—stick exactly to what you know from the context.\n"
        "  • Use simple, everyday English.\n"
        "  • Do NOT invent or hallucinate any details. "
        "If the answer isn't in the context, say \"I don't know, please consult a healthcare professional.\" "
        "Here is the context:\n\n"
        f"{ctx}\n\n"
        f"Question: {question} [/INST]"
    )

def generate_answer(prompt: str) -> str:
    # Call the HF text-generation endpoint
    output = client.text_generation(
        prompt,
        max_new_tokens=512,
        temperature=0.1,
        stop=["</s>"]
    )
    return output.strip()

def chat_loop():
    retriever = load_retriever()
    print("🤖 Ulcer Assistant (type 'exit' to quit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        # retrieve top-k
        docs = retriever.invoke(q)
        prompt = build_prompt(docs, q)
        answer = generate_answer(prompt)

        # Only show the answer (remove any prompt remnants)
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1].strip()
        
        print("\n📝 Answer:\n", answer, "\n")

if __name__ == "__main__":
    ensure_index()
    chat_loop()