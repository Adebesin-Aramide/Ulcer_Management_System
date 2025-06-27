import os
import pickle
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS

# 1) Load secrets
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    st.error("Missing HUGGINGFACEHUB_API_TOKEN (set in Streamlit Secrets)")
    st.stop()

# 2) Load FAISS index
try:
    with open("ulcer_faiss_index.pkl", "rb") as f:
        vectorstore = pickle.load(f)
except FileNotFoundError:
    st.error("ulcer_faiss_index.pkl not found. Run build_index.py first.")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3) Init Mistral via HF Inference API
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=HF_TOKEN
)

def build_prompt(docs, question: str) -> str:
    bullets = []
    for d in docs[:3]:
        text = d.page_content.replace("\n", " ")
        bullets.append(f"- {text}")
    context = "\n".join(bullets) if bullets else "No context available."
    return (
        "<s>[INST] You are a medical assistant specialized in gastric ulcers. "
        "Answer using ONLY the context below. "
        "If you don't know, say \"I don't know, please consult a healthcare professional.\""
        "[/INST]\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

def generate_answer(prompt: str) -> str:
    out = client.text_generation(
        prompt,
        max_new_tokens=256,
        temperature=0.1,
        stop=["[/INST]"]
    )
    return out.split("[/INST]")[-1].strip()

# --- Streamlit UI ---
st.set_page_config(page_title="UlcerMate Assistant")
st.title("UlcerMate Assistant")

# Use a form to capture Enter key as submit
with st.form("query_form"):
    question = st.text_input("Enter your question about ulcers:", "")
    submitted = st.form_submit_button("Submit")

if submitted:
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context…"):
            docs = retriever.get_relevant_documents(question)
        with st.spinner("Generating answer…"):
            prompt = build_prompt(docs, question)
            answer = generate_answer(prompt)

        st.subheader("Answer")
        st.write(answer)
