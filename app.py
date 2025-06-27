import os
import pickle
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import InferenceClient, hf_hub_download
from langchain_community.vectorstores import FAISS

# --- 1) Load secrets ---
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    st.error("Missing HUGGINGFACEHUB_API_TOKEN (set in Streamlit Secrets)")
    st.stop()

# --- 2) Download FAISS index from Hugging Face Hub ---
try:
    file_path = hf_hub_download(
        repo_id="Aramide/ulcer_faiss_index",  # Replace with your HF username/repo
        filename="ulcer_faiss_index.pkl",
        token=HF_TOKEN  # If repo is private; remove token= if public
    )
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load FAISS index from Hugging Face: {e}")
    st.stop()

# --- 3) Create retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- 4) Init InferenceClient ---
client = InferenceClient(
    provider="novita",
    api_key=HF_TOKEN
)

# --- 5) Build prompt function ---
def build_prompt(docs, question: str) -> str:
    bullets = []
    for d in docs[:3]:
        text = d.page_content.replace("\n", " ")
        bullets.append(f"- {text}")
    context = "\n".join(bullets) if bullets else "No context available."
    return (
        "You are a medical assistant specialized in gastric ulcers. "
        "Answer the question using ONLY the context below. "
        "Provide a clear and concise answer. "
        "Do not include any references, sources, or notes in your response. "
        "If you don't know, say 'I don't know, please consult a healthcare professional.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

# --- 6) Generate answer function ---
def generate_answer(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    return completion.choices[0].message.content.strip()

# --- 7) Streamlit UI ---
st.set_page_config(page_title="UlcerMate Assistant")
st.title("UlcerMate Assistant")

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
