import os
import pickle
import faiss
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="India Health RAG Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 India Health Transformation - RAG Assistant")
st.write("Ask questions about the PIB document.")

# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------
# Cache Models
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_store():
    index = faiss.read_index("vector_store/faiss_index.index")

    with open("vector_store/documents.pkl", "rb") as f:
        documents = pickle.load(f)

    return index, documents

model = load_embedding_model()
index, documents = load_vector_store()

# -----------------------------
# Question Input
# -----------------------------
question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is Ayushman Bharat?"
)

# -----------------------------
# Search + Generate
# -----------------------------
if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:

        with st.spinner("Searching relevant information..."):

            query_embedding = model.encode([question])

            distances, indices = index.search(
                np.array(query_embedding).astype("float32"),
                3
            )

            context = ""

            for idx in indices[0]:
                if idx != -1:
                    context += documents[idx]
                    context += "\n\n"

            prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

If the answer is not found in the context, reply exactly:

"I could not find the answer in the provided PIB document."

Context:
{context}

Question:
{question}
"""

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )

        st.success("Answer")

        st.write(response.text)

        with st.expander("Retrieved Context"):

            st.text(context)