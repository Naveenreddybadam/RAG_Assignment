import os
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai


# -----------------------------
# Load Gemini API
# -----------------------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------
# Load Embedding Model
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load FAISS Index
# -----------------------------
print("Loading FAISS index...")
index = faiss.read_index("vector_store/faiss_index.index")

# -----------------------------
# Load Documents
# -----------------------------
print("Loading documents...")

with open("vector_store/documents.pkl", "rb") as f:
    documents = pickle.load(f)

print(f"Loaded {len(documents)} chunks.")

# -----------------------------
# Ask User Question
# -----------------------------
query = input("\nEnter your question: ")

# -----------------------------
# Convert Question to Embedding
# -----------------------------
query_embedding = model.encode([query])

# -----------------------------
# Search Top 3 Chunks
# -----------------------------
k = 3
distances, indices = index.search(np.array(query_embedding).astype("float32"), k)

# -----------------------------
# Build Context
# -----------------------------
context = ""

for idx in indices[0]:
    if idx != -1:
        context += documents[idx]
        context += "\n\n"

# -----------------------------
# Prompt
# -----------------------------
prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context provided below.

If the answer is not available in the context, reply exactly:

"I could not find the answer in the provided PIB document."

Context:
{context}

Question:
{query}
"""

# -----------------------------
# Gemini Response
# -----------------------------
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt
)

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)
print(response.text)