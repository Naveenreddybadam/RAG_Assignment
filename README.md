# 🏥 India Health Transformation - RAG Assistant

A Retrieval-Augmented Generation (RAG) application built using **FAISS**, **Sentence Transformers**, **Google Gemini**, and **Streamlit**. The assistant answers user questions based only on the provided PIB (Press Information Bureau) document about India's Health Transformation.

---

## 📌 Project Overview

This project implements a RAG pipeline that:

- Extracts text from a PIB webpage.
- Splits the document into smaller chunks.
- Generates embeddings using Sentence Transformers.
- Stores embeddings in a FAISS vector database.
- Retrieves the most relevant chunks based on the user's question.
- Uses Google Gemini to generate answers from the retrieved context.
- Displays results through a Streamlit web interface.

---

## 🚀 Features

- Document ingestion from PIB webpage
- Text chunking
- Semantic search using FAISS
- Sentence Transformer embeddings
- Google Gemini integration
- Streamlit user interface
- Context-based answer generation
- Prevents hallucinations by answering only from the retrieved document

---

## 🛠️ Technologies Used

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Google Gemini API
- LangChain Text Splitter
- BeautifulSoup
- Requests

---

## 📂 Project Structure

```
RAG_Assignment/
│
├── data/
│   ├── pib_document.txt
│   └── chunks.txt
│
├── vector_store/
│   ├── faiss_index.index
│   └── documents.pkl
│
├── app.py
├── ingest.py
├── rag.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Naveenreddybadam/RAG_Assignment.git
cd RAG_Assignment
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

## ▶️ Run the Project

Generate the vector database:

```bash
python ingest.py
```

Start the Streamlit application:

```bash
streamlit run app.py
```

---

## 💡 Sample Questions

- What is Ayushman Bharat?
- What is ABHA?
- What is Ayushman Bharat Digital Mission?
- What are the four pillars of Ayushman Bharat?

---

## 🧠 RAG Workflow

```
PIB Document
      │
      ▼
Text Extraction
      │
      ▼
Chunking
      │
      ▼
Embeddings
      │
      ▼
FAISS Vector Store
      │
      ▼
User Question
      │
      ▼
Semantic Retrieval
      │
      ▼
Gemini LLM
      │
      ▼
Final Answer
```

---

## 📸 Sample Output

**Question**

```
What is Ayushman Bharat?
```

**Answer**

```
Ayushman Bharat provides affordable and quality healthcare to citizens through four key pillars including health insurance, health and wellness centres, digital health infrastructure, and critical healthcare support.
```

---

## 👨‍💻 Author

**Naveen Reddy Badam**

B.Tech Information Technology
