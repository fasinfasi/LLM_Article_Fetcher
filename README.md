# 📚🔍Research Question Answering with LLMs

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-2E7D32?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-CD7B21?logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21F?logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Semantic%20Search-5C6BC0?logo=semanticweb&logoColor=white" />
  <img src="https://img.shields.io/badge/Faiss-40C9CE?logo=faiss&logoColor=white" />
  <img src="https://img.shields.io/badge/Vector%20Database-006BFF?logo=database&logoColor=white" />
  <img src="https://img.shields.io/badge/Embedding-7B1FA2?logo=brain&logoColor=white" />
  <img src="https://img.shields.io/badge/Web%20Scraping-DFCB7C?logo=scrapy&logoColor=white" />
</p>


This project demonstrates how to build a question-answering system that extracts and analyzes content from online news articles. By inputting URLs of financial or general news sources, users can ask natural language questions and receive relevant, source-backed answers.

It combines modern NLP techniques like **semantic search** with **vector databases**, **large language models (LLMs)**, and **retrieval-augmented generation (RAG)** and so on. The system fetches content from the web, splits it into manageable chunks, converts them into embeddings for efficient similarity search, and uses an LLM to generate accurate responses based on the retrieved context.

This setup reflects how real-world applications like financial analysis tools, research assistants, and document summarizers can be built using open-source tools and APIs.

---

## ✅ Features

- 🔗 Input up to **3 article URLs** (supports unstructured web pages).
- 📄 Automatically **loads and splits** article content into chunks.
- 🧠 Generates **vector embeddings** using HuggingFace sentence-transformers.
- 🔍 Stores embeddings in a **FAISS vector database**.
- ❓ Ask questions in natural language and get answers backed by article content.
- 📚 Displays **answer sources**.
- 💾 Uses **local pickling** to persist the FAISS vector store.

---

## 🧰 Tech Stack

| Component        | Description                                       |
|------------------|---------------------------------------------------|
| `Streamlit`      | Frontend UI for user interaction                  |
| `LangChain`      | Manages LLM chains, document splitting, QA logic |
| `FAISS`          | Vector similarity search                          |
| `HuggingFace`    | Embeddings and Zephyr-7B LLM endpoint             |
| `dotenv`         | Manages HuggingFace API key                       |
| `pickle`         | For saving/loading the FAISS index                |

---

## 🤔 Why Not Just Use ChatGPT or Gemini or any other GPTs instead of this?

**The answer is yes😀, they can do this as well but there are some limitations with it😟**
- **Token Limits**: You can’t paste multiple long articles into ChatGPT — it will either truncate or reject them due to context size limits.
- **Manual Effort**: You have to manually copy-paste each article. That’s slow and error-prone when dealing with 3–10+ sources.
- **No Source Tracking**: Generic LLMs usually don’t tell you which part of the input they based their answer on. There’s no transparency.
- **Not trustable**: Once your ChatGPT session resets, your previous inputs are gone. This system persists the article knowledge in a vector database.
- **No Semantic Search**: ChatGPT reads what you paste, but it doesn’t perform retrieval of the most relevant chunks like vector databases do.
- **Not Scalable**: If you're building a real product (for clients, analysts, etc.), manual GPT usage is not scalable or automatable.

---

## ✅ Why This Project Makes Sense

This project give:
- An automated pipeline to ingest multiple articles via URLs.
- Semantic search powered by vector embeddings — retrieves only the most relevant parts.
- A custom LLM-powered QA system that works only on the given sources (controlled, factual).
- Scalability — you can later expand it to analyze hundreds of URLs, add scheduled scraping, or switch to private LLMs.
- A real use case — e.g., a financial analyst wants to ask “What’s the latest update on Adani Group?” across 5 financial sites without reading all of them.


**📌In short: I'm not replace GPTs, instead solving a specific problem that ChatGPT isn’t optimized for.**
---

## 🧪 How It Works

### 1. **Input URLs**
The user enters up to 3 news article URLs in the sidebar.

### 2. **Text Extraction & Splitting**
The app fetches the raw content using `UnstructuredURLLoader` and splits it into overlapping chunks using `RecursiveCharacterTextSplitter`.

### 3. **Embeddings + FAISS Indexing**
Chunks are converted into vector embeddings (`all-MiniLM-L6-v2`) and stored in a FAISS index, which is pickled to disk.

### 4. **Question Answering**
User inputs a question. The FAISS index retrieves relevant chunks, and a **HuggingFace-hosted Zephyr-7B model** generates the answer using LangChain's `RetrievalQAWithSourcesChain`.

---

## 📦 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/fasinfasi/LLM_Article_Fetcher.git
   cd LLM_Article_Fetcher
   ```
2. Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Add your HuggingFace token
   create .env file with following content
   ```
   HF_TOKEN='your_huggingface_api_key'
   ```
4. Run the Streamlit app
   ```
   streamlit run main.py
   ```

---

## 📝 Notes
- This app uses the CPU version of the HuggingFace embeddings. For faster performance, GPU support can be added.
- Currently, it saves one FAISS index (faiss_store_openai.pkl) for all URLs. If you want to handle multiple sessions or user queries, you'll need a more dynamic FAISS store strategy.
- Error handling is minimal (e.g., no validation if the URL fails or returns empty content).

#### I'm Fasin, author of this
🤗Feel free to connect or check out my other projects.

