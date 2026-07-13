<div align="center">
  <h1>🤖 Gemini RAG Chatbot</h1>
  <p>A powerful, fully-local Retrieval-Augmented Generation (RAG) assistant built with Google Gemini, LangChain, and Streamlit.</p>
  
  ![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
  ![Gemini](https://img.shields.io/badge/Google%20Gemini-1.5%20Flash-orange?style=for-the-badge)
  ![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg?style=for-the-badge)
  ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg?style=for-the-badge&logo=streamlit)
  ![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple?style=for-the-badge)
</div>

<br/>

## ✨ Features
- **📄 Chat with your Documents**: Upload `.pdf` or `.txt` files directly in the UI.
- **🧠 Advanced RAG Engine**: Built using modern **LangChain Expression Language (LCEL)**.
- **⚡ Blazing Fast**: Powered by Google's latest `gemini-1.5-flash` model and `embedding-001`.
- **🔍 Source Citations**: See exactly which chunks of text the AI used to generate its answer.
- **💾 Local Vector Storage**: Uses ChromaDB to securely store document embeddings entirely on your machine.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **AI / LLM:** Google Gemini API
* **Orchestration:** LangChain (LCEL)
* **Vector Database:** ChromaDB

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/kamalesh4044/rag-chatbot.git
cd rag-chatbot
```

### 2. Install dependencies
Ensure you have Python 3.11+ installed.
```bash
pip install -r requirements.txt
```

### 3. Add your API Key
Create a `.env` file in the root of the project and add your Google Gemini API key:
```env
GOOGLE_API_KEY="your_api_key_here"
```
*(Note: `.env` is ignored by git to keep your key secure!)*

### 4. Run the Application
```bash
streamlit run app.py
```
Your browser will automatically open to `http://localhost:8501`.

---

## 💡 How to Use
1. **Upload**: Use the sidebar to upload a PDF or TXT file.
2. **Process**: Click "Process Document" to chunk, embed, and save your document to the local ChromaDB database.
3. **Chat**: Ask questions about your document in the main chat window. Click the "View Sources" dropdown to see the exact text snippets the AI retrieved!

<div align="center">
  <i>Built with ❤️</i>
</div>
