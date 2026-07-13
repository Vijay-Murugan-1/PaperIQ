# 📄 PaperIQ

> An AI-powered research assistant that helps users understand academic papers through Retrieval-Augmented Generation (RAG).

PaperIQ enables users to upload research papers, generate concise summaries, ask questions, and extract key insights using Large Language Models (LLMs). It combines semantic search with Retrieval-Augmented Generation (RAG) to provide context-aware responses grounded in the uploaded document.

---

## ✨ Features

- 📄 Upload and process PDF research papers
- ✂️ Token-based text chunking
- 🧠 Semantic embeddings using Sentence Transformers
- 🗂️ FAISS vector database for semantic retrieval
- 🔍 Context-aware Question Answering (RAG)
- 📝 AI-generated Paper Summary
- 💡 AI-generated Key Insights
- ⚡ Session-state caching to avoid repeated processing
- 🎯 Modular architecture for future AI features

---

## 🚧 Upcoming Features

- ❓ Quiz Generator
- 📚 Explain Concepts
- 📝 Flashcards
- 📖 Source Citations
- 📑 Multi-document Retrieval

---

# 🏗️ Architecture

```
                PDF Research Paper
                        │
                        ▼
              PDF Text Extraction
                        │
                        ▼
             Token-based Chunking
                        │
                        ▼
          Sentence Transformer Embeddings
                        │
                        ▼
              FAISS Vector Database
                        │
                        ▼
               Semantic Retrieval
                        │
                        ▼
               Context Construction
                        │
                        ▼
               Prompt Engineering
                        │
                        ▼
                 Gemini LLM
                        │
                        ▼
             ┌──────────┬──────────────┬
             ▼          ▼              ▼             
           Summary     Q&A         Key Insights      
```

---

# 📂 Project Structure

```
PaperIQ/
│
├── app.py
│
├── modules/
│   ├── pdf_processor.py
│   ├── text_chunker.py
│   ├── embedding_generator.py
│   ├── vector_store.py
│   ├── context_builder.py
│   ├── prompt_builder.py
│   ├── summary_builder.py
│   ├── insight_builder.py
│   └── llm.py
│
├── requirements.txt
├── README.md
└── .streamlit/
```

---

# ⚙️ Technologies Used

## Programming Language

- Python

## Framework

- Streamlit

## LLM

- Google Gemini API

## Embedding Model

- BAAI/bge-small-en-v1.5

## Vector Database

- FAISS

## PDF Processing

- PyMuPDF

## Libraries

- Sentence Transformers
- Transformers
- NumPy

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Vijay-Murugan-1/PaperIQ.git

cd PaperIQ
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Configure Gemini API Key

Create the following file:

```
.streamlit/secrets.toml
```

Add your Gemini API key:

```toml
GEMINI_API_KEY = "YOUR_API_KEY"
```

> **Note:** The `secrets.toml` file is intentionally excluded from version control and should never be committed.

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 💻 Current Workflow

1. Upload a research paper (PDF)
2. Extract text from the document
3. Split the text into token-based chunks
4. Generate semantic embeddings
5. Store embeddings in a FAISS vector database
6. Choose one of the available AI features:
   - Paper Summary
   - Ask Questions
   - Key Insights
7. Receive context-aware AI responses generated using Gemini.

---

# 🧠 How Question Answering Works

```
User Question
      │
      ▼
Generate Query Embedding
      │
      ▼
Semantic Search (FAISS)
      │
      ▼
Retrieve Top Relevant Chunks
      │
      ▼
Context Builder
      │
      ▼
Prompt Builder
      │
      ▼
Gemini
      │
      ▼
Final Answer
```

---

# 🎯 Project Goals

PaperIQ aims to simplify research paper reading by combining semantic retrieval with Large Language Models to provide accurate, context-aware assistance for researchers and students.

---

# 🔮 Future Improvements

- Multi-document search
- Hybrid Retrieval (BM25 + FAISS)
- Source citations with page references
- Cross-encoder reranking
- Streaming LLM responses
- Research paper metadata extraction
- Conversation history
- Export summaries and notes
- Local LLM support

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Vijay B V**

Integrated M.Tech Computer Science and Engineering (Data Science)

VIT Vellore
