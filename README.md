# 🧠 AI-Powered PDF RAG Assistant

A professional **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and interact with them using natural language.

The application processes documents, creates semantic embeddings, stores them in a ChromaDB vector database, retrieves relevant information using **MMR search**, and generates context-aware answers using a **Groq LLM**.

## ✨ Features

* 📄 Upload PDF documents directly through the Streamlit UI
* 🧩 Intelligent document chunking with LangChain
* 🔢 Semantic embeddings using HuggingFace
* 🗄️ Persistent vector storage with ChromaDB
* 🔎 MMR-based document retrieval
* 🤖 AI-generated answers using Groq
* 📚 Displays retrieved document sources and page numbers
* 💬 Interactive chat interface
* 🧹 Clear and rebuild the knowledge base
* 🎨 Professional neumorphic Streamlit UI
* 🔐 API keys stored securely using environment variables

## 🏗️ RAG Pipeline


             ┌─────────────────┐
             │    PDF Upload   │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │   PDF Loader    │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  Text Splitter  │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │   HuggingFace   │
             │    Embeddings   │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    ChromaDB     │
             │ Vector Database │
             └────────┬────────┘
                      │
                 User Query
                      │
                      ▼
             ┌─────────────────┐
             │  MMR Retriever  │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    Groq LLM     │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  AI Response +  │
             │     Sources     │
             └─────────────────┘


## 🛠️ Tech Stack

| Technology  | Purpose                              |
| ----------- | ------------------------------------ |
| Python      | Core programming language            |
| Streamlit   | Web application interface            |
| LangChain   | RAG pipeline and document processing |
| HuggingFace | Sentence embeddings                  |
| ChromaDB    | Vector database                      |
| Groq        | LLM inference                        |
| PyPDF       | PDF document loading                 |
| dotenv      | Environment variable management      |

## 📁 Project Structure


ragproject/
│
├── main.py
├── create_database.py
├── requirements.txt
├── .env
├── .gitignore
│
└── chroma_db/
    └── Vector database files


> chroma_db and .env should not be committed to GitHub.

## ⚙️ Installation

### 1. Clone the repository


git clone https://github.com/your-username/your-repository.git
cd your-repository


### 2. Create a virtual environment

python -m venv .venv


Activate it on Windows:

.venv\Scripts\activate


### 3. Install dependencies


pip install -r requirements.txt


### 4. Configure environment variables

Create a `.env` file in the project root:

env
GROQ_API_KEY=your_groq_api_key


Do not upload your `.env` file to GitHub.

## ▶️ Run the Application

Start the Streamlit application:


streamlit run main.py


The application will open in your browser.

## 📖 How to Use

1. Open the RAG application.
2. Upload a PDF from the sidebar.
3. Click **Build Knowledge Base**.
4. The PDF is loaded and split into smaller chunks.
5. HuggingFace generates embeddings for the chunks.
6. ChromaDB stores the vectors.
7. Enter a question in the chat box.
8. The MMR retriever finds relevant document sections.
9. The Groq LLM generates an answer using the retrieved context.
10. Retrieved sources and page numbers are displayed below the response.

## 🔍 Retrieval Configuration

The application uses **Maximum Marginal Relevance (MMR)** retrieval:

python
search_kwargs={
    "k": 4,
    "fetch_k": 10,
    "lambda_mult": 0.5
}


This retrieves relevant chunks while also attempting to reduce redundancy between retrieved results.

## 🧠 Embedding Model

The project uses:


sentence-transformers/all-mpnet-base-v2


This model converts document chunks and user queries into numerical vectors, allowing the application to perform semantic similarity search.

## 🤖 LLM

The application uses Groq for fast LLM inference:

python
ChatGroq(
    model="openai/gpt-oss-20b"
)


The model receives the retrieved document context along with the user's question and generates the final response.

## 🔐 Security

Sensitive information is kept outside the source code using `.env`:
env
GROQ_API_KEY=your_api_key


The `.gitignore` file prevents secrets, uploaded PDFs, and ChromaDB files from being committed.

## 🚀 Future Improvements

* 🔐 User authentication
* 📚 Support for multiple PDFs
* 🗂️ Multiple knowledge bases
* 💾 Conversation history
* 📊 Retrieval confidence and relevance scores
* 📝 Support for DOCX and TXT files
* 🌐 Cloud deployment
* ⚡ Streaming LLM responses
* 🎙️ Voice-based questions
* 🧠 Conversational memory

## 👨‍💻 Author

**Yaatharth Bawankar**
