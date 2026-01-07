# 📚 DocIntel AI - Advanced Document Intelligence System

## 🚀 Overview
DocIntel AI is a powerful, fully-local document intelligence platform that allows you to upload PDF documents and interact with them using natural language. The system uses advanced **RAG (Retrieval-Augmented Generation)** techniques with vector search, chat memory, and intelligent context-aware responses.

---

## ✨ Features

### 🔧 Core Features
- **100% Local Processing** – No API keys or internet required after installation  
- **PDF Upload & Processing** – Drag & drop or click to upload PDF files  
- **Intelligent Q&A** – Ask questions about your documents and get accurate answers  
- **Hybrid Search** – Combines vector similarity and keyword search for best results  
- **Chat Memory** – Maintains conversation context across sessions  
- **Document Highlights** – Automatically extracts and displays relevant sections  
- **Streaming Responses** – Real-time streaming answers for better UX  
- **Session Management** – Multiple document sessions with persistence  

### 🎨 UI/UX Features
- **Modern Responsive Design** – Works on desktop, tablet, and mobile  
- **Real-time Chat Interface** – Clean, modern chat UI with message history  
- **Document Insights Panel** – Visual highlights and source references  
- **Progress Indicators** – Real-time upload and processing progress  
- **Sample Questions** – Quick-start questions to explore documents  
- **Dark/Light Mode Ready** – Built with Tailwind CSS for easy theming  

---
```
## 📁 Project Structure

docintel-ai/
├── frontend/ # React/Next.js Frontend
│ ├── app/ # Next.js app directory
│ ├── components/ # React components
│ ├── public/ # Static assets
│ └── package.json
├── backend/ # FastAPI Backend
│ ├── hello.py # Main FastAPI application
│ ├── requirements.txt # Python dependencies
│ └── chroma_db/ # Vector database storage
└── README.md # This file
```

yaml
Copy code

---

## 🛠️ Installation

### Prerequisites
- **Node.js 18+** and **npm/yarn**  
- **Python 3.9+**  
- **4GB RAM minimum**, 8GB recommended  

### Backend Setup
1. Navigate to backend directory:

```bash
cd backend
Create virtual environment:

bash
Copy code
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the backend:

bash
Copy code
python hello.py
# OR
uvicorn hello:app --host 127.0.0.1 --port 8001 --reload
Frontend Setup
Navigate to frontend directory:

bash
Copy code
cd frontend
Install dependencies:

bash
Copy code
npm install
# OR
yarn install
Create environment file:

bash
Copy code
echo "NEXT_PUBLIC_API_URL=http://localhost:8001" > .env.local
Run the frontend:

bash
Copy code
npm run dev
# OR
yarn dev
🎯 Usage
Access the Application

Frontend: http://localhost:3000

Backend API: http://localhost:8001

API Documentation: http://localhost:8001/docs

Upload a Document

Click "Choose File" or drag & drop a PDF

Click "Process with AI"

Wait for the document to be processed (progress will be shown)

Ask Questions

Type your question in the chat input

Click "Send" for regular response

Click "Stream" for streaming response

Or try sample questions from the panel

Explore Features

Active Sessions: Switch between different documents

Hybrid Search: Search across all uploaded documents

Document Insights: View highlighted sections and statistics

Export: Download chat history as JSON

Summary: Generate document summaries

🔧 Technical Architecture
Backend Stack
FastAPI: Modern, fast web framework for APIs

LangChain: Framework for LLM applications

ChromaDB: Vector database for embeddings

Sentence Transformers: Local embeddings model

PyPDF: PDF text extraction

Frontend Stack
Next.js 14: React framework with App Router

Tailwind CSS: Utility-first CSS framework

Lucide React: Icon library

React Hooks: State management

Key Technologies
RAG Pipeline: Document → Chunks → Embeddings → Vector Store → Retrieval

Hybrid Search: Combines vector similarity (ChromaDB) + keyword search

Context-Aware Responses: Intelligent template-based responses

Streaming: Server-Sent Events (SSE) for real-time responses

Session Management: In-memory session storage with persistence