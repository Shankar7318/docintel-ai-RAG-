from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import tempfile
import uuid
import json
import asyncio
from datetime import datetime
import traceback

# Local imports - simplified
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI(title="Local Document Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== CONFIGURATION ==========
print("🚀 Starting Local Document Intelligence API")

# Initialize embeddings (works offline)
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Embeddings loaded successfully")
except Exception as e:
    print(f"❌ Failed to load embeddings: {e}")
    raise

# Initialize Chroma
chroma_dir = "./chroma_db"
if not os.path.exists(chroma_dir):
    os.makedirs(chroma_dir)

# Global variables
vectorstore = None
chat_sessions = {}
documents_store = []

# ========== MODELS ==========
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class HighlightInfo(BaseModel):
    file: str
    page: int
    text: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[HighlightInfo]
    session_id: str

# ========== HELPER FUNCTIONS ==========
def initialize_vectorstore():
    """Initialize or load Chroma vectorstore"""
    global vectorstore
    try:
        os.environ['ANONYMIZED_TELEMETRY'] = 'False'

        vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings,
            collection_name="documents"
        )

        try:
            doc_count = len(vectorstore.get()["documents"])
        except Exception:
            doc_count = 0

        print(f"✅ Loaded existing vectorstore with {doc_count} documents")
    except Exception as e:
        print(f"⚠️ Could not load existing vectorstore: {e}")
        print("Creating new vectorstore...")
        try:
            vectorstore = Chroma(
                collection_name="documents",
                embedding_function=embeddings,
                persist_directory=chroma_dir
            )
            print("✅ Created new vectorstore")
        except Exception as e2:
            print(f"❌ Failed to create vectorstore: {e2}")
            raise

def search_documents(query: str, k: int = 3):
    """Search documents using vector similarity"""
    global vectorstore

    if vectorstore is None:
        initialize_vectorstore()

    try:
        try:
            doc_count = len(vectorstore.get()["documents"])
        except Exception:
            doc_count = 0

        actual_k = min(k, max(1, doc_count))

        if doc_count == 0:
            return []

        results = vectorstore.similarity_search_with_score(query, k=actual_k)

        formatted_results = []
        for doc, score in results:
            metadata = doc.metadata
            formatted_results.append({
                "text": doc.page_content,
                "metadata": metadata,
                "score": 1.0 - float(score),
                "type": "vector"
            })

        return formatted_results

    except Exception as e:
        print(f"⚠️ Search error: {e}")
        return []

def simple_keyword_search(query: str, k: int = 2):
    """Simple keyword matching search"""
    if not documents_store:
        return []

    query_lower = query.lower().split()
    results = []

    for doc in documents_store:
        text_lower = doc["text"].lower()
        score = 0

        for word in query_lower:
            if len(word) > 3 and word in text_lower:
                score += 1

        if score > 0:
            relevance = min(1.0, score / len(query_lower))
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": relevance,
                "type": "keyword"
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]

def extract_highlight(doc_result: Dict):
    metadata = doc_result.get("metadata", {})
    return HighlightInfo(
        file=metadata.get("source", "document.pdf"),
        page=metadata.get("page", 0) + 1,
        text=doc_result["text"][:300],
        score=round(doc_result.get("score", 0.5), 2)
    )

def generate_intelligent_response(context: str, question: str, highlights: List[Dict]):
    """Generate intelligent response based on context"""
    pages = []
    key_sentences = []

    for highlight in highlights:
        if highlight["metadata"].get("page") is not None:
            pages.append(highlight["metadata"]["page"] + 1)

        text = highlight["text"]
        sentences = text.split('.')
        for sentence in sentences[:3]:
            if len(sentence.strip()) > 20:
                key_sentences.append(sentence.strip())

    pages = sorted(list(set(pages)))
    key_sentences = list(set(key_sentences))[:5]

    question_lower = question.lower()

    # Summary-like logic
    if any(word in question_lower for word in ['summary', 'summarize', 'overview', 'overall']):
        return f"""Based on the document, here's a summary:

Main topics discussed:
• {key_sentences[0] if key_sentences else 'Various topics covered in the document'}
• {key_sentences[1] if len(key_sentences) > 1 else 'Important information presented'}
• {key_sentences[2] if len(key_sentences) > 2 else 'Key findings detailed'}

Key pages referenced: {', '.join(map(str, pages)) if pages else 'Throughout the document'}"""
    
    # Other response types (define, count, list, compare)
    elif any(word in question_lower for word in ['what is', 'define', 'explain']):
        return f"""According to the document{' on page ' + str(pages[0]) if pages else ''}:

{key_sentences[0] if key_sentences else 'The document provides detailed information about this topic.'}

Additional details include: {' '.join(key_sentences[1:3]) if len(key_sentences) > 1 else 'Refer to the document for comprehensive coverage.'}"""

    elif any(word in question_lower for word in ['how many', 'how much', 'number', 'count']):
        return f"""The document mentions this topic on page{'s ' if len(pages) > 1 else ' '}{', '.join(map(str, pages)) if pages else 'various pages'}.

Specific information found: {key_sentences[0] if key_sentences else 'Quantitative details are provided in the document.'}"""

    elif any(word in question_lower for word in ['list', 'enumerate', 'items']):
        items = [f"{i+1}. {s}" for i, s in enumerate(key_sentences[:4])]
        response = "Based on the document:\n\n" + ("\n".join(items) if items else "• Relevant items discussed")
        if pages:
            response += f"\n\nReference pages: {', '.join(map(str, pages))}"
        return response

    elif any(word in question_lower for word in ['compare', 'difference', 'similar']):
        return f"""The document discusses this on page{'s ' if len(pages) > 1 else ' '}{', '.join(map(str, pages)) if pages else 'multiple pages'}.

Key points of comparison:
• {key_sentences[0] if key_sentences else 'Various aspects are compared'}
• {key_sentences[1] if len(key_sentences) > 1 else 'Differences and similarities are highlighted'}
• {key_sentences[2] if len(key_sentences) > 2 else 'Detailed analysis is provided'}"""

    else:
        response = f"""Based on the document{' (pages ' + ', '.join(map(str, pages)) + ')' if pages else ''}:

{key_sentences[0] if key_sentences else 'The document contains relevant information about this topic.'}
"""
        if len(key_sentences) > 1:
            response += "\nAdditional details:\n"
            for i, sentence in enumerate(key_sentences[1:4], 2):
                response += f"• {sentence}\n"
        response += "\nFor more specific information, please refer to the highlighted sections in the document."
        return response

async def stream_response(response_text: str):
    words = response_text.split()
    for i, word in enumerate(words):
        await asyncio.sleep(0.03)
        yield word + (" " if i < len(words) - 1 else "")

# ========== API ENDPOINTS ==========
@app.get("/")
async def root():
    return {
        "message": "Local Document Intelligence API",
        "status": "running",
        "version": "4.0",
        "model": "Context-Aware Response Engine",
        "features": [
            "100% Local - No API keys needed",
            "PDF Upload & Processing",
            "Vector Search with ChromaDB",
            "Intelligent Context-Aware Responses",
            "Chat Memory",
            "Document Highlights"
        ]
    }

@app.get("/health")
async def health():
    doc_count = 0
    if vectorstore:
        try:
            doc_count = len(vectorstore.get()["documents"])
        except:
            pass

    return {
        "status": "healthy",
        "sessions": len(chat_sessions),
        "documents": len(documents_store),
        "vectorstore_docs": doc_count,
        "response_engine": "context-aware"
    }

# ------------------ UPLOAD, QUERY, STREAM, CHAT ENDPOINTS ------------------
# (All other endpoints remain unchanged but internally use len(vectorstore.get()["documents"]) instead of _collection.count())
# ... (your previous code logic remains the same)

# ------------------ START SERVER ------------------
if __name__ == "__main__":
    import uvicorn

    os.environ['ANONYMIZED_TELEMETRY'] = 'False'

    try:
        initialize_vectorstore()
    except Exception as e:
        print(f"⚠️ Could not initialize vectorstore: {e}")
        print("⚠️ Starting with empty vectorstore")

    print("\n" + "="*60)
    print("🚀 LOCAL DOCUMENT INTELLIGENCE API")
    print("🌐 Server: http://127.0.0.1:8001")
    print("📄 API Docs: http://127.0.0.1:8001/docs")
    print("="*60 + "\n")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info"
    )
