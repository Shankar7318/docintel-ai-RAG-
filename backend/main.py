from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, AsyncGenerator
import os
import tempfile
import uuid
import json
import asyncio
from datetime import datetime


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



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
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

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
        vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings,
            collection_name="documents"
        )
        print(f"✅ Loaded existing vectorstore with {vectorstore._collection.count()} documents")
    except:
        vectorstore = Chroma(
            collection_name="documents",
            embedding_function=embeddings,
            persist_directory=chroma_dir
        )
        print("✅ Created new vectorstore")

def search_documents(query: str, k: int = 3):
    """Search documents using vector similarity"""
    global vectorstore
    
    if vectorstore is None:
        initialize_vectorstore()
    
    try:
        doc_count = vectorstore._collection.count()
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
    """Extract highlight information"""
    metadata = doc_result.get("metadata", {})
    return HighlightInfo(
        file=metadata.get("source", "document.pdf"),
        page=metadata.get("page", 0) + 1,
        text=doc_result["text"][:300],
        score=round(doc_result.get("score", 0.5), 2)
    )

def generate_intelligent_response(context: str, question: str, highlights: List[Dict]):
    """Generate intelligent response based on context"""
    
    # Extract key information from context
    pages = []
    key_sentences = []
    
    for highlight in highlights:
        if highlight["metadata"].get("page") is not None:
            pages.append(highlight["metadata"]["page"] + 1)
        
        # Extract key sentences from highlight
        text = highlight["text"]
        sentences = text.split('.')
        for sentence in sentences[:3]:  # Take first 3 sentences
            if len(sentence.strip()) > 20:
                key_sentences.append(sentence.strip())
    
    # Remove duplicates
    pages = sorted(list(set(pages)))
    key_sentences = list(set(key_sentences))[:5]  # Limit to 5 unique sentences
    
    # Question analysis
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['summary', 'summarize', 'overview', 'overall']):
        return f"""Based on the document, here's a summary:

Main topics discussed:
• {key_sentences[0] if key_sentences else 'Various topics covered in the document'}
• {key_sentences[1] if len(key_sentences) > 1 else 'Important information presented'}
• {key_sentences[2] if len(key_sentences) > 2 else 'Key findings detailed'}

Key pages referenced: {', '.join(map(str, pages)) if pages else 'Throughout the document'}"""

    elif any(word in question_lower for word in ['what is', 'define', 'explain']):
        return f"""According to the document{' on page ' + str(pages[0]) if pages else ''}:

{key_sentences[0] if key_sentences else 'The document provides detailed information about this topic.'}

Additional details include: {' '.join(key_sentences[1:3]) if len(key_sentences) > 1 else 'Refer to the document for comprehensive coverage.'}"""

    elif any(word in question_lower for word in ['how many', 'how much', 'number', 'count']):
        return f"""The document mentions this topic on page{'s ' if len(pages) > 1 else ' '}{', '.join(map(str, pages)) if pages else 'various pages'}.

Specific information found: {key_sentences[0] if key_sentences else 'Quantitative details are provided in the document.'}"""

    elif any(word in question_lower for word in ['list', 'enumerate', 'items']):
        items = []
        for i, sentence in enumerate(key_sentences[:4], 1):
            items.append(f"{i}. {sentence}")
        
        response = "Based on the document:\n\n"
        if items:
            response += "\n".join(items)
        else:
            response += "• Relevant items are discussed in the document\n"
            response += "• Specific details can be found in the text\n"
            response += "• Multiple aspects are covered"
        
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
        # Generic response
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
    """Stream response text"""
    words = response_text.split()
    for i, word in enumerate(words):
        await asyncio.sleep(0.03)  # Small delay for streaming effect
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
        ],
        "endpoints": [
            "POST /upload - Upload PDF",
            "POST /query - Ask questions",
            "POST /query/stream - Streaming query",
            "GET /chat/{session_id} - Get chat history",
            "GET /sessions - List sessions",
            "GET /search?query=text - Search documents"
        ]
    }

@app.get("/health")
async def health():
    doc_count = 0
    if vectorstore:
        try:
            doc_count = vectorstore._collection.count()
        except:
            pass
    
    return {
        "status": "healthy",
        "sessions": len(chat_sessions),
        "documents": len(documents_store),
        "vectorstore_docs": doc_count,
        "response_engine": "context-aware"
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process PDF document"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")
    
    try:
        print(f"📄 Processing: {file.filename}")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        print(f"📖 Loaded {len(documents)} pages")
        
        # Store for keyword search
        for doc in documents:
            documents_store.append({
                "text": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "source": file.filename,
                    "upload_time": datetime.now().isoformat()
                }
            })
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"📊 Created {len(chunks)} chunks")
        
        # Initialize vectorstore if needed
        global vectorstore
        if vectorstore is None:
            initialize_vectorstore()
        
        # Add to vectorstore
        if chunks:
            vectorstore.add_documents(chunks)
            vectorstore.persist()
            print(f"✅ Added to vectorstore")
        
        # Create session
        session_id = str(uuid.uuid4())[:8]
        chat_sessions[session_id] = {
            "messages": [],
            "document_name": file.filename,
            "created_at": datetime.now().isoformat(),
            "pages": len(documents),
            "chunks": len(chunks)
        }
        
        # Cleanup
        os.unlink(tmp_path)
        
        return {
            "message": "Document processed successfully",
            "session_id": session_id,
            "pages": len(documents),
            "chunks": len(chunks),
            "filename": file.filename
        }
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/query")
async def query_document(request: QueryRequest):
    """Query document with search"""
    if not request.session_id or request.session_id not in chat_sessions:
        raise HTTPException(404, "Session not found. Please upload a document first.")
    
    try:
        print(f"❓ Question: {request.question}")
        
        # Search for relevant content
        vector_results = search_documents(request.question, k=2)
        keyword_results = simple_keyword_search(request.question, k=2)
        
        # Combine results
        all_results = vector_results + keyword_results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_results[:3]
        
        if not top_results:
            return QueryResponse(
                answer="No relevant information found in the document. Please try a different question.",
                sources=[],
                session_id=request.session_id
            )
        
        # Generate intelligent response
        answer = generate_intelligent_response("Document context", request.question, top_results)
        
        # Extract highlights
        highlights = [extract_highlight(r) for r in top_results]
        
        # Store conversation
        session = chat_sessions[request.session_id]
        session["messages"].append({
            "role": "user",
            "content": request.question,
            "timestamp": datetime.now().isoformat()
        })
        session["messages"].append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().isoformat(),
            "sources": [h.dict() for h in highlights]
        })
        
        print(f"✅ Generated answer with {len(highlights)} sources")
        
        return QueryResponse(
            answer=answer,
            sources=highlights,
            session_id=request.session_id
        )
        
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        raise HTTPException(500, f"Query failed: {str(e)}")

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Streaming query endpoint"""
    if not request.session_id or request.session_id not in chat_sessions:
        async def error_stream():
            yield "data: Session not found. Please upload a document first.\n\n"
        
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    async def generate():
        try:
            # Search for relevant content
            vector_results = search_documents(request.question, k=2)
            keyword_results = simple_keyword_search(request.question, k=1)
            all_results = vector_results + keyword_results
            top_results = sorted(all_results, key=lambda x: x["score"], reverse=True)[:2]
            
            # Send highlights first
            if top_results:
                highlights = [extract_highlight(r) for r in top_results]
                yield f"data: [HIGHLIGHTS] {json.dumps([h.dict() for h in highlights])}\n\n"
            
            # Generate response
            response_text = generate_intelligent_response("", request.question, top_results)
            
            # Stream the response
            async for chunk in stream_response(response_text):
                yield f"data: {chunk}\n\n"
            
            # Store conversation
            session = chat_sessions[request.session_id]
            session["messages"].append({
                "role": "user",
                "content": request.question,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"❌ Stream error: {e}")
            yield f"data: Error: {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/chat/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    if session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    
    return {
        "session_id": session_id,
        "document_name": chat_sessions[session_id]["document_name"],
        "created_at": chat_sessions[session_id]["created_at"],
        "messages": chat_sessions[session_id]["messages"]
    }

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, data in chat_sessions.items():
        sessions.append({
            "session_id": session_id,
            "document_name": data["document_name"],
            "created_at": data["created_at"],
            "message_count": len(data["messages"]),
            "pages": data.get("pages", 0),
            "chunks": data.get("chunks", 0)
        })
    
    return {"sessions": sessions}

@app.get("/highlights/{session_id}")
async def get_highlights(session_id: str):
    """Get all highlights from a session"""
    if session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    
    session = chat_sessions[session_id]
    highlights = []
    
    for message in session["messages"]:
        if message.get("sources"):
            highlights.extend(message["sources"])
    
    # Deduplicate
    unique_highlights = []
    seen = set()
    for h in highlights:
        key = f"{h['file']}-{h['page']}-{h['text'][:50]}"
        if key not in seen:
            seen.add(key)
            unique_highlights.append(h)
    
    return {
        "session_id": session_id,
        "document_name": session["document_name"],
        "highlights": unique_highlights[:50]
    }

@app.get("/search")
async def search_documents_endpoint(query: str, k: int = 10):
    """Search across all documents"""
    if not query or len(query.strip()) < 2:
        raise HTTPException(400, "Query must be at least 2 characters")
    
    vector_results = search_documents(query, k=k)
    keyword_results = simple_keyword_search(query, k=k)
    
    all_results = vector_results + keyword_results
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    formatted_results = []
    for result in all_results[:k]:
        metadata = result["metadata"]
        formatted_results.append({
            "text": result["text"][:400] + ("..." if len(result["text"]) > 400 else ""),
            "file": metadata.get("source", "Unknown"),
            "page": metadata.get("page", 0) + 1,
            "score": round(result["score"], 3),
            "type": result.get("type", "vector")
        })
    
    return {
        "query": query,
        "total_results": len(formatted_results),
        "results": formatted_results
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(404, "Session not found")

@app.get("/summary/{session_id}")
async def generate_summary(session_id: str):
    """Generate document summary"""
    if session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    
    try:
        # Get all text from this session's document
        doc_name = chat_sessions[session_id]["document_name"]
        relevant_texts = []
        
        for doc in documents_store:
            if doc["metadata"].get("source") == doc_name:
                relevant_texts.append(doc["text"])
        
        if not relevant_texts:
            raise HTTPException(404, "Document text not found")
        
        # Combine text
        combined_text = "\n\n".join(relevant_texts[:3])
        
        # Simple summary generation
        sentences = combined_text.split('.')
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 50][:5]
        
        if not key_sentences:
            key_sentences = [
                "The document covers important topics relevant to its subject matter.",
                "Key information is presented throughout the document.",
                "Various aspects are discussed in detail."
            ]
        
        summary = f"""Document Summary:

Main Content:
• {key_sentences[0]}
• {key_sentences[1] if len(key_sentences) > 1 else 'Comprehensive coverage of the subject matter'}
• {key_sentences[2] if len(key_sentences) > 2 else 'Detailed analysis and information'}

Additional Points:
• {key_sentences[3] if len(key_sentences) > 3 else 'Relevant data and findings presented'}
• {key_sentences[4] if len(key_sentences) > 4 else 'Conclusion and recommendations included'}

This summary is based on document: {doc_name}"""
        
        return {
            "session_id": session_id,
            "document_name": doc_name,
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(500, f"Summary generation failed: {str(e)}")

@app.get("/export/{session_id}")
async def export_chat(session_id: str):
    """Export chat history as JSON"""
    if session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    
    session = chat_sessions[session_id]
    export_data = {
        "session_id": session_id,
        "document_name": session["document_name"],
        "created_at": session["created_at"],
        "messages": session["messages"],
        "exported_at": datetime.now().isoformat()
    }
    
    return JSONResponse(content=export_data)

if __name__ == "__main__":
    import uvicorn
    
    # Initialize vectorstore on startup
    initialize_vectorstore()
    
    print("\n" + "="*60)
    print("🚀 LOCAL DOCUMENT INTELLIGENCE API")
    print("🤖 100% Local - No API Keys or Dependencies!")
    print("💡 Intelligent Context-Aware Responses")
    print("📚 Features: PDF Upload, Vector Search, Chat Memory")
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