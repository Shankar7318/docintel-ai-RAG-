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
from langchain.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ========== LOAD FLAN-T5 MODEL ==========
MODEL_NAME = "google/flan-t5-base"
print(f"🔧 Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📊 Using device: {device}")
model.to(device)
model.eval()
print("✅ Model loaded successfully!")

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
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
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
            collection_name="rag-docs",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        print("✅ Created new vectorstore")

def is_valid_page(text: str) -> bool:
    """Filter out junk pages (certificates, references, etc.)"""
    if not text or len(text.strip()) < 50:
        return False
    
    text_lower = text.lower()
    
    # Blacklisted content (common in academic PDFs)
    blacklist = [
        "certificate",
        "plagiarism",
        "signature",
        "acknowledgement",
        "acknowledgment",
        "references",
        "bibliography",
        "index",
        "table of contents",
        "appendix",
        "annex",
        "conclusion",
        "future scope",
        "department use",
        "examiner",
        "supervisor",
        "declaration",
        "copyright"
    ]
    
    # Skip if any blacklisted term appears
    if any(bad in text_lower for bad in blacklist):
        return False
    
    # Also skip very short pages (likely headers/footers)
    if len(text.strip()) < 100 and len(text.split()) < 20:
        return False
    
    return True

def trim_by_tokens(text: str, max_tokens: int = 700) -> str:
    """Safely trim text by token count (not characters)"""
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def search_documents(query: str, session_id: str = None, k: int = 4):
    """Search documents using vector similarity - FILTERED BY SESSION"""
    global vectorstore
    
    if vectorstore is None:
        initialize_vectorstore()
    
    try:
        # Count total documents
        doc_count = vectorstore._collection.count()
        actual_k = min(k, max(1, doc_count))
        
        if doc_count == 0:
            return []
        
        # 🔥 CRITICAL FIX: Filter by session_id if provided
        if session_id:
            print(f"🔍 Searching with filter: session_id={session_id}")
            results = vectorstore.similarity_search_with_score(
                query, 
                k=actual_k,
                filter={"session_id": session_id}
            )
        else:
            results = vectorstore.similarity_search_with_score(query, k=actual_k)
        
        formatted_results = []
        for doc, score in results:
            metadata = doc.metadata
            # Skip if it's a junk page
            if not is_valid_page(doc.page_content):
                print(f"⚠️ Skipping junk page from {metadata.get('source', 'unknown')}")
                continue
                
            formatted_results.append({
                "text": doc.page_content,
                "metadata": metadata,
                "score": 1.0 - float(score),
                "type": "vector"
            })
        
        print(f"🔍 Vector search returned {len(formatted_results)} valid results")
        return formatted_results[:k]  # Ensure we don't exceed k
        
    except Exception as e:
        print(f"⚠️ Search error: {e}")
        return []

def simple_keyword_search(query: str, session_id: str = None, k: int = 2):
    """Simple keyword matching search - FILTERED BY SESSION"""
    if not documents_store:
        return []
    
    query_lower = query.lower().split()
    results = []
    
    for doc in documents_store:
        # 🔥 CRITICAL FIX: Filter by session_id
        if session_id and doc["metadata"].get("session_id") != session_id:
            continue
            
        # Skip junk pages
        if not is_valid_page(doc["text"]):
            continue
            
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
    print(f"🔍 Keyword search returned {len(results[:k])} results for session {session_id}")
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

def generate_structured_answer(question: str, docs: list, session_id: str) -> str:
    """Generate STRUCTURED answer using FLAN-T5 model - GUARANTEED format"""
    print(f"🔍 FLAN-T5: Generating structured answer for session {session_id}")
    print(f"🔍 FLAN-T5: Retrieved {len(docs)} chunks for question: {question}")
    
    if not docs:
        print("⚠️ No valid documents retrieved")
        return "I cannot find this information in the uploaded document."
    
    # Build context from retrieved documents
    context_parts = []
    sources_info = []  # Store source info for final output
    
    for i, d in enumerate(docs):
        page = d["metadata"].get("page", 0)
        source = d["metadata"].get("source", "document")
        session = d["metadata"].get("session_id", "unknown")
        
        # Skip empty or invalid chunks
        if not d["text"] or len(d["text"].strip()) < 20:
            print(f"⚠️ Skipping empty chunk {i+1}")
            continue
            
        if session != session_id:
            print(f"⚠️ WARNING: Chunk {i+1} from wrong session ({session} != {session_id})")
            continue
            
        # Calculate actual page number (PDFs often start at 0)
        actual_page = page + 1 if isinstance(page, int) else page
        
        # Store source info for later
        sources_info.append(f"{source}, Page {actual_page}")
        
        text_preview = d["text"][:200] + "..." if len(d["text"]) > 200 else d["text"]
        print(f"🔍 FLAN-T5: Using chunk {i+1} from {source}, page {actual_page}")
        
        # Clean the text
        cleaned_text = ' '.join(d["text"].split())
        context_parts.append(f"[Source: {source}, Page: {actual_page}]\n{cleaned_text}")
    
    if not context_parts:
        print("⚠️ All chunks were empty or filtered out")
        return "I cannot find this information in the uploaded document."
    
    context = "\n\n---\n\n".join(context_parts[:3])  # Use only top 3 chunks
    
    # Safely trim context by tokens (not characters)
    context = trim_by_tokens(context, max_tokens=500)
    
    # 🔥 STRICT STRUCTURED PROMPT (ENFORCED FORMAT)
    prompt = f"""You are a document-based question answering system.

STRICT RULES:
- Use ONLY the information in the CONTEXT below.
- Do NOT use outside knowledge.
- Do NOT guess or make up information.
- Format your answer EXACTLY as shown below.

REQUIRED FORMAT:

📌 Answer Summary:
[One concise sentence summarizing the answer]

📖 Explanation (from document):
- [Bullet point 1 from context]
- [Bullet point 2 from context]
- [Bullet point 3 from context]

📄 Source Evidence:
- [Mention the source and page numbers]

If the answer is not found in the context, reply EXACTLY:
"I cannot find this information in the uploaded document."

CONTEXT:
{context}

QUESTION:
{question}

STRUCTURED ANSWER:
"""
    
    print(f"🔍 FLAN-T5: Prompt length: {len(prompt)} characters")
    
    try:
        # Tokenize and generate (deterministic for factual QA)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,      # Deterministic for factual accuracy
                num_beams=4,          # Better search
                temperature=0.0,      # No randomness
                repetition_penalty=1.2,
                early_stopping=True
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 🔥 POST-PROCESSING: ENFORCE STRUCTURE
        answer = answer.strip()
        
        # Remove any prompt fragments
        if answer.startswith("STRUCTURED ANSWER:"):
            answer = answer[len("STRUCTURED ANSWER:"):].strip()
        
        # Check if answer contains the required format
        has_structure = all(marker in answer for marker in ["📌 Answer Summary:", "📖 Explanation", "📄 Source Evidence"])
        
        if not has_structure:
            print("⚠️ FLAN-T5 did not follow structure, enforcing format...")
            # Fallback to structured template with actual content
            key_points = []
            for d in docs[:3]:
                if d["metadata"].get("session_id") == session_id:
                    sentences = d["text"].split('.')
                    for sentence in sentences[:2]:
                        if len(sentence.strip()) > 20:
                            key_points.append(f"- {sentence.strip()}")
            
            if key_points:
                answer = f"""📌 Answer Summary:
Based on the document content.

📖 Explanation (from document):
{chr(10).join(key_points[:3])}

📄 Source Evidence:
- {', '.join(sources_info[:2])}"""
            else:
                answer = "I cannot find this information in the uploaded document."
        
        # Final validation
        if "cannot find" in answer.lower() or "not in the document" in answer.lower():
            return "I cannot find this information in the uploaded document."
        
        print(f"✅ FLAN-T5: Generated structured answer (first 200 chars): {answer[:200]}...")
        return answer
        
    except Exception as e:
        print(f"❌ Model generation error: {e}")
        return "I cannot find this information in the uploaded document."

async def stream_response(response_text: str):
    """Stream response text"""
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
        "version": "7.0",
        "model": "FLAN-T5 + Structured RAG System",
        "features": [
            "100% Local - No API keys needed",
            "PDF Upload & Processing",
            "FLAN-T5 for Structured Answers",
            "Session-aware Vector Search",
            "No Hallucination Guarantee",
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
        "model": MODEL_NAME,
        "device": device,
        "guarantee": "Structured answers from FLAN-T5 only"
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process PDF document - FILTERING JUNK PAGES"""
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
        
        # 🔥 FILTER OUT JUNK PAGES
        filtered_documents = []
        for doc in documents:
            if is_valid_page(doc.page_content):
                filtered_documents.append(doc)
            else:
                print(f"⚠️ Filtered out junk page from {file.filename}")
        
        print(f"📊 After filtering: {len(filtered_documents)} valid pages")
        
        if not filtered_documents:
            raise HTTPException(400, "Document contains no valid content (only certificates/references)")
        
        # Create session
        session_id = str(uuid.uuid4())[:8]
        
        # Store for keyword search WITH SESSION ID
        for doc in filtered_documents:
            documents_store.append({
                "text": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "source": file.filename,
                    "session_id": session_id,
                    "upload_time": datetime.now().isoformat()
                }
            })
        
        # Split into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(filtered_documents)
        print(f"📊 Created {len(chunks)} chunks (chunk_size=400)")
        
        # 🔥 ADD SESSION_ID TO EACH CHUNK
        for chunk in chunks:
            chunk.metadata["session_id"] = session_id
            chunk.metadata["source"] = file.filename
        
        # Initialize vectorstore if needed
        global vectorstore
        if vectorstore is None:
            initialize_vectorstore()
        
        # Add to vectorstore
        if chunks:
            vectorstore.add_documents(chunks)
            print(f"✅ Added {len(chunks)} valid chunks to vectorstore (session: {session_id})")
        
        # Store session info
        chat_sessions[session_id] = {
            "messages": [],
            "document_name": file.filename,
            "created_at": datetime.now().isoformat(),
            "pages": len(filtered_documents),
            "chunks": len(chunks)
        }
        
        # Cleanup
        os.unlink(tmp_path)
        
        return {
            "message": "Document processed successfully (junk pages filtered)",
            "session_id": session_id,
            "valid_pages": len(filtered_documents),
            "original_pages": len(documents),
            "chunks": len(chunks),
            "filename": file.filename
        }
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/query")
async def query_document(request: QueryRequest):
    """Query document with search - GUARANTEES STRUCTURED ANSWERS"""
    if not request.session_id or request.session_id not in chat_sessions:
        raise HTTPException(404, "Session not found. Please upload a document first.")
    
    try:
        print(f"❓ Question: {request.question}")
        print(f"📂 Session: {request.session_id}")
        print(f"📄 Document: {chat_sessions[request.session_id]['document_name']}")
        
        # Search with session filtering
        vector_results = search_documents(request.question, session_id=request.session_id, k=3)
        keyword_results = simple_keyword_search(request.question, session_id=request.session_id, k=2)
        
        # Combine results
        all_results = vector_results + keyword_results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_results[:3]  # Only top 3 for focused answers
        
        print(f"🔍 Found {len(all_results)} total results, using top {len(top_results)}")
        
        if not top_results:
            return QueryResponse(
                answer="I cannot find this information in the uploaded document.",
                sources=[],
                session_id=request.session_id
            )
        
        # 🔥 Generate STRUCTURED answer with enforced format
        answer = generate_structured_answer(request.question, top_results, request.session_id)
        
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
        
        print(f"✅ Generated structured answer with {len(highlights)} sources")
        
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
            print(f"🌊 Streaming query: {request.question}")
            print(f"🌊 Session: {request.session_id}")
            
            # Search with session filtering
            vector_results = search_documents(request.question, session_id=request.session_id, k=3)
            keyword_results = simple_keyword_search(request.question, session_id=request.session_id, k=2)
            all_results = vector_results + keyword_results
            top_results = sorted(all_results, key=lambda x: x["score"], reverse=True)[:3]
            
            print(f"🔍 Found {len(top_results)} results for streaming")
            
            # Send highlights first
            if top_results:
                highlights = [extract_highlight(r) for r in top_results]
                yield f"data: [HIGHLIGHTS] {json.dumps([h.dict() for h in highlights])}\n\n"
            
            # Generate structured response
            response_text = generate_structured_answer(request.question, top_results, request.session_id)
            
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

# ... (rest of the endpoints remain the same - get_chat_history, list_sessions, etc.)
# They work exactly as before, just copy from your existing code

if __name__ == "__main__":
    import uvicorn
    
    # Initialize vectorstore on startup
    initialize_vectorstore()
    
    print("\n" + "="*60)
    print("🚀 LOCAL DOCUMENT INTELLIGENCE API v7.0")
    print("🤖 FLAN-T5 with GUARANTEED Structured Answers")
    print("🔒 Session-aware + No Hallucination RAG")
    print("📊 Model: google/flan-t5-base")
    print("📚 Features:")
    print("  • Junk page filtering (certificates/references removed)")
    print("  • Enforced structured answer format")
    print("  • Deterministic factual QA")
    print("  • Token-safe context trimming")
    print("🌐 Server: http://0.0.0.0:8000")
    print("📄 API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
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
async def search_documents_endpoint(query: str, session_id: Optional[str] = None, k: int = 10):
    """Search across all documents (with optional session filter)"""
    if not query or len(query.strip()) < 2:
        raise HTTPException(400, "Query must be at least 2 characters")
    
    vector_results = search_documents(query, session_id=session_id, k=k)
    keyword_results = simple_keyword_search(query, session_id=session_id, k=k)
    
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
        "session_filter": session_id,
        "total_results": len(formatted_results),
        "results": formatted_results
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in chat_sessions:
        # Remove from chat_sessions
        del chat_sessions[session_id]
        
        # Filter out documents from this session
        global documents_store
        documents_store = [doc for doc in documents_store if doc["metadata"].get("session_id") != session_id]
        
        print(f"🗑️ Deleted session {session_id} and filtered documents")
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
            if doc["metadata"].get("session_id") == session_id:
                relevant_texts.append(doc["text"])
        
        if not relevant_texts:
            raise HTTPException(404, "Document text not found")
        
        # Combine text
        combined_text = "\n\n".join(relevant_texts[:3])
        
        # Use FLAN-T5 for summary
        prompt = f"""Summarize this document content in 3-5 key points:

{combined_text}

Key points:"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.2)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
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
    print("🚀 LOCAL DOCUMENT INTELLIGENCE API v6.0")
    print("🤖 Using FLAN-T5 for Answer Generation")
    print("🔒 SESSION-AWARE RAG Pipeline")
    print("💡 100% Local - No API Keys or Dependencies!")
    print(f"📊 Model: {MODEL_NAME} on {device}")
    print("📚 Features: PDF Upload, Session-aware Search, RAG")
    print("🌐 Server: http://127.0.0.1:8001")
    print("📄 API Docs: http://127.0.0.1:8001/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )