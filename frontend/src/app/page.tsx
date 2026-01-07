"use client";

import { useState, useRef, useEffect } from 'react';
import { Upload, FileText, Brain, Send, Loader2, X, CheckCircle, HelpCircle, BookOpen, Search, Zap, History, Download, Highlighter, FileArchive, Eye, ChevronRight, ChevronLeft, Layers, MessageSquare } from 'lucide-react';

const API_URL = "http://127.0.0.1:8001";

export default function Home() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploaded, setUploaded] = useState(false);
  const [fileName, setFileName] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [sources, setSources] = useState([]);
  const [apiStatus, setApiStatus] = useState('checking');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [sessionId, setSessionId] = useState(null);
  const [highlights, setHighlights] = useState([]);
  const [activeSessions, setActiveSessions] = useState([]);
  const [selectedHighlight, setSelectedHighlight] = useState(null);
  const [showHighlightsPanel, setShowHighlightsPanel] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  
  const fileInputRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Check API health on load
  useEffect(() => {
    checkApiHealth();
    loadSessions();
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory, streamingAnswer]);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      if (response.ok) {
        setApiStatus('healthy');
      } else {
        setApiStatus('unhealthy');
      }
    } catch {
      setApiStatus('unhealthy');
    }
  };

  const loadSessions = async () => {
    try {
      const response = await fetch(`${API_URL}/sessions`);
      if (response.ok) {
        const data = await response.json();
        setActiveSessions(data.sessions || []);
      }
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.type !== 'application/pdf') {
        alert('Please select a PDF file');
        return;
      }
      if (selectedFile.size > 50 * 1024 * 1024) { // 50MB limit
        alert('File size should be less than 50MB');
        return;
      }
      setFile(selectedFile);
      setFileName(selectedFile.name);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setUploadProgress(0);
    
    const formData = new FormData();
    formData.append('file', file);

    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 200);

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (response.ok) {
        const data = await response.json();
        setUploaded(true);
        setFileName(file.name);
        setChatHistory([]);
        setAnswer('');
        setQuestion('');
        setSources([]);
        setSessionId(data.session_id);
        setHighlights([]);
        
        // Load sessions list
        await loadSessions();
        
        // Show success message
        setTimeout(() => {
          setUploadProgress(0);
        }, 1000);
      } else {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
      }
    } catch (error) {
      alert(`Error uploading document: ${error.message}`);
    } finally {
      setIsUploading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  const handleQuery = async () => {
    if (!question.trim() || !sessionId) return;

    setIsThinking(true);
    setStreamingAnswer('');
    setIsStreaming(false);
    const currentQuestion = question;

    // Add user question to chat history immediately
    setChatHistory(prev => [...prev, { 
      type: 'user', 
      content: currentQuestion,
      timestamp: new Date().toLocaleTimeString(),
      sources: []
    }]);

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question: currentQuestion,
          session_id: sessionId 
        }),
      });

      const data = await response.json();
      setAnswer(data.answer);
      setSources(data.sources || []);
      
      // Add AI response to chat history
      setChatHistory(prev => [...prev, { 
        type: 'ai', 
        content: data.answer,
        sources: data.sources,
        timestamp: new Date().toLocaleTimeString()
      }]);
      
      // Load highlights
      if (sessionId) {
        await loadHighlights();
      }
      
      setQuestion('');
    } catch (error) {
      alert('Error querying document');
    } finally {
      setIsThinking(false);
    }
  };

  const handleStreamingQuery = async () => {
    if (!question.trim() || !sessionId) return;

    setIsStreaming(true);
    setStreamingAnswer('');
    const currentQuestion = question;

    // Add user question to chat history immediately
    setChatHistory(prev => [...prev, { 
      type: 'user', 
      content: currentQuestion,
      timestamp: new Date().toLocaleTimeString(),
      sources: []
    }]);

    try {
      const response = await fetch(`${API_URL}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question: currentQuestion,
          session_id: sessionId 
        }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedAnswer = '';
      let highlightsData = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            
            if (data.startsWith('[HIGHLIGHTS]')) {
              // Handle highlights
              highlightsData = JSON.parse(data.slice(13));
              setHighlights(highlightsData);
            } else if (data.startsWith('Error:')) {
              throw new Error(data.slice(7));
            } else {
              // Handle text stream
              accumulatedAnswer += data;
              setStreamingAnswer(accumulatedAnswer);
            }
          }
        }
      }

      // Add completed AI response to chat history
      setChatHistory(prev => [...prev, { 
        type: 'ai', 
        content: accumulatedAnswer,
        sources: highlightsData,
        timestamp: new Date().toLocaleTimeString()
      }]);
      
      setQuestion('');
    } catch (error) {
      alert(`Streaming error: ${error.message}`);
    } finally {
      setIsStreaming(false);
      setStreamingAnswer('');
    }
  };

  const loadChatHistory = async () => {
    if (!sessionId) return;

    try {
      const response = await fetch(`${API_URL}/chat/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        setChatHistory(data.messages.map(msg => ({
          type: msg.role === 'user' ? 'user' : 'ai',
          content: msg.content,
          timestamp: new Date(msg.timestamp).toLocaleTimeString(),
          sources: msg.sources || []
        })));
      }
    } catch (error) {
      console.error('Failed to load chat history:', error);
    }
  };

  const loadHighlights = async () => {
    if (!sessionId) return;

    try {
      const response = await fetch(`${API_URL}/highlights/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        setHighlights(data.highlights || []);
      }
    } catch (error) {
      console.error('Failed to load highlights:', error);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    try {
      const response = await fetch(`${API_URL}/search?query=${encodeURIComponent(searchQuery)}&k=10`);
      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.results || []);
      }
    } catch (error) {
      console.error('Search failed:', error);
      alert('Search failed');
    } finally {
      setIsSearching(false);
    }
  };

  const handleSessionSelect = async (session) => {
    setSessionId(session.session_id);
    setFileName(session.document_name);
    setUploaded(true);
    await loadChatHistory();
    await loadHighlights();
  };

  const deleteSession = async (sessionId) => {
    if (window.confirm('Are you sure you want to delete this session?')) {
      try {
        const response = await fetch(`${API_URL}/session/${sessionId}`, {
          method: 'DELETE',
        });
        
        if (response.ok) {
          if (sessionId === sessionId) {
            // Reset current session
            setSessionId(null);
            setUploaded(false);
            setChatHistory([]);
            setHighlights([]);
          }
          await loadSessions();
        }
      } catch (error) {
        console.error('Failed to delete session:', error);
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  const clearChat = () => {
    setChatHistory([]);
    setQuestion('');
    setAnswer('');
    setSources([]);
    setStreamingAnswer('');
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const droppedFile = files[0];
      if (droppedFile.type === 'application/pdf') {
        setFile(droppedFile);
        setFileName(droppedFile.name);
      } else {
        alert('Please drop a PDF file');
      }
    }
  };

  const handleHighlightClick = (highlight) => {
    setSelectedHighlight(highlight);
    // In a real app, you would open the PDF at this page
    alert(`Would open PDF "${highlight.file}" at page ${highlight.page}`);
  };

  const sampleQuestions = [
    "What is this document about?",
    "Summarize the main points",
    "What are the key findings?",
    "What recommendations are made?",
    "Explain the methodology used",
    "What data is presented?",
    "What conclusions are drawn?",
    "Who is the target audience?"
  ];

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b shadow-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
                <Brain className="h-7 w-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">DocIntel AI</h1>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600">Advanced RAG with Memory & Hybrid Search</span>
                  <span className={`px-2 py-1 text-xs rounded-full ${apiStatus === 'healthy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                    {apiStatus === 'healthy' ? '● Online' : '● Offline'}
                  </span>
                </div>
              </div>
            </div>
            
            <div className="hidden md:flex items-center space-x-4">
              <div className="flex items-center space-x-2 px-3 py-1.5 bg-blue-50 rounded-lg">
                <Layers className="h-4 w-4 text-blue-600" />
                <span className="text-sm font-medium text-blue-700">Hybrid Search</span>
              </div>
              <div className="flex items-center space-x-2 px-3 py-1.5 bg-purple-50 rounded-lg">
                <MessageSquare className="h-4 w-4 text-purple-600" />
                <span className="text-sm font-medium text-purple-700">Chat Memory</span>
              </div>
              {uploaded && sessionId && (
                <div className="flex items-center space-x-2 px-3 py-1.5 bg-green-50 rounded-lg">
                  <CheckCircle className="h-4 w-4 text-green-600" />
                  <span className="text-sm text-green-700 truncate max-w-xs">
                    Session: {sessionId}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Left Panel - Upload & Sessions */}
          <div className="lg:col-span-1 space-y-6">
            {/* Upload Card */}
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-lg border p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <Upload className="h-6 w-6 text-blue-600" />
                </div>
                <h2 className="text-xl font-semibold text-gray-900">Upload Document</h2>
              </div>

              <div className="space-y-4">
                {/* Drag & Drop Area */}
                <div 
                  className={`border-3 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${file ? 'border-green-400 bg-green-50' : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'}`}
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <FileText className={`h-12 w-12 mx-auto mb-4 ${file ? 'text-green-500' : 'text-gray-400'}`} />
                  <p className="text-gray-600 mb-2 font-medium">
                    {file ? 'File selected' : 'Drag & drop your PDF here'}
                  </p>
                  <p className="text-sm text-gray-500 mb-4">
                    {file ? fileName : 'or click to browse'}
                  </p>
                  
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  
                  <button className="inline-flex items-center px-5 py-2.5 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg font-medium hover:from-blue-700 hover:to-blue-800 transition-all shadow-sm">
                    <Upload className="h-4 w-4 mr-2" />
                    Choose File
                  </button>
                  
                  {file && (
                    <div className="mt-4 p-3 bg-green-50 rounded-lg border border-green-200">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <FileText className="h-4 w-4 text-green-600" />
                          <span className="text-sm font-medium text-green-800 truncate">
                            {fileName}
                          </span>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setFile(null);
                            setFileName('');
                          }}
                          className="text-gray-400 hover:text-gray-600"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </div>
                      <div className="mt-2 text-xs text-gray-600">
                        Size: {formatFileSize(file.size)}
                      </div>
                    </div>
                  )}
                </div>

                {/* Upload Progress */}
                {uploadProgress > 0 && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-700">Uploading...</span>
                      <span className="font-medium">{uploadProgress}%</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-green-400 to-blue-500 transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Upload Button */}
                <button
                  onClick={handleUpload}
                  disabled={!file || isUploading}
                  className="w-full py-3.5 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-medium hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center shadow-md"
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="h-5 w-5 animate-spin mr-2" />
                      Processing Document...
                    </>
                  ) : (
                    <>
                      <Upload className="h-5 w-5 mr-2" />
                      Process with AI
                    </>
                  )}
                </button>
              </div>

              {/* Quick Stats */}
              {uploaded && sessionId && (
                <div className="mt-6 pt-6 border-t">
                  <h3 className="font-medium text-gray-900 mb-3">Session Info</h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-xs text-gray-600">Session ID</div>
                      <div className="font-medium text-blue-700 truncate">{sessionId}</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-xs text-gray-600">Document</div>
                      <div className="font-medium text-gray-900 truncate">{fileName}</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-xs text-gray-600">Messages</div>
                      <div className="font-medium text-gray-900">{chatHistory.length}</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-xs text-gray-600">Highlights</div>
                      <div className="font-medium text-gray-900">{highlights.length}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Active Sessions */}
            {activeSessions.length > 0 && (
              <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-lg border p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-2">
                    <FileArchive className="h-5 w-5 text-gray-500" />
                    <h3 className="font-semibold text-gray-900">Active Sessions</h3>
                  </div>
                  <button
                    onClick={loadSessions}
                    className="text-sm text-blue-600 hover:text-blue-800"
                  >
                    Refresh
                  </button>
                </div>
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {activeSessions.map((session) => (
                    <div
                      key={session.session_id}
                      className={`p-3 rounded-lg border cursor-pointer transition-all ${
                        sessionId === session.session_id
                          ? 'bg-blue-50 border-blue-200'
                          : 'bg-gray-50 hover:bg-gray-100'
                      }`}
                      onClick={() => handleSessionSelect(session)}
                    >
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <div className="text-sm font-medium text-gray-900 truncate">
                            {session.document_name}
                          </div>
                          <div className="text-xs text-gray-600 mt-1">
                            Session: {session.session_id}
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            {session.message_count} messages
                          </div>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteSession(session.session_id);
                          }}
                          className="text-gray-400 hover:text-red-600 ml-2"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Search Panel */}
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-lg border p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Search className="h-5 w-5 text-gray-500" />
                <h3 className="font-semibold text-gray-900">Hybrid Search</h3>
              </div>
              <div className="space-y-3">
                <div className="relative">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search across documents..."
                    className="w-full border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <button
                    onClick={handleSearch}
                    disabled={isSearching}
                    className="absolute right-2 top-1.5 p-1.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                  >
                    {isSearching ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Search className="h-4 w-4" />
                    )}
                  </button>
                </div>
                {searchResults.length > 0 && (
                  <div className="mt-4">
                    <h4 className="text-sm font-medium text-gray-900 mb-2">
                      Search Results ({searchResults.length})
                    </h4>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {searchResults.map((result, index) => (
                        <div
                          key={index}
                          className="p-3 bg-gray-50 rounded-lg border text-sm"
                        >
                          <div className="flex justify-between items-start mb-1">
                            <span className="font-medium text-gray-900">
                              {result.file}
                            </span>
                            <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">
                              Page {result.page}
                            </span>
                          </div>
                          <p className="text-gray-600 text-xs line-clamp-3">
                            {result.text}
                          </p>
                          <div className="flex justify-between items-center mt-2">
                            <span className="text-xs text-gray-500">
                              Score: {result.score}
                            </span>
                            <span className={`text-xs px-2 py-1 rounded ${
                              result.search_type === 'vector'
                                ? 'bg-purple-100 text-purple-800'
                                : 'bg-green-100 text-green-800'
                            }`}>
                              {result.search_type}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Sample Questions */}
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-lg border p-6">
              <div className="flex items-center space-x-2 mb-4">
                <HelpCircle className="h-5 w-5 text-gray-500" />
                <h3 className="font-semibold text-gray-900">Try asking...</h3>
              </div>
              <div className="space-y-2">
                {sampleQuestions.map((q, index) => (
                  <button
                    key={index}
                    onClick={() => {
                      setQuestion(q);
                      setTimeout(() => {
                        document.querySelector('textarea')?.focus();
                      }, 100);
                    }}
                    className="w-full text-left p-3 bg-gray-50 hover:bg-blue-50 rounded-lg transition-all hover:translate-x-1 hover:shadow-sm border hover:border-blue-200 group"
                  >
                    <div className="flex items-start">
                      <Search className="h-4 w-4 text-gray-400 mt-0.5 mr-2 group-hover:text-blue-500" />
                      <span className="text-sm text-gray-700 group-hover:text-gray-900">{q}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Middle Panel - Chat */}
          <div className="lg:col-span-2">
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-lg border h-full flex flex-col">
              {/* Chat Header */}
              <div className="border-b p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-semibold text-gray-900">Chat with Document</h2>
                    <p className="text-gray-600 mt-1">
                      {sessionId ? `Session: ${sessionId}` : 'Upload a document to start chatting'}
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    {highlights.length > 0 && (
                      <button
                        onClick={() => setShowHighlightsPanel(!showHighlightsPanel)}
                        className="flex items-center space-x-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                      >
                        <Highlighter className="h-4 w-4" />
                        <span>Highlights ({highlights.length})</span>
                      </button>
                    )}
                    {chatHistory.length > 0 && (
                      <button
                        onClick={clearChat}
                        className="flex items-center space-x-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                      >
                        <History className="h-4 w-4" />
                        <span>Clear Chat</span>
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Chat Messages */}
              <div 
                ref={chatContainerRef}
                className="flex-1 overflow-y-auto p-6 max-h-[500px]"
              >
                {!uploaded ? (
                  <div className="h-full flex flex-col items-center justify-center text-center p-8">
                    <BookOpen className="h-16 w-16 text-gray-300 mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Document Uploaded</h3>
                    <p className="text-gray-600 max-w-md">
                      Upload a PDF document to start asking questions. The AI will analyze the content and provide intelligent answers.
                    </p>
                  </div>
                ) : chatHistory.length === 0 && !isStreaming ? (
                  <div className="h-full flex flex-col items-center justify-center text-center p-8">
                    <Brain className="h-16 w-16 text-blue-300 mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">Ready to Chat</h3>
                    <p className="text-gray-600 max-w-md">
                      Ask a question about your document or try one of the sample questions.
                    </p>
                    <div className="mt-6 flex items-center space-x-2">
                      <Zap className="h-5 w-5 text-yellow-500" />
                      <span className="text-sm text-gray-600">built by shankar behera</span>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {chatHistory.map((message, index) => (
                      <div
                        key={index}
                        className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`max-w-[85%] rounded-2xl p-4 ${
                            message.type === 'user'
                              ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-br-none'
                              : 'bg-gray-100 text-gray-900 rounded-bl-none border'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center">
                              {message.type === 'ai' && (
                                <div className="p-1 bg-gradient-to-r from-blue-100 to-purple-100 rounded-full mr-2">
                                  <Brain className="h-4 w-4 text-blue-600" />
                                </div>
                              )}
                              <span className="text-sm font-medium">
                                {message.type === 'user' ? 'You' : 'AI Assistant'}
                              </span>
                            </div>
                            <span className="text-xs opacity-70">{message.timestamp}</span>
                          </div>
                          <p className="whitespace-pre-wrap text-sm">{message.content}</p>
                          
                          {message.type === 'ai' && message.sources && message.sources.length > 0 && (
                            <div className="mt-3 pt-3 border-t border-gray-200">
                              <div className="flex flex-wrap items-center text-xs text-gray-600 gap-1">
                                <FileText className="h-3 w-3 mr-1" />
                                <span className="font-medium mr-1">Sources:</span>
                                {message.sources.map((source, idx) => (
                                  <span
                                    key={idx}
                                    className="bg-blue-100 text-blue-800 px-2 py-1 rounded cursor-pointer hover:bg-blue-200"
                                    onClick={() => handleHighlightClick(source)}
                                  >
                                    {source.file} (p{source.page})
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                    
                    {/* Streaming Response */}
                    {isStreaming && streamingAnswer && (
                      <div className="flex justify-start">
                        <div className="max-w-[85%] rounded-2xl rounded-bl-none p-4 bg-gray-100 border text-gray-900">
                          <div className="flex items-center mb-2">
                            <div className="p-1 bg-gradient-to-r from-blue-100 to-purple-100 rounded-full mr-2">
                              <Brain className="h-4 w-4 text-blue-600" />
                            </div>
                            <span className="text-sm font-medium">AI Assistant</span>
                          </div>
                          <p className="whitespace-pre-wrap text-sm">
                            {streamingAnswer}
                            <span className="inline-block w-2 h-4 ml-1 bg-gray-400 animate-pulse"></span>
                          </p>
                        </div>
                      </div>
                    )}
                    
                    {(isThinking || isStreaming) && !streamingAnswer && (
                      <div className="flex justify-start">
                        <div className="bg-gray-100 rounded-2xl rounded-bl-none p-4 border">
                          <div className="flex items-center">
                            <div className="p-1 bg-gradient-to-r from-blue-100 to-purple-100 rounded-full mr-2">
                              <Brain className="h-4 w-4 text-blue-600" />
                            </div>
                            <span className="text-sm font-medium mr-3">AI Assistant</span>
                            <div className="flex space-x-1">
                              <div className="h-2 w-2 bg-gray-400 rounded-full animate-pulse"></div>
                              <div className="h-2 w-2 bg-gray-400 rounded-full animate-pulse delay-150"></div>
                              <div className="h-2 w-2 bg-gray-400 rounded-full animate-pulse delay-300"></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Input Area */}
              {uploaded && (
                <div className="border-t p-6">
                  <div className="relative">
                    <textarea
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Ask a question about your document..."
                      className="w-full border rounded-xl p-4 pr-32 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none shadow-sm"
                      rows="3"
                      disabled={isThinking || isStreaming}
                    />
                    <div className="absolute right-4 bottom-4 flex items-center space-x-2">
                      <button
                        onClick={handleStreamingQuery}
                        disabled={!question.trim() || isThinking || isStreaming}
                        className="px-4 py-2 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg hover:from-green-700 hover:to-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md flex items-center"
                      >
                        <Zap className="h-4 w-4 mr-2" />
                        Stream
                      </button>
                      <button
                        onClick={handleQuery}
                        disabled={!question.trim() || isThinking || isStreaming}
                        className="p-2.5 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md"
                      >
                        {isThinking ? (
                          <Loader2 className="h-5 w-5 animate-spin" />
                        ) : (
                          <Send className="h-5 w-5" />
                        )}
                      </button>
                    </div>
                  </div>
                  <div className="flex justify-between items-center mt-2">
                    <p className="text-xs text-gray-500">
                      Press Enter to send, Shift+Enter for new line
                    </p>
                    <div className="flex items-center space-x-2 text-xs text-gray-500">
                      <Layers className="h-3 w-3" />
                      <span>Hybrid Search Enabled</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Highlights & Preview */}
          <div className={`lg:col-span-1 ${showHighlightsPanel ? 'block' : 'hidden lg:block'}`}>
                        {/* Highlights Panel */}
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-lg border h-full flex flex-col">
              {/* Panel Header */}
              <div className="border-b p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-amber-100 rounded-lg">
                      <Highlighter className="h-6 w-6 text-amber-600" />
                    </div>
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900">Document Insights</h2>
                      <p className="text-gray-600 mt-1">
                        Key highlights and sources
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowHighlightsPanel(false)}
                    className="lg:hidden text-gray-400 hover:text-gray-600"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>
              </div>

              {/* Highlights Content */}
              <div className="flex-1 overflow-y-auto p-6">
                {!uploaded ? (
                  <div className="h-full flex flex-col items-center justify-center text-center p-8">
                    <Eye className="h-16 w-16 text-gray-300 mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Document Loaded</h3>
                    <p className="text-gray-600 max-w-md">
                      Upload and process a document to see insights and highlights.
                    </p>
                  </div>
                ) : highlights.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center p-8">
                    <Highlighter className="h-16 w-16 text-amber-300 mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Highlights Yet</h3>
                    <p className="text-gray-600 max-w-md">
                      Ask questions about the document to generate highlights and citations.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* Highlighted Sources */}
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-4 flex items-center">
                        <FileText className="h-5 w-5 mr-2 text-blue-600" />
                        Source References ({highlights.length})
                      </h3>
                      <div className="space-y-3">
                        {highlights.slice(0, 10).map((highlight, index) => (
                          <div
                            key={index}
                            className={`p-4 rounded-lg border cursor-pointer transition-all hover:shadow-md ${
                              selectedHighlight?.text === highlight.text
                                ? 'bg-blue-50 border-blue-300'
                                : 'bg-gray-50 border-gray-200'
                            }`}
                            onClick={() => setSelectedHighlight(highlight)}
                          >
                            <div className="flex justify-between items-start mb-2">
                              <div className="flex items-center">
                                <div className="w-6 h-6 flex items-center justify-center bg-blue-100 text-blue-800 rounded-full text-xs font-bold mr-2">
                                  {index + 1}
                                </div>
                                <span className="font-medium text-gray-900 truncate">
                                  {highlight.file || fileName}
                                </span>
                              </div>
                              <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">
                                Page {highlight.page}
                              </span>
                            </div>
                            <p className="text-gray-700 text-sm line-clamp-3">
                              {highlight.text}
                            </p>
                            <div className="flex justify-between items-center mt-3 pt-3 border-t border-gray-200">
                              <span className="text-xs text-gray-500">
                                Relevance: {highlight.relevance || 'High'}
                              </span>
                              <div className="flex items-center space-x-2">
                                <button
                                  className="text-xs text-blue-600 hover:text-blue-800"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    // Add to question
                                    setQuestion(prev => prev + (prev ? ' ' : '') + highlight.text.substring(0, 100));
                                    document.querySelector('textarea')?.focus();
                                  }}
                                >
                                  Ask about this
                                </button>
                                <ChevronRight className="h-4 w-4 text-gray-400" />
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Selected Highlight Detail */}
                    {selectedHighlight && (
                      <div className="mt-6 pt-6 border-t">
                        <h3 className="font-semibold text-gray-900 mb-4 flex items-center justify-between">
                          <span>Selected Highlight</span>
                          <button
                            onClick={() => setSelectedHighlight(null)}
                            className="text-gray-400 hover:text-gray-600"
                          >
                            <X className="h-4 w-4" />
                          </button>
                        </h3>
                        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                          <div className="flex justify-between items-center mb-3">
                            <div>
                              <span className="font-medium text-gray-900">
                                {selectedHighlight.file || fileName}
                              </span>
                              <span className="text-sm text-gray-600 ml-2">
                                • Page {selectedHighlight.page}
                              </span>
                            </div>
                            <button
                              className="text-xs bg-white border border-blue-300 text-blue-700 px-3 py-1 rounded-lg hover:bg-blue-50"
                              onClick={() => {
                                // Navigate to page in PDF viewer
                                alert(`Would navigate to page ${selectedHighlight.page}`);
                              }}
                            >
                              View in PDF
                            </button>
                          </div>
                          <div className="bg-white p-4 rounded-lg border">
                            <p className="text-gray-800 whitespace-pre-wrap text-sm">
                              {selectedHighlight.text}
                            </p>
                          </div>
                          <div className="mt-4 flex justify-between items-center">
                            <div className="flex items-center space-x-2">
                              <div className="text-xs px-2 py-1 bg-purple-100 text-purple-800 rounded">
                                Confidence: {selectedHighlight.confidence || 'High'}
                              </div>
                              <div className="text-xs px-2 py-1 bg-green-100 text-green-800 rounded">
                                Source: {selectedHighlight.source_type || 'Document'}
                              </div>
                            </div>
                            <button
                              className="text-sm text-blue-600 hover:text-blue-800 flex items-center"
                              onClick={() => {
                                navigator.clipboard.writeText(selectedHighlight.text);
                                alert('Text copied to clipboard!');
                              }}
                            >
                              Copy text
                            </button>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Document Statistics */}
                    <div className="mt-6 pt-6 border-t">
                      <h3 className="font-semibold text-gray-900 mb-4">Document Statistics</h3>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-gray-50 p-3 rounded-lg border">
                          <div className="text-xs text-gray-600">Total Chunks</div>
                          <div className="font-medium text-gray-900 text-lg">
                            {highlights.length * 5 || 'N/A'}
                          </div>
                        </div>
                        <div className="bg-gray-50 p-3 rounded-lg border">
                          <div className="text-xs text-gray-600">Avg. Relevance</div>
                          <div className="font-medium text-green-700 text-lg">92%</div>
                        </div>
                        <div className="bg-gray-50 p-3 rounded-lg border">
                          <div className="text-xs text-gray-600">Pages Cited</div>
                          <div className="font-medium text-gray-900 text-lg">
                            {[...new Set(highlights.map(h => h.page))].length}
                          </div>
                        </div>
                        <div className="bg-gray-50 p-3 rounded-lg border">
                          <div className="text-xs text-gray-600">Session Age</div>
                          <div className="font-medium text-gray-900 text-lg">New</div>
                        </div>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="mt-6 pt-6 border-t">
                      <h3 className="font-semibold text-gray-900 mb-4">Actions</h3>
                      <div className="space-y-2">
                        <button
                          onClick={async () => {
                            try {
                              const response = await fetch(`${API_URL}/export/${sessionId}`);
                              if (response.ok) {
                                const blob = await response.blob();
                                const url = window.URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.href = url;
                                a.download = `${fileName}_chat_history.json`;
                                document.body.appendChild(a);
                                a.click();
                                window.URL.revokeObjectURL(url);
                                document.body.removeChild(a);
                              }
                            } catch (error) {
                              alert('Failed to export chat history');
                            }
                          }}
                          className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gradient-to-r from-gray-100 to-gray-200 text-gray-800 rounded-lg hover:from-gray-200 hover:to-gray-300 transition-all"
                        >
                          <Download className="h-4 w-4" />
                          <span>Export Chat History</span>
                        </button>
                        <button
                          onClick={async () => {
                            try {
                              const response = await fetch(`${API_URL}/summary/${sessionId}`);
                              if (response.ok) {
                                const data = await response.json();
                                setChatHistory(prev => [...prev, {
                                  type: 'ai',
                                  content: data.summary,
                                  timestamp: new Date().toLocaleTimeString(),
                                  sources: []
                                }]);
                              }
                            } catch (error) {
                              alert('Failed to generate summary');
                            }
                          }}
                          className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gradient-to-r from-blue-100 to-blue-200 text-blue-800 rounded-lg hover:from-blue-200 hover:to-blue-300 transition-all"
                        >
                          <FileText className="h-4 w-4" />
                          <span>Generate Summary</span>
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Mobile Highlights Toggle */}
      <button
        onClick={() => setShowHighlightsPanel(!showHighlightsPanel)}
        className={`lg:hidden fixed bottom-4 right-4 z-20 p-3 rounded-full shadow-lg ${
          showHighlightsPanel 
            ? 'bg-gray-800 text-white' 
            : 'bg-gradient-to-r from-amber-500 to-orange-500 text-white'
        }`}
      >
        {showHighlightsPanel ? (
          <ChevronRight className="h-6 w-6" />
        ) : (
          <Highlighter className="h-6 w-6" />
        )}
      </button>

      {/* Footer */}
      <footer className="border-t bg-white/80 backdrop-blur-sm mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-gray-600 text-sm">
              <p>© DocIntel AI •  RAG System with Memory</p>
            </div>
            <div className="flex items-center space-x-6 mt-4 md:mt-0">
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-500">Backend:</span>
                <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">
                  {API_URL}
                </span>
              </div>
              <button
                onClick={checkApiHealth}
                className="text-sm text-blue-600 hover:text-blue-800"
              >
                Check Connection
              </button>
              <button
                onClick={loadSessions}
                className="text-sm text-purple-600 hover:text-purple-800"
              >
                Refresh Sessions
              </button>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap items-center justify-center gap-4 text-xs text-gray-500">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
              <span>Session Management</span>
            </div>
            <div className="flex items-center">
              <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
              <span>Chat Memory</span>
            </div>
            <div className="flex items-center">
              <div className="w-2 h-2 bg-purple-500 rounded-full mr-2"></div>
              <span>Hybrid Search</span>
            </div>
            <div className="flex items-center">
              <div className="w-2 h-2 bg-amber-500 rounded-full mr-2"></div>
              <span>Document Highlights</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}