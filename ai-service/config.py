"""
Centralized configuration for the AI Interviewer service.
All environment variables, model names, and paths are managed here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ===============================
# Server
# ===============================
AI_SERVICE_PORT = int(os.getenv("AI_SERVICE_PORT", 8000))

# ===============================
# Gemini LLM
# ===============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

# ===============================
# Embedding Model
# ===============================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/text-embedding-004")

# ===============================
# ChromaDB Vector Store
# ===============================
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(os.path.dirname(__file__), "chroma_db"))

# Collection names
COMPANY_COLLECTION = "company_interviews"
TECHNICAL_COLLECTION = "technical_knowledge"

# ===============================
# Knowledge Base Paths
# ===============================
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
COMPANY_KB_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "companies")
TECHNICAL_KB_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "technical")

# ===============================
# RAG Settings
# ===============================
RAG_TOP_K = 8              # Number of documents to retrieve
RAG_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score

# ===============================
# Resume Upload
# ===============================
MAX_RESUME_SIZE_MB = 5
ALLOWED_RESUME_EXTENSIONS = {".pdf", ".docx"}
