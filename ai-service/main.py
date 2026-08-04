from __future__ import annotations

import io
import json
import os
import tempfile
from typing import Optional

import whisper
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydub import AudioSegment
import uvicorn

from config import AI_SERVICE_PORT, GEMINI_MODEL_NAME
from rag_service import evaluate_answer, generate_questions, parse_resume_file


OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "mistral")


app = FastAPI(title="AI Interviewer Microservice", version="2.0")

origins = [
    os.getenv("FRONTEND_URL", ""),
    "http://localhost:5173",
    "http://localhost:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in origins if origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WHISPER_MODEL = None


def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("Loading Whisper Model ...")
        WHISPER_MODEL = whisper.load_model("base.en")
        print("Whisper Model Loaded Successfully")
    return WHISPER_MODEL


class QuestionRequest(BaseModel):
    mode: str = "general"
    company: Optional[str] = None
    role: str = "MERN Stack Developer"
    level: str = "Junior"
    difficulty: str = "Medium"
    count: int = 5
    interview_type: str = "coding-mix"
    topic: Optional[str] = None


class QuestionResponse(BaseModel):
    questions: list[dict]
    mode: str
    company: Optional[str] = None
    role: str
    difficulty: str
    level: str
    model_used: str
    retrieved_context: str
    resume_profile: Optional[dict] = None


class EvaluationRequest(BaseModel):
    question: str
    question_type: str
    role: str
    level: str
    user_answer: Optional[str] = None
    user_code: Optional[str] = None
    ideal_answer: Optional[str] = None
    concept_tags: Optional[list[str]] = None


class EvaluationResponse(BaseModel):
    technicalScore: int
    confidenceScore: int
    overallScore: int
    semanticSimilarity: float
    aiFeedback: str
    idealAnswer: str
    conceptsCorrectlyExplained: list[str] = []
    missingConcepts: list[str] = []


@app.get("/")
async def root():
    return {
        "message": "Hello from AI Interviewer Microservice !",
        "default_provider": os.getenv("LLM_PROVIDER", "gemini"),
        "gemini_model": GEMINI_MODEL_NAME,
        "ollama_model": OLLAMA_MODEL_NAME,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate-questions", response_model=QuestionResponse)
async def generate_interview_questions(request: QuestionRequest):
    try:
        payload = generate_questions(
            mode=request.mode,
            company=request.company,
            role=request.role,
            difficulty=request.difficulty,
            level=request.level,
            interview_type=request.interview_type,
            count=request.count,
            topic=request.topic,
        )
        return QuestionResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/generate-resume-questions", response_model=QuestionResponse)
async def generate_resume_questions(
    role: str = Form("Resume Candidate"),
    level: str = Form("Junior"),
    difficulty: str = Form("Medium"),
    count: int = Form(5),
    interview_type: str = Form("coding-mix"),
    resume_file: UploadFile = File(...),
):
    try:
        file_bytes = await resume_file.read()
        resume_text = parse_resume_file(file_bytes, resume_file.filename or "resume.txt")
        payload = generate_questions(
            mode="resume",
            company=None,
            role=role,
            difficulty=difficulty,
            level=level,
            interview_type=interview_type,
            count=count,
            topic=None,
            resume_text=resume_text,
            resume_filename=resume_file.filename,
        )
        return QuestionResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio_in_memory = io.BytesIO(audio_bytes)
        audio_segment = AudioSegment.from_file(audio_in_memory)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            temp_audio_path = tmp.name
            audio_segment.export(temp_audio_path, format="mp3")

        model = get_whisper_model()
        result = model.transcribe(temp_audio_path)

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        return {"transcription": result["text"].strip()}
    except Exception as exc:
        if "temp_audio_path" in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    try:
        payload = evaluate_answer(
            question=request.question,
            question_type=request.question_type,
            role=request.role,
            level=request.level,
            user_answer=request.user_answer or "",
            user_code=request.user_code or "",
            ideal_answer=request.ideal_answer or "",
            concept_tags=request.concept_tags or [],
        )
        return EvaluationResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=AI_SERVICE_PORT)
