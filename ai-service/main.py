import uvicorn
import os
import io
import json
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import ollama
import whisper
from pydub import AudioSegment

load_dotenv()

AI_SERVICE_PORT = int(os.getenv("AI_SERVICE_PORT", 8000))
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "mistral")

app = FastAPI(
    title="AI Interviewer Microservice",
    version="1.0"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================
# Lazy Load Whisper Model
# ===============================

WHISPER_MODEL = None


def get_whisper_model():
    global WHISPER_MODEL

    if WHISPER_MODEL is None:
        print("Loading Whisper Model ...")
        WHISPER_MODEL = whisper.load_model("base.en")
        print("Whisper Model Loaded Successfully")

    return WHISPER_MODEL


# ===============================
# Models
# ===============================

class QuestionResquest(BaseModel):
    role: str = "MERN Stack Developer"
    level: str = "Junior"
    count: int = 5
    interview_type: str = "coding-mix"


class QuestionResponse(BaseModel):
    questions: list[str]
    model_used: str


class EvaluationRequest(BaseModel):
    question: str
    question_type: str
    role: str
    level: str
    user_answer: Optional[str] = None
    user_code: Optional[str] = None


class EvaluationResponse(BaseModel):
    technicalScore: int
    confidenceScore: int
    aiFeedback: str
    idealAnswer: str


@app.get("/")
async def root():
    return {
        "message": "Hello from AI Interviewer Microservice !",
        "model": OLLAMA_MODEL_NAME
    }


# ===============================
# Generate Questions
# ===============================

@app.post("/generate-questions", response_model=QuestionResponse)
async def generate_questions(request: QuestionResquest):

    try:
        if request.interview_type == "coding-mix":

            coding_count = int(request.count * 0.2)
            oral_count = request.count - coding_count

            instruction = (
                f"The first {coding_count} questions MUST be coding challenge requiring function implementation. "
                f"The remaining {oral_count} questions MUST be conceptual oral questions."
            )

        else:
            instruction = (
                "All questions MUST be conceptual oral questions. "
                "Do Not generate coding challenges."
            )

        system_prompt = (
            "You are a professional technical interviewer. "
            "Task: Generate interview questions. "
            "No conversational text or numbering. "
            f"Crucial: {instruction} "
            "Output exactly one question per line."
        )

        user_prompt = (
            f"Generate exactly {request.count} unique interview questions "
            f"for a {request.level} level {request.role}"
        )

        response = ollama.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=user_prompt,
            system=system_prompt,
            options={"temperature": 0.6}
        )

        raw_text = response["response"].strip()

        questions = [
            q.strip()
            for q in raw_text.split("\n")
            if q.strip()
        ]

        return QuestionResponse(
            questions=questions[:request.count],
            model_used=OLLAMA_MODEL_NAME
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ===============================
# Transcribe Audio
# ===============================

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):

    try:
        audio_bytes = await file.read()

        audio_in_memory = io.BytesIO(audio_bytes)

        audio_segment = AudioSegment.from_file(
            audio_in_memory
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        ) as tmp:

            temp_audio_path = tmp.name

            audio_segment.export(
                temp_audio_path,
                format="mp3"
            )


        # Load whisper only when needed
        model = get_whisper_model()

        result = model.transcribe(
            temp_audio_path
        )

        os.remove(temp_audio_path)

        return {
            "transcription": result["text"].strip()
        }


    except Exception as e:

        if (
            "temp_audio_path" in locals()
            and os.path.exists(temp_audio_path)
        ):
            os.remove(temp_audio_path)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ===============================
# Evaluate Answer
# ===============================

@app.post(
    "/evaluate",
    response_model=EvaluationResponse
)
async def evaluate(request: EvaluationRequest):

    try:

        if request.question_type == "oral":

            assessment_instruction = (
                "This is a conceptual oral question. "
                "Focus on verbal explanation. "
                "If answer is irrelevant, score 0."
            )

        else:

            assessment_instruction = (
                "This is a coding challenge. "
                "Evaluate logic and efficiency. "
                "If code is empty or random, score 0."
            )


        system_prompt = (
            "You are a strict technical interviewer. "
            "Return only JSON. "
            "Required keys: technicalScore, confidenceScore, aiFeedback, idealAnswer. "
            f"Context: {assessment_instruction}"
        )


        user_prompt = (
            f"Role: {request.role}\n"
            f"Question: {request.question}\n"
            f"Level: {request.level}\n"
            f"Answer: {request.user_answer}\n"
            f"Code: {request.user_code}"
        )


        response = ollama.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=user_prompt,
            system=system_prompt,
            format="json",
            options={
                "temperature": 0.1
            }
        )


        response_text = response["response"].strip()


        try:

            evaluation_data = json.loads(
                response_text
            )

            return EvaluationResponse(
                **evaluation_data
            )


        except:

            return EvaluationResponse(
                technicalScore=0,
                confidenceScore=0,
                aiFeedback="Failed to parse response",
                idealAnswer="Failed to parse response"
            )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=AI_SERVICE_PORT
    )