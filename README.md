# 🧠 AI Interviewer

Because getting humbled by an algorithm is excellent preparation for getting humbled by a real interviewer.

AI Interviewer is a full-stack platform that simulates technical interviews using AI-generated questions, voice responses, coding challenges, and performance analytics. It helps developers sharpen their interview skills before facing actual technical interviews.

---

## 🚀 Features

### Custom Interview Generation

- Role-based interview creation
- Multiple difficulty levels
- Conceptual and coding-focused interview modes

### Voice-Based Responses

- Browser-based audio recording
- Speech-to-text transcription using Whisper
- AI evaluation of communication and technical understanding

### Coding Challenges

- Integrated Monaco Editor
- Real-time coding environment
- AI-powered code review and feedback

### Performance Analytics

- Technical and confidence scores
- Session history tracking
- Detailed question-wise feedback
- Performance visualizations using Chart.js

### Secure Authentication

- JWT-based authentication
- Password hashing with bcrypt
- Protected user sessions

---

## 🏗️ Architecture

The project follows a microservice-inspired architecture:

### Frontend (React)

- User Interface
- Audio Recording
- Coding Environment
- Analytics Dashboard

### Backend (Node.js)

- Authentication
- Database Management
- Session Handling
- API Gateway

### AI Service (Python)

- Question Generation
- Speech Transcription
- Response Evaluation
- Feedback Generation

### Ollama (Mistral)

Provides dynamic interview questions and AI-powered evaluation.

---

## 🛠️ Tech Stack

### Frontend

- React (Vite)
- Redux Toolkit
- Tailwind CSS
- Monaco Editor
- Chart.js
- React Router

### Backend

- Node.js
- Express.js
- MongoDB
- Mongoose
- JWT
- bcryptjs

### AI Service

- Python
- FastAPI
- Ollama
- Mistral
- OpenAI Whisper
- PyDub
- FFmpeg

---

## ⚡ Getting Started

### Prerequisites

- Node.js (v16+)
- Python (v3.9+)
- MongoDB
- Ollama
- FFmpeg

Install the model:

```bash
ollama pull mistral
