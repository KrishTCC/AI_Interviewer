# 🤖 AI Mock Interview Platform

> An AI-powered mock interview platform that simulates real technical interviews using **Retrieval-Augmented Generation (RAG)**, **LangChain**, **Gemini**, **Whisper**, **FastAPI**, and the **MERN Stack**.

![React](https://img.shields.io/badge/Frontend-React-61DAFB?style=for-the-badge\&logo=react)
![Node.js](https://img.shields.io/badge/Backend-Node.js-339933?style=for-the-badge\&logo=node.js)
![Express](https://img.shields.io/badge/Express.js-000000?style=for-the-badge\&logo=express)
![MongoDB](https://img.shields.io/badge/Database-MongoDB-47A248?style=for-the-badge\&logo=mongodb)
![FastAPI](https://img.shields.io/badge/AI-FastAPI-009688?style=for-the-badge\&logo=fastapi)
![LangChain](https://img.shields.io/badge/RAG-LangChain-blue?style=for-the-badge)
![Gemini](https://img.shields.io/badge/LLM-Google_Gemini-4285F4?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

# 📖 Overview

Preparing for technical interviews often means solving generic questions that do not reflect real interview experiences.

This platform solves that problem by using **Retrieval-Augmented Generation (RAG)** to retrieve company-specific interview experiences and combines them with **Google Gemini** to generate realistic interview questions.

Candidates can:

* 🎯 Attempt AI-generated mock interviews
* 🎤 Answer using voice or text
* 📝 Get speech converted into text using Whisper
* 📊 Receive AI-based feedback and scoring
* 📚 Practice interviews tailored to companies and roles

---

# ✨ Features

## 👤 User Features

* Secure Authentication
* Google OAuth Login
* Create Unlimited Mock Interviews
* Text-Based Answers
* Voice-Based Answers
* AI Evaluation & Feedback
* Interview History
* Responsive UI

---

## 🧠 AI Features

* Retrieval-Augmented Generation (RAG)
* LangChain Pipeline
* ChromaDB Vector Database
* Google Gemini LLM
* Whisper Speech-to-Text
* Company-specific Interview Questions
* Context-aware Question Generation
* AI Feedback & Scoring

---

# 🏗️ System Architecture

```text
                +---------------------+
                |      React UI       |
                +----------+----------+
                           |
                    REST APIs
                           |
                +----------v----------+
                | Node.js + Express   |
                | Authentication      |
                | Sessions            |
                | Database            |
                +----------+----------+
                           |
                    HTTP Requests
                           |
                +----------v----------+
                |    FastAPI AI       |
                |---------------------|
                | LangChain           |
                | Gemini              |
                | ChromaDB            |
                | Whisper             |
                +----------+----------+
                           |
                   Vector Retrieval
                           |
                Company Interview Data
```

---

# 🧩 Tech Stack

## Frontend

* React.js
* Tailwind CSS
* React Router
* Axios
* Socket.io Client

---

## Backend

* Node.js
* Express.js
* JWT Authentication
* Google OAuth
* Socket.io
* MongoDB
* Mongoose

---

## AI Service

* FastAPI
* LangChain
* Google Gemini
* ChromaDB
* Whisper
* Sentence Transformers

---

## Database

* MongoDB
* ChromaDB (Vector Database)

---

# 📂 Project Structure

```text
AI-MOCK-INTERVIEW/
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── context/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── utils/
│   │   └── App.jsx
│   └── package.json
│
├── backend/
│   ├── config/
│   ├── controllers/
│   ├── middleware/
│   ├── models/
│   ├── routes/
│   ├── sockets/
│   ├── utils/
│   ├── server.js
│   └── package.json
│
├── ai-service/
│   ├── rag/
│   ├── vectorstore/
│   ├── embeddings/
│   ├── prompts/
│   ├── services/
│   ├── api/
│   ├── main.py
│   └── requirements.txt
│
├── screenshots/
├── README.md
└── .env.example
```

> **Note:** Folder names inside the AI service may vary depending on your implementation. Update them if your repository uses different names.

---

# 🔄 Application Flow

```text
User
   │
   ▼
Login / Register
   │
   ▼
Select Company & Role
   │
   ▼
Node Backend
   │
   ▼
FastAPI AI Service
   │
   ▼
Retrieve Context using RAG
   │
   ▼
Gemini Generates Questions
   │
   ▼
User Answers (Voice/Text)
   │
   ▼
Whisper Converts Speech → Text
   │
   ▼
Gemini Evaluates Response
   │
   ▼
Feedback Stored in MongoDB
```

---

# 🧠 How RAG Works

```text
Company Interview Dataset
            │
            ▼
Text Chunking
            │
            ▼
Embeddings
            │
            ▼
ChromaDB
            │
            ▼
Relevant Context Retrieval
            │
            ▼
Gemini Prompt
            │
            ▼
Interview Questions
```

---

# 🚀 Getting Started

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/AI-Mock-Interview.git

cd AI-Mock-Interview
```

---

## 2. Install Dependencies

### Frontend

```bash
cd frontend
npm install
```

### Backend

```bash
cd backend
npm install
```

### AI Service

```bash
cd ai-service
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

## Backend

```env
PORT=

MONGO_URI=

JWT_SECRET=

GOOGLE_CLIENT_ID=

GOOGLE_CLIENT_SECRET=

AI_SERVICE_URL=
```

---

## AI Service

```env
GEMINI_API_KEY=

CHROMA_DB_PATH=

EMBEDDING_MODEL=
```

---

# ▶️ Run the Project

## Backend

```bash
npm run dev
```

---

## Frontend

```bash
npm run dev
```

---

## AI Service

```bash
uvicorn main:app --reload
```

---

# 📸 Screenshots

Add screenshots of your application here.

```text
screenshots/

├── Home.png

├── Login.png

├── Dashboard.png

├── Interview.png

└── Feedback.png
```

Example:

```markdown
## Home

![Home](screenshots/Home.png)
```

---

# 🎯 Future Improvements

* Resume Parsing
* Adaptive Difficulty
* Coding Interviews
* Video Interview Support
* AI Follow-up Questions
* Performance Analytics Dashboard
* Personalized Learning Roadmap
* Multi-language Interviews
* Company-wise Interview Analytics

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push your branch

```bash
git push origin feature/new-feature
```

5. Open a Pull Request
