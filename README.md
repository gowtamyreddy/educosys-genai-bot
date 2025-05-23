
# 📘 Educosys RAG Notes Chatbot

A local, privacy-first AI assistant that uses LLaMA 3 and LangChain to answer questions from handwritten and typed notes via semantic search.

---

## ✨ Features

- 🔍 Ask natural language questions from your personal notes
- 📂 Supports PDF, DOCX, and TXT formats (typed and handwritten)
- 🧠 Powered by FAISS + MiniLM for fast semantic retrieval
- 🤖 Uses LLaMA 3 via Groq API for contextual answers
- ⚡ Built with Streamlit for a lightweight, interactive UI
- 🔒 Runs entirely on your machine (except LLM inference)

---

## 🧠 System Overview


```mermaid
flowchart TD
    A[User Question] --> B[Vector Search]
    B --> C[Relevant Chunks]
    C --> D[Prompt Construction]
    D --> E[LLAMA 3 via Groq]
    E --> F[Answer in Streamlit]
```

## 📸 Output Preview

Here’s what the app looks like in action:

![Educosys LLM Screenshot](https://github.com/gowtamyreddy/educosys-genai-bot/blob/8077be50e0fb2593f65801b12c22399c389ed0f3/screenshot.png?raw=true)



