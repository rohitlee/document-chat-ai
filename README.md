# 📄 Document AI Chatbot

A multi-lingual, Retrieval-Augmented Generation (RAG) chatbot built with Streamlit. Upload your documents in various languages and ask questions about their content.

![App Screenshot](app_screenshot.png) 
*(**Action:** Take a nice screenshot of your app and save it as `app_screenshot.png` in your project folder)*

---

## ✨ Features

-   **Multi-Lingual Support:** Upload documents and ask questions in English, Hindi, Tamil, and more.
-   **Multiple File Formats:** Supports PDF (`.pdf`), Microsoft Word (`.docx`), and Text (`.txt`) files.
-   **AI-Powered Responses:** Uses state-of-the-art open-source models from Hugging Face for question-answering.
-   **Cloud Translation:** Leverages Sarvam AI for fast and accurate language detection and translation.
-   **Local Vector Storage:** Uses ChromaDB to store document embeddings locally for privacy and speed.
-   **Interactive UI:** A clean and modern user interface built with Streamlit.

## 🛠️ Tech Stack

-   **Framework:** Streamlit
-   **LLM (via API):** Hugging Face Inference API (Mixtral / Llama 3)
-   **Translation:** Sarvam AI SDK
-   **Embeddings:** `sentence-transformers` (multilingual models)
-   **Vector Database:** ChromaDB
-   **Language:** Python

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

-   Python 3.9+
-   An account on [Hugging Face](https://huggingface.co/)
-   An account and API key from [Sarvam AI](https://www.sarvam.ai/)

### 2. Installation

Clone the repository to your local machine:
```bash
git clone https://github.com/rohitlee/document-chat-ai.git
cd document-chat-ai