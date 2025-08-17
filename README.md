# YouTube Transcript RAG System

* Developed a YouTube Retrieval-Augmented Generation (RAG) system using LangChain, FAISS, and Hugging
Face models to enable context-aware Q&A over video transcripts.
* Integrated Mistral-7B (4-bit quantized) and Gemma-3 1B for efficient inference, optimizing GPU/CPU usage with
bitsandbytes and caching strategies.
* Built an interactive Streamlit interface for transcript indexing, retrieval, and real-time question answering.
* Containerized the application with Docker for reproducible deployment on local and cloud environments.


This project implements a Retrieval-Augmented Generation (RAG) pipeline that allows users to query YouTube video transcripts in real time.  
The app fetches the transcript, chunks and embeds it using FAISS, and then uses an open-source LLM (Mistral/Gemma) to provide context-aware answers.  
The interface is built with Streamlit and packaged with Docker for easy deployment.

## Demo



[![Watch the demo](https://img.youtube.com/vi/hmtuvNfytjM/0.jpg)](https://www.youtube.com/watch?v=hmtuvNfytjM)




![App Demo](<assets/Screenshot 2025-08-17 131354.png>)