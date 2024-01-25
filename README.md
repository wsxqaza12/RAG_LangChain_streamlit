# Streamlit Showcase: Unleashing the Power of RAG and LangChain

![Demo](data/demo.gif)

## Overview

The Retrieval Augmented Engine (RAG) is a powerful tool for document retrieval, summarization, and interactive question-answering. This project utilizes LangChain, Streamlit, and Pinecone to provide a seamless web application for users to perform these tasks. With RAG, you can easily upload multiple PDF documents, generate vector embeddings for text within these documents, and perform conversational interactions with the documents. The chat history is also remembered for a more interactive experience.


## Features

- **Streamlit Web App**: The project is built using Streamlit, providing an intuitive and interactive web interface for users.
- **Input Fields**: Users can input essential credentials like LLM URL or OpenAI API key through dedicated input fields.
- **Document Uploader**: Users can upload multiple PDF files, which are then processed for further analysis.
- **Document Splitting**: The uploaded PDFs are split into smaller text chunks, ensuring compatibility with models with token limits.
- **Vector Embeddings**: The text chunks are converted into vector embeddings, making it easier to perform retrieval and question-answering tasks.
- **Flexible Vector Storage**: You can choose to store vector embeddings in a local vector store (Chroma).
- **Interactive Conversations**: Users can engage in interactive conversations with the documents, asking questions and receiving answers. The chat history is preserved for reference.


## Prerequisites

Before running the project, make sure you have the following prerequisites:

- Python 3.7+
- LangChain
- Streamlit
- An OpenAI API key
- PDF documents to upload

## Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/wsxqaza12/RAG_LangChain_streamlit.git
   cd RAG_LangChain_streamlit
   ```

2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run rag_engine.py
   ```

4. Access the app by opening a web browser and navigating to the provided URL.

5. Input your LLM URL or OpenAI API key, 

6. Upload the PDF documents you want to analyze.

7. Click the "Submit Documents" button to process the documents and generate vector embeddings.

8. Engage in interactive conversations with the documents by typing your questions in the chat input box.
