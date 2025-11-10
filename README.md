# Placement Interview Preparation Assistant

## Overview

This project is a Retrieval-Augmented Generation (RAG) based system that generates company and role specific technical interview questions. The system retrieves context from a local knowledge base and uses a language model to generate structured interview question sets with short answers.

## Features

* Generates interview questions tailored to a specific company and role.
* Retrieval-Augmented Generation pipeline to ensure responses stay grounded in stored context.
* Local inference (no external API dependency).
* Simple Gradio-based user interface.

## System Flow

1. User enters a query containing the company and/or role.
2. System performs semantic search in a vector database (ChromaDB).
3. Top matching context chunks are retrieved.
4. Language model generates interview questions based on retrieved context.
5. Output is displayed in the UI.

## Tech Stack

| Component           | Tool                                        |
| ------------------- | ------------------------------------------- |
| Embeddings          | SentenceTransformer                         |
| Vector Store        | ChromaDB                                    |
| Retrieval Framework | LangChain                                   |
| Language Model      | Phi Mini (local)                            |
| UI                  | Gradio                                      |
| Environment         | Python, GPU backend (Vast.ai or local CUDA) |

## Setup Instructions

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Ensure Models Are Available Locally

```
models/
 ├── phi_mini/                  # LLM directory
 └── embeddings/                # SentenceTransformer embedding model
```

### 3. Start the Application

```
python gradio_ui.py
```

### 4. Access the UI

If running locally:

```
http://localhost:7860
```

If running on cloud (e.g., Vast.ai), use the mapped public port:

```
http://<instance-ip>:<public-port>
```

## Directory Structure

```
project/
 ├── models/
 │   ├── phi_mini/
 │   └── embeddings/
 ├── chroma/                       # Vector DB storage
 ├── gradio_ui.py                  # Gradio interface
 ├── rag_pipeline.py               # Retrieval and generation logic
 ├── README.md
 └── requirements.txt
```

## Future Enhancements

* Add aptitude and HR question modules.
* Add role-wise dataset expansion.
* Improve UI layout and presentation styling.
* Optional web deployment and user authentication.

