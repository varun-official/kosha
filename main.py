from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import psycopg2
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from PyPDF2 import PdfReader
from docx import Document
import httpx
import time
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# Initialize Mistral client
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
llm = ChatMistralAI(api_key=MISTRAL_API_KEY)

# Database connection function
def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_CONNECTION_STRING"))

# Ensure pgvector extension is enabled and update schema
def setup_database():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS document_embeddings (
        id SERIAL PRIMARY KEY,
        doc_name TEXT,
        page INT,
        chunk_index INT,  -- New field to track chunk order
        text TEXT,
        embedding VECTOR(1024)  -- Ensure this matches MistralAIEmbeddings
    );

    """)
    conn.commit()
    cur.close()
    conn.close()

setup_database()

def split_text_into_chunks(text, chunk_size=400, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Initialize MistralAI Embeddings
embedding_model = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    extracted_text = []

    # Extract text from PDF
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file.file)
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                extracted_text.append((page_num, page_text))
    
    # Extract text from DOCX
    elif file.filename.endswith(".docx"):
        doc = Document(file.file)
        page_num = 1  # Assuming a single logical page for DOCX files
        for para in doc.paragraphs:
            extracted_text.append((page_num, para.text))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a PDF or DOCX.")

    conn = get_db_connection()
    cur = conn.cursor()

    # Process extracted text with chunking
    for page_num, page_text in extracted_text:
        cleaned_text = page_text.replace("\n", " ").strip()
        chunks = split_text_into_chunks(cleaned_text)

        for i, chunk in enumerate(chunks):
            embedding = embedding_model.embed_query(chunk)
            cur.execute(
                "INSERT INTO document_embeddings (doc_name, page, chunk_index, text, embedding) VALUES (%s, %s, %s, %s, %s)",
                (file.filename, page_num, i, chunk, np.array(embedding).tolist()),
            )

    conn.commit()
    cur.close()
    conn.close()

    return {"message": "File processed successfully", "filename": file.filename}

def handle_rate_limit(chain, query, context, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return chain.invoke({"query": query, "context": context})
        except httpx.HTTPStatusError as e:
            if "429" in str(e):  # Check if it's a rate limit error
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    return "Rate limit exceeded. Please try again later."
            else:
                raise e  # Reraise non-rate limit errors

@app.post("/query")
async def query_embeddings(request: QueryRequest):
    query_embedding = embedding_model.embed_query(request.query)
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT doc_name, page, text, 1 - (embedding <=> %s::vector) AS similarity
        FROM document_embeddings
        ORDER BY similarity DESC
        LIMIT %s
        """,
        (np.array(query_embedding).tolist(), request.top_k),
    )
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    # Construct context for MistralAI using text content and document details
    context = "\n".join([f"Document: {r[0]}, Page: {r[1]}\nContent: {r[2]}" for r in results])

    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant that answers queries strictly based on the given documents.  
    - **Only use** the provided content to generate responses.  
    - **Do not** use external knowledge or speculate.  
    - If the provided text does **not** contain relevant information, respond **only** with:  
      'I do not have any information about the query.'  
    - **Only if an answer is found**, conclude with: (Source: doc_name, Page: page) 
    - Double check the answer before sending 

    Query: {query}  

    Below is the relevant content extracted from documents:  
    {context}"""),
])
    
    chain = prompt | llm
    response = handle_rate_limit(chain, request.query, context)
    
    return {
        "query": request.query,
        "response": response.content
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
