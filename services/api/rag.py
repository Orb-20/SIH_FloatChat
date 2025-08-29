# services/api/rag.py
import os
import chromadb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from sqlalchemy import create_engine, text
from chromadb.config import Settings
import google.generativeai as genai

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=GOOGLE_API_KEY)

# Database connection details from environment variables
DB_USER = os.getenv("POSTGRES_USER", "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME = os.getenv("POSTGRES_DB", "argo_db")
DB_HOST = os.getenv("POSTGRES_HOST", "db")  # prefer env, default to Docker service name
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# ChromaDB connection details (try multiple env var names for robustness)
CHROMA_HOST = (
    os.getenv("CHROMA_SERVER_HOST")
    or os.getenv("CHROMA_HOST")
    or os.getenv("CHROMA_SERVER")
    or os.getenv("CHROMA_SERVER_URL")
    or "chroma"
)
CHROMA_PORT = (
    os.getenv("CHROMA_SERVER_HTTP_PORT")
    or os.getenv("CHROMA_PORT")
    or os.getenv("CHROMA_SERVER_PORT")
    or "8000"
)

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "argo_float_metadata")

# --- Database and Vector Store Connections ---

# Connect to PostgreSQL
try:
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    print(f"Successfully connected to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME}.")
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")
    engine = None

# Connect to ChromaDB
# Connect to ChromaDB (robust, avoids settings vs args mismatch)
retriever = None
try:
    # Resolve env vars robustly
    CHROMA_HOST = (
        os.getenv("CHROMA_SERVER_HOST")
        or os.getenv("CHROMA_HOST")
        or os.getenv("CHROMA_SERVER")
        or "chroma"
    )
    CHROMA_PORT = (
        os.getenv("CHROMA_SERVER_HTTP_PORT")
        or os.getenv("CHROMA_PORT")
        or "8000"
    )

    print(f"Attempting to connect to Chroma at {CHROMA_HOST}:{CHROMA_PORT} ...")

    client_settings = Settings(
        chroma_api_impl="rest",
        chroma_server_host=str(CHROMA_HOST),
        chroma_server_http_port=str(CHROMA_PORT),
    )

    # IMPORTANT: pass host and port args **and** the settings to avoid the mismatch error
    chroma_client = chromadb.HttpClient(
        host=str(CHROMA_HOST),
        port=int(CHROMA_PORT),
        settings=client_settings,
    )

    # quick heartbeat check (will raise if unreachable)
    chroma_client.heartbeat()
    print(f"Chroma heartbeat OK ({CHROMA_HOST}:{CHROMA_PORT}).")

    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = Chroma(
        client=chroma_client,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print(f"Successfully connected to ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    retriever = None


# --- LLM and Prompt Engineering ---
llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-pro"), temperature=0)

prompt_template = """
You are an expert AI assistant that converts natural language questions about ARGO oceanographic data into SQL queries.
You must only respond with a SQL query and nothing else. Do not wrap the query in markdown or any other formatting.

Database Schema:
The database contains two tables: `argo_floats` and `argo_profiles`.

1. `argo_floats` table:
   - `float_id` (INTEGER, PRIMARY KEY)
   - `wmo` (INTEGER)
   - `project_name` (TEXT)
   - `launch_date` (TIMESTAMP)
   - `end_date` (TIMESTAMP)

2. `argo_profiles` table:
   - `profile_id` (INTEGER, PRIMARY KEY)
   - `float_id` (INTEGER, FOREIGN KEY)
   - `profile_date` (TIMESTAMP)
   - `latitude` (FLOAT)
   - `longitude` (FLOAT)
   - `temp` (FLOAT[])
   - `psal` (FLOAT[])
   - `pres` (FLOAT[])

Context from relevant documents:
{context}

User Question:
{question}

Based on the schema and context, generate the SQL query to answer the user's question.
SQL Query:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
sql_generation_chain = LLMChain(llm=llm, prompt=PROMPT)

# --- FastAPI Application ---
app = FastAPI(title="FloatChat API", description="An API for converting natural language to ARGO data queries.")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    natural_language_response: str
    sql_query: str
    data: list | None = None
    error: str | None = None

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not available.")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Vector store (Chroma) not available.")

    try:
        # 1. Retrieve context from Chroma
        retrieved_docs = retriever.get_relevant_documents(request.question)
        context = "\n".join([getattr(d, "page_content", str(d)) for d in retrieved_docs])

        # 2. Generate SQL query
        sql_query = sql_generation_chain.run({"context": context, "question": request.question}).strip()

        # Basic validation to prevent harmful queries
        if any(keyword in sql_query.upper() for keyword in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]):
            raise ValueError("Generated SQL query contains a restricted keyword.")

        # 3. Execute SQL query
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            rows = result.fetchall()
            cols = result.keys()
            df = pd.DataFrame(rows, columns=cols)
            data = df.to_dict(orient="records")

        # 4. Generate natural language response
        response_prompt = f"""
        Based on the user's question: "{request.question}"
        And the following data retrieved from the database:
        {df.to_string() if not df.empty else 'No rows returned.'}

        Provide a concise, natural language answer.
        If the data is empty, state that no results were found.
        If there are results, summarize them clearly.
        """
        response_llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-pro"), temperature=0.2)
        natural_language_response = response_llm.invoke(response_prompt).content

        return QueryResponse(
            natural_language_response=natural_language_response,
            sql_query=sql_query,
            data=data,
        )

    except Exception as e:
        print(f"An error occurred in /query: {e}")
        return QueryResponse(
            natural_language_response="I'm sorry — I encountered an error trying to answer your question.",
            sql_query="No SQL query generated due to an error.",
            error=str(e),
        )

@app.get("/")
def read_root():
    return {"message": "Welcome to the FloatChat API"}
