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
import google.generativeai as genai

# --- Configuration ---
# IMPORTANT: Set your Google API key in your environment variables.
# For Docker, this is set in the docker-compose.yml file.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

genai.configure(api_key=GOOGLE_API_KEY)

# Database connection details from environment variables
DB_USER = os.getenv("POSTGRES_USER", "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME = os.getenv("POSTGRES_DB", "argo_db")
DB_HOST = "db"  # Service name in docker-compose
DB_PORT = "5432"

# ChromaDB connection details
CHROMA_HOST = "chroma" # Service name in docker-compose
CHROMA_PORT = "8000"
CHROMA_COLLECTION_NAME = "argo_float_metadata"

# --- Database and Vector Store Connections ---

# Connect to PostgreSQL
try:
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    print("Successfully connected to PostgreSQL.")
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")
    engine = None

# Connect to ChromaDB
try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

# Define the prompt template for converting questions to SQL
prompt_template = """
You are an expert AI assistant that converts natural language questions about ARGO oceanographic data into SQL queries.
You must only respond with a SQL query and nothing else. Do not wrap the query in markdown or any other formatting.

Database Schema:
The database contains two tables: `argo_floats` and `argo_profiles`.

1. `argo_floats` table:
   - `float_id` (INTEGER, PRIMARY KEY): Unique identifier for each float.
   - `wmo` (INTEGER): World Meteorological Organization number for the float.
   - `project_name` (TEXT): Name of the project.
   - `launch_date` (TIMESTAMP): Date the float was launched.
   - `end_date` (TIMESTAMP): Date of the last transmission.

2. `argo_profiles` table:
   - `profile_id` (INTEGER, PRIMARY KEY): Unique identifier for each profile.
   - `float_id` (INTEGER, FOREIGN KEY): Links to the `argo_floats` table.
   - `profile_date` (TIMESTAMP): Date of the profile measurement.
   - `latitude` (FLOAT): Latitude of the profile.
   - `longitude` (FLOAT): Longitude of the profile.
   - `temp` (FLOAT[]): Array of temperature readings.
   - `psal` (FLOAT[]): Array of salinity readings.
   - `pres` (FLOAT[]): Array of pressure readings (depth).

Context from relevant documents:
{context}

User Question:
{question}

Based on the schema and context, generate the SQL query to answer the user's question.
SQL Query:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create the LangChain
sql_generation_chain = LLMChain(llm=llm, prompt=PROMPT)


# --- FastAPI Application ---

app = FastAPI(
    title="FloatChat API",
    description="An API for converting natural language to ARGO data queries.",
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    natural_language_response: str
    sql_query: str
    data: list | None = None
    error: str | None = None


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Handles a natural language query from the user.
    1. Retrieves context from ChromaDB.
    2. Generates an SQL query using an LLM.
    3. Executes the SQL query against PostgreSQL.
    4. Generates a natural language response.
    5. Returns the complete response.
    """
    if not engine or not retriever:
        raise HTTPException(status_code=503, detail="Database or Vector Store not available.")

    try:
        # 1. Retrieve context
        retrieved_docs = retriever.invoke(request.question)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # 2. Generate SQL query
        sql_query = sql_generation_chain.run({"context": context, "question": request.question}).strip()

        # Basic validation to prevent harmful queries
        if any(keyword in sql_query.upper() for keyword in ["DROP", "DELETE", "UPDATE", "INSERT"]):
            raise ValueError("Generated SQL query contains a restricted keyword.")

        # 3. Execute SQL query
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            data = df.to_dict(orient='records')

        # 4. Generate natural language response
        response_prompt = f"""
        Based on the user's question: "{request.question}"
        And the following data retrieved from the database:
        {df.to_string()}

        Provide a concise, natural language answer.
        If the data is empty, state that no results were found.
        If there are results, summarize them clearly.
        """
        response_llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
        natural_language_response = response_llm.invoke(response_prompt).content

        return QueryResponse(
            natural_language_response=natural_language_response,
            sql_query=sql_query,
            data=data
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        return QueryResponse(
            natural_language_response="I'm sorry, I encountered an error trying to answer your question.",
            sql_query="No SQL query generated due to an error.",
            error=str(e)
        )

@app.get("/")
def read_root():
    return {"message": "Welcome to the FloatChat API"}
