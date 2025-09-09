import os
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_postgres import PostgresChatMessageHistory
from langchain.memory import ConversationBufferMemory
import psycopg
from langchain.callbacks.base import BaseCallbackHandler
from sqlalchemy import create_engine
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Setup
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Replace with your actual OpenAI API key
DB_URI = "postgresql+psycopg2://user:password@localhost:5432/bookstore"
CHAT_HISTORY_CONN = "postgresql://user:password@localhost:5432/bookstore"
CHAT_HISTORY_TABLE = "chat_history"
psycopg_conn = psycopg.connect(CHAT_HISTORY_CONN)

# Run this only once
# try:
#     PostgresChatMessageHistory.create_tables(CHAT_HISTORY_CONN, CHAT_HISTORY_TABLE)
#     print(f"Chat history table '{CHAT_HISTORY_TABLE}' created or already exists")
# except Exception as e:
#     print(f"Note: {e}")

# Database setup
engine = create_engine(DB_URI)
# Define custom table info for better LLM context
custom_table_info = {
    "authors": (
        "A table of authors.\n"
        "- id (SERIAL PRIMARY KEY): Unique ID of author\n"
        "- name (VARCHAR): Name of the author\n"
        "- birth_year (INTEGER): Year of birth\n"
        "- nationality (VARCHAR): Nationality of the author\n"
    ),
    "books": (
        "A table of books.\n"
        "- id (SERIAL PRIMARY KEY): Unique ID of book\n"
        "- title (VARCHAR): Title of the book\n"
        "- author_id (INTEGER): References authors(id)\n"
        "- genre (VARCHAR): Genre of the book\n"
        "- publication_year (INTEGER): Year of publication\n"
        "- rating (DECIMAL): Book rating (0–10)\n"
    ),
    "books_with_authors": (
        "A view combining books and authors.\n"
        "- book_id (INTEGER): ID of the book\n"
        "- title (VARCHAR): Title of the book\n"
        "- genre (VARCHAR): Genre of the book\n"
        "- publication_year (INTEGER): Year of publication\n"
        "- rating (DECIMAL): Rating of the book\n"
        "- author_name (VARCHAR): Name of the author\n"
        "- birth_year (INTEGER): Birth year of the author\n"
        "- nationality (VARCHAR): Nationality of the author\n"
    ),
}

# Initialize SQLDatabase with view support and custom info
db = SQLDatabase(
    engine=engine,
    include_tables=list(custom_table_info.keys()),
    custom_table_info=custom_table_info,
    view_support=True
)
llm = ChatOpenAI(model="gpt-4", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Callback handler to capture SQL results
class SQLResultHandler(BaseCallbackHandler):
    def __init__(self):
        self.latest_sql_result = None
        self.sql_run_ids = set()
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get('name', 'unknown') if isinstance(serialized, dict) else str(serialized)
        if tool_name == "sql_db_query":
            self.sql_run_ids.add(kwargs.get('run_id'))
    
    def on_tool_end(self, output, **kwargs):
        run_id = kwargs.get('run_id')
        if run_id in self.sql_run_ids:
            self.latest_sql_result = output
            self.sql_run_ids.discard(run_id)
    
    def get_latest_result(self):
        return self.latest_sql_result
async def get_session_history(session_id: str):
    async_conn = await psycopg.AsyncConnection.connect(CHAT_HISTORY_CONN)
    return PostgresChatMessageHistory(CHAT_HISTORY_TABLE, session_id, async_connection=async_conn)
async def get_memory(session_id: str):
    chat_history = await get_session_history(session_id)
    return ConversationBufferMemory(chat_memory=chat_history, memory_key="history", return_messages=True)
async def create_agent_with_memory(session_id: str):
    memory = await get_memory(session_id)
    return create_sql_agent(
        toolkit=toolkit,
        llm=llm,
        agent_type="tool-calling",
        agent_executor_kwargs={"memory": memory},
        verbose=True
    )
# FastAPI app
app = FastAPI(title="SQL Chat Agent")

class ChatRequest(BaseModel):
    message: str
    user_id: str
class ChatResponse(BaseModel):
    reply: str
    raw_sql_result: Optional[str] = None

# Endpoint for chat
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    sql_handler = SQLResultHandler()
    agent = await create_agent_with_memory(request.user_id)
    
    response = await agent.ainvoke(
        {"input": request.message},
        {"callbacks": [sql_handler]}
    )
    
    return ChatResponse(
        reply=response["output"],
        raw_sql_result=sql_handler.get_latest_result()
    )

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)