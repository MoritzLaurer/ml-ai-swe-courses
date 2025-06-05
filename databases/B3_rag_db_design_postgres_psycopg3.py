# Module B3: Designing the RAG System's Relational Database (PostgreSQL with psycopg3)

# This script focuses on designing and creating the specific PostgreSQL database schema
# for our RAG (Retrieval Augmented Generation) chatbot. It uses `psycopg3` to execute
# DDL statements to create tables for documents, chunks, embeddings (including enabling
# the `pgvector` extension), users, chat sessions, and messages, as outlined in the curriculum.
# This builds upon the PostgreSQL and `psycopg3` basics from Module B2.

import os
import psycopg # psycopg3
from dotenv import load_dotenv

# --- RAG System Data Entities ---
# For our RAG chatbot, we need to store several types of information:
# 1.  Documents: The source material for the RAG system.
# 2.  Chunks: Processed segments of the documents, suitable for embedding.
# 3.  Embeddings: Vector representations of the chunks for similarity search.
# 4.  Users: Information about users interacting with the chatbot (optional but good practice).
# 5.  Chat Sessions: To group related messages in a conversation.
# 6.  Chat Messages: Individual messages exchanged between users and the bot,
#     including links to retrieved chunks.

# --- Database Schema Design (PostgreSQL DDL) ---
# We'll define the tables, their columns, data types, primary keys (PK),
# foreign keys (FK), and some basic constraints.

# Note on pgvector:
# The `embeddings` table will use the `vector` data type from the `pgvector`
# extension. This extension needs to be enabled in your PostgreSQL database.
# The script will attempt to create it if it doesn't exist.
# The size of the vector (e.g., `vector(1536)`) depends on the embedding model
# you plan to use (e.g., OpenAI's text-embedding-ada-002 produces 1536-dimensional vectors).
# We'll use a common dimension like 1536 as an example.


# --- Helper function for DB Connection ---
def get_db_connection(conn_string):
    """Establishes a database connection."""
    return psycopg.connect(conn_string)


def get_db_connection_string():
    """Constructs the database connection string from environment variables."""
    load_dotenv()
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "myprojectdb")
    db_user = os.getenv("DB_USER", "myprojectuser")
    db_password = os.getenv("DB_PASSWORD", "yoursecurepassword")

    if not all([db_host, db_port, db_name, db_user, db_password]):
        print("Warning: One or more database connection environment variables are not fully set.")
        print("Please check your .env file. Using default values where applicable.")

    return f"host='{db_host}' port='{db_port}' dbname='{db_name}' user='{db_user}' password='{db_password}'"


def create_rag_schema(conn):
    """
    Creates the database schema for the RAG system.
    This includes enabling pgvector and creating necessary tables.
    """
    with conn.cursor() as cur:
        print("--- Setting up RAG Database Schema ---")

        # 1. Enable pgvector extension
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit() # DDL like CREATE EXTENSION might need explicit commit outside a transaction block
                          # or ensure autocommit is on for this. psycopg3's behavior can vary.
                          # For safety, we commit here.
            print("Ensured 'vector' extension is enabled.")
        except psycopg.Error as e:
            print(f"Error enabling 'vector' extension: {e}")
            conn.rollback() # Rollback if extension creation fails
            # Consider whether to raise the error or allow proceeding if tables don't use vectors
            # For this module, it's crucial for the embeddings table.
            raise

        # Table creation order matters due to Foreign Key constraints.
        # Parent tables must exist before child tables referencing them.

        # Drop tables if they exist (for idempotency during development)
        # Order of dropping is reverse of creation due to FK constraints
        tables_to_drop = [
            "chat_messages",
            "embeddings",
            "chat_sessions",
            "chunks",
            "users",
            "documents"
        ]
        for table_name in tables_to_drop:
            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;") # CASCADE drops dependent objects
            print(f"Dropped table '{table_name}' if it existed.")
        conn.commit() # Commit after dropping tables

        # 2. Create 'documents' table
        # Stores information about the source documents.
        ddl_documents = """
        CREATE TABLE documents (
            doc_id SERIAL PRIMARY KEY,
            source_name VARCHAR(255) NOT NULL,
            metadata JSONB,                            -- Flexible field for source-specific info (e.g., URL, author)
            original_content TEXT,                     -- Can be null if content is chunked immediately and not stored whole
            upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(ddl_documents)
        print("Created 'documents' table.")

        # 3. Create 'chunks' table
        # Stores processed text chunks from the documents.
        ddl_chunks = """
        CREATE TABLE chunks (
            chunk_id SERIAL PRIMARY KEY,
            doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE, -- FK to documents
            chunk_text TEXT NOT NULL,
            token_count INTEGER,                       -- Optional: for tracking chunk size
            chunk_metadata JSONB                       -- E.g., page number, section, paragraph ID
        );
        """
        cur.execute(ddl_chunks)
        print("Created 'chunks' table.")
        # Add an index on doc_id for faster lookups of chunks by document
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);")
        print("Created index on chunks(doc_id).")


        # 4. Create 'embeddings' table
        # Stores vector embeddings for each chunk.
        # VECTOR_DIMENSION should match your embedding model's output.
        VECTOR_DIMENSION = 1536 # Example: OpenAI text-embedding-ada-002
        ddl_embeddings = f"""
        CREATE TABLE embeddings (
            embedding_id SERIAL PRIMARY KEY,
            chunk_id INTEGER NOT NULL UNIQUE REFERENCES chunks(chunk_id) ON DELETE CASCADE, -- FK to chunks, unique as one embedding per chunk
            embedding_vector VECTOR({VECTOR_DIMENSION}) NOT NULL,
            model_name VARCHAR(100)                    -- E.g., 'text-embedding-ada-002'
        );
        """
        cur.execute(ddl_embeddings)
        print(f"Created 'embeddings' table with vector dimension {VECTOR_DIMENSION}.")
        # Indexing for vector similarity search (e.g., HNSW or IVFFlat) is crucial for performance.
        # This is usually done after data is loaded. Example for HNSW (pgvector recommendation):
        # CREATE INDEX ON embeddings USING hnsw (embedding_vector vector_l2_ops);
        # We will cover creating vector indexes in a later module or when data is populated.
        # For now, a GIN index might be a placeholder if you want to experiment with some operators,
        # but specific vector indexes are better. Let's add a placeholder for a GiST index.
        # Or simply create an index on chunk_id for now for standard lookups.
        cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);")
        print("Created index on embeddings(chunk_id). (Note: Specialized vector indexes like HNSW/IVFFlat are needed for fast similarity search on embedding_vector).")


        # 5. Create 'users' table (Optional, but common for chatbots)
        # Stores user information.
        ddl_users = """
        CREATE TABLE users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,     -- Or use email, or an external ID
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            user_profile JSONB                         -- For additional user preferences/info
        );
        """
        cur.execute(ddl_users)
        print("Created 'users' table.")

        # 6. Create 'chat_sessions' table
        # Groups messages into conversations.
        ddl_chat_sessions = """
        CREATE TABLE chat_sessions (
            session_id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL, -- Session can exist even if user is deleted, or use ON DELETE CASCADE
            start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            session_metadata JSONB                     -- E.g., conversation topic, initial prompt type
        );
        """
        cur.execute(ddl_chat_sessions)
        print("Created 'chat_sessions' table.")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);")
        print("Created index on chat_sessions(user_id).")

        # 7. Create 'chat_messages' table
        # Stores individual messages in a session.
        ddl_chat_messages = """
        CREATE TABLE chat_messages (
            message_id SERIAL PRIMARY KEY,
            session_id INTEGER NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            sender VARCHAR(50) NOT NULL CHECK (sender IN ('user', 'bot', 'system')), -- 'system' for context/instructions
            message_text TEXT NOT NULL,
            retrieved_chunk_ids INTEGER[],             -- Array of chunk_ids used for RAG response (FKs not directly enforceable on array elements)
            llm_response_metadata JSONB                -- E.g., model used, latency, token counts for the response
        );
        """
        cur.execute(ddl_chat_messages)
        print("Created 'chat_messages' table.")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id_timestamp ON chat_messages(session_id, timestamp);")
        print("Created index on chat_messages(session_id, timestamp).")
        # An index on retrieved_chunk_ids might be useful if you search by them, GIN index for array:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_retrieved_chunks ON chat_messages USING GIN (retrieved_chunk_ids);")
        print("Created GIN index on chat_messages(retrieved_chunk_ids).")

        conn.commit() # Commit all DDL changes
        print("\nDatabase schema for RAG system created successfully.")

def main():
    """Main function to set up the RAG database schema."""
    conn_string = get_db_connection_string()
    try:
        with get_db_connection(conn_string) as conn:
            create_rag_schema(conn)
    except psycopg.OperationalError as e:
        print(f"Database connection failed: {e}")
        print("Please check your PostgreSQL server and .env file.")
    except psycopg.Error as e: # Catch other psycopg errors
        print(f"A database error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    print("Starting Module B3: Designing the RAG System's Relational Database (PostgreSQL with psycopg3)")
    main()
    print("\nModule B3 execution finished.")
    print("Review the script 'B3_rag_db_design_postgres_psycopg3.py' for schema details.")
    print("The RAG system tables should now exist in your database.")
