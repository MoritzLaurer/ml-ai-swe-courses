# Module B4: Working with Data in PostgreSQL for RAG (with psycopg3)

# This script demonstrates how to populate the RAG database tables (created in Module B3)
# with sample data and how to query this data using psycopg3.
# It covers INSERT operations, using RETURNING, querying with JOINs,
# working with JSONB and ARRAY types, and simulating vector similarity searches.

import os
import psycopg # psycopg3
from psycopg.rows import dict_row # For fetching rows as dictionaries
from dotenv import load_dotenv
import json # For working with JSONB data
import random # For generating dummy vector data
import datetime # For working with timestamps

# --- Helper function for DB Connection ---
def get_db_connection_string():
    """Constructs the database connection string from environment variables."""
    load_dotenv()
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "myprojectdb")
    db_user = os.getenv("DB_USER", "myprojectuser")
    db_password = os.getenv("DB_PASSWORD", "yoursecurepassword")

    if not all([db_host, db_port, db_name, db_user, db_password]):
        raise ValueError("One or more database connection environment variables are not set. Please check your .env file.")
    return f"host='{db_host}' port='{db_port}' dbname='{db_name}' user='{db_user}' password='{db_password}'"

def populate_and_query_rag_data(conn_string):
    """
    Demonstrates populating and querying data in the RAG system's tables.
    """
    VECTOR_DIMENSION = 1536 # Must match the dimension defined in Module B3

    try:
        # Connect using dict_row factory for convenient dictionary-like row access
        with psycopg.connect(conn_string, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                print("--- Populating RAG Database ---")

                # 1. Insert a Document
                print("\n1. Inserting a document...")
                doc_sql = """
                INSERT INTO documents (source_name, metadata, original_content)
                VALUES (%s, %s, %s) RETURNING doc_id;
                """
                doc_metadata = {"author": "AI Course", "year": 2025, "url": "http://example.com/intro_to_db"}
                # Using json.dumps for JSONB fields
                cur.execute(doc_sql, ("Introduction to Databases", json.dumps(doc_metadata), "Databases are essential... (full content here) ..."))
                doc_id = cur.fetchone()['doc_id']
                print(f"Inserted document with doc_id: {doc_id}")

                # 2. Insert Chunks for the Document
                print("\n2. Inserting chunks for the document...")
                chunks_data = [
                    {"text": "A database is an organized collection of structured information.", "tokens": 9, "meta": {"page": 1}},
                    {"text": "Relational databases use SQL and have ACID properties.", "tokens": 9, "meta": {"page": 1}},
                    {"text": "NoSQL databases offer flexibility and scalability.", "tokens": 7, "meta": {"page": 2}},
                ]
                chunk_ids = []
                chunk_sql = """
                INSERT INTO chunks (doc_id, chunk_text, token_count, chunk_metadata)
                VALUES (%s, %s, %s, %s) RETURNING chunk_id;
                """
                for chunk_item in chunks_data:
                    cur.execute(chunk_sql, (doc_id, chunk_item["text"], chunk_item["tokens"], json.dumps(chunk_item["meta"])))
                    chunk_id = cur.fetchone()['chunk_id']
                    chunk_ids.append(chunk_id)
                    print(f"Inserted chunk with chunk_id: {chunk_id} for doc_id: {doc_id}")

                # 3. Insert Embeddings for the Chunks
                print("\n3. Inserting embeddings for the chunks...")
                # Ensure pgvector is installed and the vector extension is created in your DB.
                # `psycopg` can send a list/tuple of numbers for a `vector` type.
                embedding_sql = """
                INSERT INTO embeddings (chunk_id, embedding_vector, model_name)
                VALUES (%s, %s, %s) RETURNING embedding_id;
                """
                for i, chunk_id in enumerate(chunk_ids):
                    # Generate a dummy vector (list of floats)
                    dummy_vector = [random.uniform(-1, 1) for _ in range(VECTOR_DIMENSION)]
                    cur.execute(embedding_sql, (chunk_id, dummy_vector, "text-embedding-ada-002-dummy"))
                    embedding_id = cur.fetchone()['embedding_id']
                    print(f"Inserted embedding with embedding_id: {embedding_id} for chunk_id: {chunk_id}")

                # 4. Insert a User
                print("\n4. Inserting a user...")
                user_sql = """
                INSERT INTO users (username, user_profile)
                VALUES (%s, %s) RETURNING user_id;
                """
                user_profile_data = {"preferences": {"theme": "dark"}, "last_login": str(datetime.datetime.now(datetime.timezone.utc))}
                cur.execute(user_sql, ("test_user_1", json.dumps(user_profile_data)))
                user_id = cur.fetchone()['user_id']
                print(f"Inserted user with user_id: {user_id}")

                # 5. Insert a Chat Session
                print("\n5. Inserting a chat session...")
                session_sql = """
                INSERT INTO chat_sessions (user_id, session_metadata)
                VALUES (%s, %s) RETURNING session_id;
                """
                session_metadata = {"topic": "Database Basics", "model_used": "gpt-4-dummy"}
                cur.execute(session_sql, (user_id, json.dumps(session_metadata)))
                session_id = cur.fetchone()['session_id']
                print(f"Inserted chat session with session_id: {session_id}")

                # 6. Insert Chat Messages
                print("\n6. Inserting chat messages...")
                # Note: retrieved_chunk_ids is an INTEGER ARRAY
                messages_data = [
                    {"sender": "user", "text": "What is a database?", "retrieved_chunks": None},
                    {"sender": "bot", "text": "A database is an organized collection of structured information. We found this in chunk " + str(chunk_ids[0]), "retrieved_chunks": [chunk_ids[0]]},
                    {"sender": "user", "text": "Tell me about NoSQL.", "retrieved_chunks": None},
                    {"sender": "bot", "text": "NoSQL databases offer flexibility and scalability. This was found in chunk " + str(chunk_ids[2]), "retrieved_chunks": [chunk_ids[2]]}
                ]
                message_sql = """
                INSERT INTO chat_messages (session_id, sender, message_text, retrieved_chunk_ids)
                VALUES (%s, %s, %s, %s);
                """ # Not using RETURNING id here for simplicity
                for msg in messages_data:
                    cur.execute(message_sql, (session_id, msg["sender"], msg["text"], msg["retrieved_chunks"]))
                print(f"Inserted {len(messages_data)} messages for session_id: {session_id}")

                conn.commit() # Commit all insertions
                print("\n--- Data Population Complete ---")

                # --- Querying RAG Data ---
                print("\n--- Querying RAG Data ---")

                # 7. Querying Documents with JSONB metadata
                print("\n7. Querying documents by JSONB metadata (author):")
                # Note: The '->>' operator gets a JSON object field as text.
                # The '@>' operator checks if JSONB A contains JSONB B.
                cur.execute("SELECT doc_id, source_name, metadata->>'author' AS author FROM documents WHERE metadata @> %s;",
                            (json.dumps({"author": "AI Course"}),))
                for doc in cur.fetchall():
                    print(f"  Doc ID: {doc['doc_id']}, Source: {doc['source_name']}, Author: {doc['author']}")

                # 8. Querying Chunks with JOIN to Documents
                print("\n8. Querying chunks and their document source:")
                # Using f-string for table/column names, but parameters for values
                # `dict_row` makes accessing columns by name easy
                query_chunks_join = """
                SELECT c.chunk_id, c.chunk_text, d.source_name, c.chunk_metadata->>'page' AS page
                FROM chunks c
                JOIN documents d ON c.doc_id = d.doc_id
                WHERE d.doc_id = %s;
                """
                cur.execute(query_chunks_join, (doc_id,))
                for row in cur.fetchall():
                    print(f"  Chunk ID: {row['chunk_id']}, Text: '{row['chunk_text'][:30]}...', Source: {row['source_name']}, Page: {row['page']}")

                # 9. Querying Chat Messages with ARRAY operations
                print("\n9. Querying chat messages that retrieved a specific chunk:")
                # The ANY operator can be used with arrays.
                if chunk_ids:
                    target_chunk_id = chunk_ids[0]
                    cur.execute("SELECT message_id, sender, message_text FROM chat_messages WHERE %s = ANY(retrieved_chunk_ids);",
                                (target_chunk_id,))
                    print(f"  Messages referencing chunk_id {target_chunk_id}:")
                    for msg in cur.fetchall():
                        print(f"    Msg ID: {msg['message_id']}, Sender: {msg['sender']}, Text: '{msg['message_text'][:30]}...'")

                # 10. Simulating Vector Similarity Search (L2 distance)
                print("\n10. Simulating vector similarity search (finding chunks similar to a query vector):")
                # This requires the `pgvector` extension and an index for performance on large datasets.
                # We'll generate a new dummy query vector.
                # The `<->` operator calculates L2 distance (Euclidean distance).
                # Other operators: `<#>` for negative inner product, `<=>` for cosine distance.
                query_vector = [random.uniform(-1, 1) for _ in range(VECTOR_DIMENSION)]
                similarity_sql = """
                SELECT e.chunk_id, c.chunk_text, e.embedding_vector <-> CAST(%s AS vector) AS distance
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.chunk_id
                ORDER BY distance ASC
                LIMIT 3;
                """
                # `psycopg` can pass the list `query_vector` directly to `%s` for a vector type.
                # The CAST(%s AS vector) ensures PostgreSQL treats the parameter correctly.
                cur.execute(similarity_sql, (query_vector,))
                print(f"  Top 3 chunks by L2 distance to a dummy query vector:")
                for row in cur.fetchall():
                    print(f"    Chunk ID: {row['chunk_id']}, Text: '{row['chunk_text'][:30]}...', Distance: {row['distance']:.4f}")

                # Using psycopg's server-side parameter binding (default and recommended)
                # protects against SQL injection. All `%s` placeholders are handled safely.

                print("\n--- Data Querying Examples Complete ---")

    except psycopg.OperationalError as e:
        print(f"Database connection failed: {e}")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except psycopg.Error as e:
        print(f"A database error occurred: {e}\nSQL: {e.diag.sqlstate if hasattr(e, 'diag') else 'N/A'}\nMessage: {e.diag.message_primary if hasattr(e, 'diag') else str(e)}")
        if conn and not conn.closed:
            conn.rollback() # Rollback any pending transaction
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'conn' in locals() and conn and not conn.closed:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    print("Starting Module B4: Working with Data in PostgreSQL for RAG (with psycopg3)")
    db_conn_string = ""
    try:
        db_conn_string = get_db_connection_string()
        populate_and_query_rag_data(db_conn_string)
    except ValueError as e: # Catch error from get_db_connection_string
        print(e)
    except Exception as e: # Catch any other unexpected error during setup
        print(f"Failed to run module B4: {e}")

    print("\nModule B4 execution finished.")
    print("Review the script 'B4_rag_data_ops_postgres_psycopg3.py' for details.")
    print("Sample data has been added to your RAG tables.")
    print("To clear this sample data for a fresh run, you might need to TRUNCATE or DELETE FROM these tables, or re-run Module B3 to recreate the schema.")
