# Curriculum: Mastering Databases for LLM Applications (2025 Edition)

**Overall Goal:** Understand different database types, master PostgreSQL implementation and deployment (using `psycopg3` and `SQLAlchemy`), and learn best practices for production systems, all through the lens of building a RAG LLM chatbot backend. We will use `uv` for environment and package management.

---

## Chapter A: Foundations & Setup
*   **Goal:** Get the environment ready and understand database fundamentals.
*   Modules:
    *   `A0_setup.sh`
    *   `A1_intro_to_databases.md`

---

**Module A0: Setting Up Your Development Environment (with `uv`)**
*   `A0_setup.sh` (This script will contain instructions as comments, as actual setup is command-line based)
*   **Content:**
    *   Introduction to `uv` as a fast Python package installer and resolver.
    *   Installing `uv` (referencing official installation methods for 2025).
    *   Creating a project directory.
    *   Initializing a virtual environment with `uv`: `uv venv`.
    *   Activating the virtual environment.
    *   Installing core packages:
        *   `uv pip install psycopg[binary] python-dotenv sqlalchemy alembic fastapi uvicorn[standard] pymongo redis` (Group initial installs for core tools used across modules)
        *   Brief explanation of `python-dotenv` for managing connection strings.
    *   Setting up PostgreSQL:
        *   Options: Local installation (latest stable version), Docker (recommended for ease of use and version consistency).
        *   Basic `docker-compose.yml` example for PostgreSQL if Docker is chosen.
        *   Creating a `.env` file to store database connection details (e.g., `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`).
    *   Verifying the setup: Simple Python script to connect to PostgreSQL using `psycopg3` and print the server version.

---

**Module A1: Foundations - Understanding Databases**
*   `A1_intro_to_databases.md`
*   **Content:**
    *   What is a database? Why do we need them? Core benefits (Persistence, Concurrency, Integrity, Scalability).
    *   Overview of Database Categories:
        *   Relational Databases (SQL): Structure, ACID properties.
        *   NoSQL Databases:
            *   Document Stores (e.g., MongoDB)
            *   Key-Value Stores (e.g., Redis)
            *   Graph Databases (e.g., Neo4j)
            *   Column-Family Stores (e.g., Apache Cassandra)
            *   Vector Databases (e.g., Pinecone, Weaviate, Milvus - or `pgvector` as a PostgreSQL extension)
    *   Pros and Cons of each major category.
    *   Common use cases for each type.
    *   Introduction to the RAG LLM Chatbot System:
        *   High-level architecture.
        *   Where do databases fit? (Storing documents, embeddings, chat history, user data).
        *   Initial thoughts on what kind of data our RAG system will need.

---

## Chapter B: Relational Databases & SQL Deep Dive
*   **Goal:** Master SQL fundamentals using SQLite, then transition to PostgreSQL for the RAG project, and finally learn SQLAlchemy.
*   Modules:
    *   `B1_sql_fundamentals_sqlite.py`
    *   `B2_postgresql_intro_psycopg3.py`
    *   `B3_rag_db_design_postgres_psycopg3.py`
    *   `B4_rag_data_ops_postgres_psycopg3.py`
    *   `B5_sqlalchemy_orm_for_rag.py`

---

**Module B1: SQL Language Fundamentals with SQLite**
*   `B1_sql_fundamentals_sqlite.py`
*   **Content:**
    *   Introduction to SQLite: Serverless, file-based, Python's `sqlite3` module.
    *   SQL Command Categories (DDL, DML, DQL, TCL).
    *   Common SQL Data Types in SQLite (`INTEGER`, `REAL`, `TEXT`, `BLOB`).
    *   `CREATE TABLE` (`INTEGER PRIMARY KEY AUTOINCREMENT`), `ALTER TABLE`, `DROP TABLE`, `CREATE INDEX`. `PRAGMA foreign_keys = ON;`.
    *   `INSERT INTO` (placeholders, `cursor.lastrowid`), `UPDATE`, `DELETE FROM`.
    *   In-depth `SELECT`: `FROM`, `WHERE` (operators, `BETWEEN`, `IN`, `LIKE`, `IS NULL`), `DISTINCT`, `ORDER BY`, `LIMIT`, `AS`.
    *   Joining Tables: `INNER JOIN`, `LEFT JOIN`.
    *   Aggregate Functions: `COUNT()`, `SUM()`, `AVG()`, `MIN()`, `MAX()`.
    *   `GROUP BY` and `HAVING`. Simple Subqueries.
    *   Transaction Control with `sqlite3`: `connection.commit()`, `connection.rollback()`, `with connection:` context manager.
    *   Hands-on Python examples using `sqlite3` for generic SQL learning.

---

**Module B2: PostgreSQL Fundamentals & `psycopg3`**
*   `B2_postgresql_intro_psycopg3.py`
*   **Content:**
    *   Why PostgreSQL? (Reliability, extensibility, SQL compliance, JSONB, `pgvector`).
    *   Core Relational Concepts specific to PostgreSQL: Schemas, Tables, Rows, Columns, Data Types (e.g., `SERIAL`, `TEXT`, `VARCHAR`, `TIMESTAMP WITH TIME ZONE`, `NUMERIC`, `BOOLEAN`, `JSONB`, `ARRAY`).
    *   Keys in PostgreSQL: Primary Keys, Foreign Keys, Unique Constraints.
    *   Connecting to PostgreSQL with `psycopg3`: Connection strings from `.env`, `psycopg.connect()`, `Connection` and `Cursor` objects.
    *   Executing simple DDL (e.g., `CREATE SCHEMA IF NOT EXISTS`) and DML statements (`cursor.execute()`).
    *   Fetching results (`fetchone()`, `fetchall()`, `fetchmany()`).
    *   Parameter binding with `psycopg3` (server-side binding, `%s` placeholders).
    *   Basic error handling (`psycopg3.Error`). Using `with` statements for connection/cursor.

---

**Module B3: Designing the RAG System's Database (PostgreSQL with `psycopg3`)**
*   `B3_rag_db_design_postgres_psycopg3.py`
*   **Content:**
    *   Identifying data entities for our RAG chatbot.
    *   Designing the PostgreSQL database schema (DDL):
        *   `documents` table: `doc_id (SERIAL PK)`, `source_name (VARCHAR)`, `metadata (JSONB)`, `original_content (TEXT)`, `upload_timestamp (TIMESTAMPTZ)`.
        *   `chunks` table: `chunk_id (SERIAL PK)`, `doc_id (FK)`, `chunk_text (TEXT)`, `token_count (INT)`, `chunk_metadata (JSONB)`.
        *   `embeddings` table: `embedding_id (SERIAL PK)`, `chunk_id (FK, UNIQUE)`, `embedding_vector (vector type from pgvector)`, `model_name (VARCHAR)`. (Mention enabling `pgvector` extension: `CREATE EXTENSION IF NOT EXISTS vector;`).
        *   `users` table: `user_id (SERIAL PK)`, `username (VARCHAR, UNIQUE)`, `created_at (TIMESTAMPTZ)`.
        *   `chat_sessions` table: `session_id (SERIAL PK)`, `user_id (FK)`, `start_time (TIMESTAMPTZ)`, `session_metadata (JSONB)`.
        *   `chat_messages` table: `message_id (SERIAL PK)`, `session_id (FK)`, `timestamp (TIMESTAMPTZ)`, `sender (VARCHAR)`, `message_text (TEXT)`, `retrieved_chunk_ids (INTEGER[])`.
    *   Relationships, Normalization, Data Types in PostgreSQL context.
    *   Indexing strategies for RAG (e.g., `CREATE INDEX`, GIN for `JSONB`, basic `pgvector` index creation like HNSW/IVFFlat - `USING hnsw (embedding_vector vector_l2_ops)`).
    *   Python (`psycopg3`) code to create these tables in PostgreSQL.

---

**Module B4: Data Operations for RAG in PostgreSQL (with `psycopg3`)**
*   `B4_rag_data_ops_postgres_psycopg3.py`
*   **Content:**
    *   Populating the RAG database using `psycopg3`: `INSERT` statements, `RETURNING` clause for IDs.
    *   Querying data: `SELECT`, `JOIN`, filtering, sorting, aggregation.
    *   Working with `JSONB` (operators like `->>`, `->`, `@>`) and `ARRAY` data types (operators like `ANY`).
    *   Simulating vector similarity search with `pgvector` operators (e.g., `<->`, `<=>`, `<#>`).
    *   Storing user queries and LLM responses into `chat_messages`.
    *   Using `psycopg3`'s row factories (e.g., `dict_row`).
    *   Re-emphasize parameterization to prevent SQL injection.

---

**Module B5: ORMs: `SQLAlchemy` with PostgreSQL for RAG**
*   `B5_sqlalchemy_orm_for_rag.py`
*   **Content:**
    *   Introduction to Object-Relational Mappers (ORMs) and `SQLAlchemy`.
    *   `SQLAlchemy 2.x` style: Core vs. ORM.
    *   Engine, Session, Declarative Base (`from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column`).
    *   Defining database models (Python classes) for the RAG system's tables (from B3) using SQLAlchemy 2.x syntax. Map PostgreSQL types to SQLAlchemy types.
    *   Creating the schema using `Base.metadata.create_all(engine)`.
    *   Basic CRUD operations (Create, Read, Update, Delete) using the SQLAlchemy ORM (Session API: `session.add()`, `session.get()`, `session.query().filter().first()`, `session.commit()`).
    *   Querying with SQLAlchemy ORM: selecting specific columns, filtering, ordering, joining.
    *   Comparison: `psycopg3` direct SQL vs. `SQLAlchemy` ORM for RAG operations. Advantages of ORM (abstraction, less SQL boilerplate, Pythonic).

---

## Chapter C: Exploring NoSQL Options for RAG
*   **Goal:** Understand how NoSQL databases can complement a PostgreSQL-based RAG system.
*   Modules:
    *   `C0_intro_nosql_for_rag.md`
    *   `C1_mongodb_for_rag.py`
    *   `C2_redis_for_rag.py`
    *   `C3_vector_databases_for_rag.py`

---

**Module C0: Introduction to NoSQL for Complementing RAG Systems**
*   `C0_intro_nosql_for_rag.md`
*   **Content:**
    *   Recap: Why consider NoSQL alongside PostgreSQL? (Scalability, schema flexibility, specialized use cases).
    *   Brief intro to Document Databases (MongoDB), Key-Value Stores (Redis), Standalone Vector Databases.
    *   General trade-offs. Pointers to sub-modules.

---

**Module C1: Deep Dive - Document Databases (MongoDB) for RAG**
*   `C1_mongodb_for_rag.py`
*   **Content:**
    *   Intro to MongoDB: Documents, Collections, BSON. `pymongo` driver.
    *   Connecting. CRUD ops. Indexing. Schema design/evolution.
    *   RAG use cases: Detailed chat logs, variable source document metadata.
    *   Comparison: JSON vs. BSON vs. JSONB. `pymongo` examples.

---

**Module C2: Deep Dive - Key-Value Stores (Redis) for RAG**
*   `C2_redis_for_rag.py`
*   **Content:**
    *   Intro to Redis: In-memory, data structures. `redis-py` driver.
    *   Connecting. Common data types/commands (Strings, Hashes, Lists, Sets, Sorted Sets).
    *   RAG use cases: Caching LLM responses, frequently accessed data, session management, rate limiting.
    *   Persistence overview. `redis-py` examples.

---

**Module C3: Deep Dive - Standalone Vector Databases for RAG**
*   `C3_vector_databases_for_rag.py`
*   **Content:**
    *   When to consider dedicated Vector DBs over `pgvector`.
    *   Overview of popular options (Pinecone, Weaviate, Milvus, Qdrant).
    *   Core concepts: Embeddings, Indexes (HNSW, IVF), Similarity Metrics, Metadata Filtering.
    *   Architectural considerations. Conceptual API interaction.
    *   Pros/Cons vs. `pgvector`. (More conceptual, potentially `chromadb` for local demo).

---

## Chapter D: Productionizing PostgreSQL and API Integration
*   **Goal:** Learn about advanced PostgreSQL usage, database administration, performance, and API integration, primarily using SQLAlchemy.
*   Modules:
    *   `D1_advanced_postgres_transactions.py`
    *   `D2_production_databases_admin_migrations.py`
    *   `D3_db_integration_fastapi.py`

---

**Module D1: Advanced PostgreSQL Features & Transactions with SQLAlchemy**
*   `D1_advanced_postgres_transactions.py`
*   **Content:**
    *   **In-depth Transactions with `SQLAlchemy`:**
        *   ACID properties recap (briefly).
        *   SQLAlchemy Session transaction lifecycle: `session.begin()`, `session.commit()`, `session.rollback()`, `session.close()`.
        *   Using `with Session(engine) as session:` and `with session.begin():` (recommended pattern for context management and automatic commit/rollback).
        *   Nested transactions (savepoints) with SQLAlchemy.
        *   Handling errors within transactions and ensuring rollback.
    *   **Interacting with Advanced PostgreSQL Features via `SQLAlchemy` (Brief Overview & Python examples):**
        *   Views: Querying views as if they were tables using SQLAlchemy models or `table()` constructs.
        *   Materialized Views: Refreshing and querying.
        *   Stored Procedures & Functions: Executing using `session.execute(text(...))` with parameters.
        *   Triggers (conceptual).
        *   Full-Text Search in PostgreSQL: Using `func` for PostgreSQL specific functions like `to_tsvector`, `to_tsquery` and custom operators or `text()` for complex FTS queries.

---

**Module D2: Productionizing Databases: Admin, Performance & Migrations with SQLAlchemy**
*   `D2_production_databases_admin_migrations.py`
*   **Content:**
    *   **Database Administration (Conceptual, PostgreSQL-general):** Users, roles, permissions in PostgreSQL. Backup/recovery (`pg_dump`, `pg_restore`).
    *   **Performance and Scalability (Focus on SQLAlchemy interaction):**
        *   Connection Pooling with SQLAlchemy: `Engine` configuration (`pool_size`, `max_overflow`, etc.).
        *   Monitoring: Using `EXPLAIN` and `EXPLAIN ANALYZE` with SQLAlchemy queries (e.g., by compiling query and prepending `EXPLAIN`). Querying `pg_stat_statements` via SQLAlchemy `text()` execution.
        *   Indexing review and best practices (SQLAlchemy can declare indexes on models).
        *   Read Replicas, Sharding (conceptual, how SQLAlchemy might target different engines).
    *   **Security (Focus on SQLAlchemy):** SQL Injection prevention by using the ORM. Secure connections (SSL configuration in `create_engine`). Principle of least privilege (DB user for app).
    *   **Database Migrations with Alembic (for SQLAlchemy - review from B5 context):**
        *   Recap: Setting up Alembic with SQLAlchemy models.
        *   Generating and applying migration scripts. Autogenerate features for schema changes.
        *   Branching, merging, and managing complex migration histories.

---

**Module D3: Integrating Databases with APIs (FastAPI & SQLAlchemy)**
*   `D3_db_integration_fastapi.py`
*   **Content:**
    *   Role of middleware (FastAPI).
    *   Designing API endpoints for the RAG chatbot (e.g., document upload, query, get chat history).
    *   Python (FastAPI) example using SQLAlchemy 2.x for database interaction:
        *   Dependency Injection for SQLAlchemy Sessions in FastAPI (e.g., `Depends` with a session provider).
        *   Structuring backend code: Routers, Pydantic models for request/response, service layers, repository/DAO patterns using SQLAlchemy Session and models.
    *   Asynchronous database operations with FastAPI and SQLAlchemy (using `AsyncSession` and an async driver like `asyncpg`).
    *   Frontend Interaction (Conceptual).
    *   Authentication and Authorization considerations at the API layer (briefly).

---

## Chapter E: Capstone Project & Future Directions
*   **Goal:** Apply learnings in a mini-project and explore future trends.
*   Modules:
    *   `E1_rag_backend_service_project.py`
    *   `E2_advanced_db_topics_future.py`

---

**Module E1: Mini-Project - Building a Simplified RAG Backend Service**
*   `E1_rag_backend_service_project.py`
*   **Content:**
    *   Combine learnings: FastAPI, PostgreSQL with SQLAlchemy 2.x.
    *   Core functionalities:
        *   API endpoint for document upload (simplified chunking/embedding - perhaps mock embedding generation or use a simple sentence transformer).
        *   API endpoint for Q&A: Takes a query, performs similarity search (basic `pgvector` via SQLAlchemy using `text()` or a compatible extension), constructs prompt, (optionally calls a mock LLM), stores interaction.
        *   API endpoint to retrieve chat history for a session.
    *   Focus on robust database interactions (SQLAlchemy session management, error handling), API structure, Pydantic models.

---

**Module E2: Advanced Database Topics and Future Learning (2025 Perspective)**
*   `E2_advanced_db_topics_future.py`
*   **Content (Primarily conceptual, with pointers for further exploration):**
    *   Graph Databases for RAG (latest trends, e.g., knowledge graph augmentation).
    *   Time-Series Databases (latest use cases for LLM ops, monitoring).
    *   Data Warehousing vs. OLTP. Data Lakes, Lakehouses (current state).
    *   The evolving database landscape: NewSQL, serverless databases, multi-modal databases (latest innovations).
    *   Continuous learning resources.

---

