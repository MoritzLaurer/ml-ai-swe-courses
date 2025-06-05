# Module D1: Advanced PostgreSQL Features & Transactions with SQLAlchemy

# This script delves into advanced transaction management with SQLAlchemy
# and demonstrates how to interact with various PostgreSQL-specific features
# like views, materialized views, stored procedures, triggers (conceptually),
# and full-text search using SQLAlchemy.
# It builds upon the models and SQLAlchemy setup from Module B5.

# --- Module Overview and Importance of Topics ---
# This module is structured into two main parts:
# 1.  In-depth Transactions with SQLAlchemy:
#     -   Importance: Understanding transactions (ACID properties, commit, rollback, savepoints)
#         is crucial for maintaining data integrity and consistency, especially in
#         complex applications where multiple database operations must succeed or fail
#         as a single atomic unit. SQLAlchemy provides robust mechanisms for managing
#         transactions effectively.
#
# 2.  Interacting with Advanced PostgreSQL Features via SQLAlchemy:
#     -   Views:
#         -   Importance: Views simplify complex queries by providing a virtual table
#             based on the result-set of a stored query. They can enhance security by
#             restricting access to underlying table data and improve code readability.
#     -   Materialized Views:
#         -   Importance: Materialized views store the result-set of a query physically,
#             which can significantly speed up access to pre-computed, complex, or
#             infrequently changing data, especially for reporting or analytical workloads.
#     -   Stored Procedures & Functions:
#         -   Importance: These allow encapsulating database logic on the server-side.
#             They can improve performance by reducing network traffic (multiple SQL
#             statements executed as one call), enhance security, and promote code reuse.
#     -   Triggers:
#         -   Importance: Triggers are database operations automatically performed in
#             response to certain events (e.g., INSERT, UPDATE, DELETE) on a specified table.
#             They are useful for maintaining data integrity, logging changes (auditing),
#             or automating complex business rules directly within the database.
#     -   Full-Text Search (FTS):
#         -   Importance: For applications dealing with large amounts of text (like our RAG
#             system's document chunks), PostgreSQL's built-in FTS capabilities allow
#             for efficient and sophisticated searching within text documents, including
#             support for stemming, ranking, and various query operators. SQLAlchemy
#             allows leveraging these features.
#
# The script will demonstrate these concepts using the RAG system's database schema
# as a practical example, showing how SQLAlchemy facilitates these advanced interactions.
# ---

import os
import datetime
import random
import traceback

from dotenv import load_dotenv

from sqlalchemy import create_engine, select, func, text, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, TSVECTOR
from sqlalchemy import String, Text, ForeignKey, Integer, TIMESTAMP, Index, UniqueConstraint

# Import Vector type from pgvector-sqlalchemy
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    print("pgvector.sqlalchemy not found. Please ensure 'pgvector' package is installed: uv pip install sqlalchemy pgvector")
    class Vector: # type: ignore
        def __init__(self, dimensions): self.dimensions = dimensions
        def __call__(self, *args, **kwargs): return None

# --- Database Connection URL ---
def get_db_url():
    """Constructs the PostgreSQL connection URL from environment variables."""
    load_dotenv()
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "myprojectdb")
    db_user = os.getenv("DB_USER", "myprojectuser")
    db_password = os.getenv("DB_PASSWORD", "yoursecurepassword")

    if not all([db_host, db_port, db_name, db_user, db_password]):
        raise ValueError("One or more database connection environment variables are not fully set. Please check your .env file.")
    return f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# --- Define Declarative Base ---
class Base(DeclarativeBase):
    pass

# --- Define RAG System Models (from B5, with FTS extension for Chunk) ---
VECTOR_DIMENSION = 1536

class Document(Base):
    __tablename__ = "documents_sqlalchemy" # Using same names as B5 for consistency

    # Mapped_column defines a direct mapping to a physical database column.
    # primary_key=True: Designates this column as the table's primary key.
    # autoincrement=True: The database automatically generates a value for new rows.
    doc_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False) # nullable=False means this column cannot be NULL in the database
    doc_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    original_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    upload_timestamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationship defines a link to another mapped class (table) for object-oriented navigation.
    # It does not create a column in this table but uses ForeignKeys defined elsewhere.
    # 'back_populates' links to the corresponding relationship attribute in the 'Chunk' class.
    # 'cascade' defines how operations (e.g., delete) on this Document affect related Chunks.
    chunks: Mapped[list["Chunk"]] = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    def __repr__(self): return f"<Document(doc_id={self.doc_id}, source_name='{self.source_name}')>"

class Chunk(Base):
    __tablename__ = "chunks_sqlalchemy"

    chunk_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # ForeignKey creates a physical column that links to the 'documents_sqlalchemy' table.
    # ondelete specifies action on child rows (Chunk) when parent row (Document) is deleted (e.g., CASCADE, RESTRICT, SET NULL)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents_sqlalchemy.doc_id", ondelete="CASCADE"), nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    
    # For Full-Text Search: TSVECTOR is a PostgreSQL data type optimized for text search.
    # It stores processed text (lexemes) for efficient querying.
    chunk_text_tsv: Mapped[TSVECTOR | None] = mapped_column(TSVECTOR, nullable=True)

    # Relationships: 
    # This 'document' attribute allows navigating from a Chunk object to its parent Document object.
    # It uses the 'doc_id' foreign key column for the linkage.
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    # This 'embedding' attribute allows navigating from a Chunk object to its (optional) Embedding object.
    # 'uselist=False' indicates a one-to-one relationship.
    # It relies on a ForeignKey in the 'Embedding' table pointing back to this Chunk.
    embedding: Mapped["Embedding | None"] = relationship("Embedding", back_populates="chunk", uselist=False, cascade="all, delete-orphan")
    
    __table_args__ = (
        # A GIN index on the TSVECTOR column is crucial for Full-Text Search (FTS) performance.
        # It indexes the individual lexemes (processed words) within the TSVECTOR,
        # enabling fast lookups when using FTS operators like `@@`.
        Index('idx_chunk_text_tsv', chunk_text_tsv, postgresql_using='gin'),
    )
    def __repr__(self): return f"<Chunk(chunk_id={self.chunk_id}, text_preview='{self.chunk_text[:30]}...')>"

class Embedding(Base):
    __tablename__ = "embeddings_sqlalchemy"
    # UniqueConstraint ensures that the specified column(s) have unique values across the table.
    # Here, it enforces a one-to-one relationship between Chunk and Embedding via chunk_id.
    # This means each chunk_id can appear at most once in the embeddings table, ensuring a chunk has at most one embedding.
    __table_args__ = (UniqueConstraint('chunk_id', name='uq_embedding_chunk_id_d1'),)
    embedding_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks_sqlalchemy.chunk_id", ondelete="CASCADE"), nullable=False, unique=True)
    embedding_vector: Mapped[list[float]] = mapped_column(Vector(VECTOR_DIMENSION), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True) # String(100) specifies a VARCHAR column with a maximum length of 100 characters.
    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="embedding")
    def __repr__(self): return f"<Embedding(embedding_id={self.embedding_id}, chunk_id={self.chunk_id})>"

class User(Base):
    __tablename__ = "users_sqlalchemy"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    # JSONB is used for user_profile because it allows storing flexible, semi-structured data (like a dictionary).
    # It's efficient for querying and indexing compared to TEXT, and avoids creating numerous nullable columns
    # for potentially varying profile attributes. PostgreSQL provides rich functions for querying JSONB.
    user_profile: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    chat_sessions: Mapped[list["ChatSession"]] = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    def __repr__(self): return f"<User(user_id={self.user_id}, username='{self.username}', is_active={self.is_active})>"

class ChatSession(Base):
    __tablename__ = "chat_sessions_sqlalchemy"
    session_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users_sqlalchemy.user_id", ondelete="SET NULL"), nullable=True)
    start_time: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    session_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    user: Mapped["User | None"] = relationship("User", back_populates="chat_sessions")
    messages: Mapped[list["ChatMessage"]] = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.timestamp")
    def __repr__(self): return f"<ChatSession(session_id={self.session_id}, user_id={self.user_id})>"

class ChatMessage(Base):
    __tablename__ = "chat_messages_sqlalchemy"
    message_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions_sqlalchemy.session_id", ondelete="CASCADE"), nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    sender: Mapped[str] = mapped_column(String(50), nullable=False)
    message_text: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[list[int] | None] = mapped_column(ARRAY(Integer), nullable=True)
    llm_response_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")
    def __repr__(self): return f"<ChatMessage(message_id={self.message_id}, sender='{self.sender}')>"

# For Trigger Demo
class AuditLog(Base):
    __tablename__ = "audit_log_sqlalchemy_d1"
    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_name: Mapped[str] = mapped_column(String(100))
    row_id: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(50)) # e.g., 'INSERT', 'UPDATE'
    change_timestamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    def __repr__(self): return f"<AuditLog(log_id={self.log_id}, table='{self.table_name}', action='{self.action}')>"


# --- 1. In-depth Transactions with SQLAlchemy ---
def demonstrate_transactions(SessionLocal):
    print("\n--- 1. Demonstrating SQLAlchemy Transactions ---")

    # ACID Properties of Transactions Recap:
    # - Atomicity: All operations in a transaction complete successfully, or none do.
    # - Consistency: A transaction brings the database from one valid state to another.
    # - Isolation: Concurrent transactions do not interfere with each other's intermediate states.
    # - Durability: Once a transaction is committed, its changes are permanent.

    print("\na) Explicit session.begin(), commit(), rollback()")
    session = SessionLocal()
    try:
        transaction = session.begin() # Start a new transaction
        user1 = User(username="trans_user_1", user_profile={"info": "explicit begin"})
        session.add(user1)
        # session.flush() # Optionally flush to get ID before commit
        print(f"  Added user1 (ID before commit: {user1.user_id})") # ID might be None or temp if not flushed

        # Simulate an operation that might fail
        if random.choice([True, False]): # Make it succeed or fail randomly
             # This would cause an error if another "trans_user_1" exists due to unique constraint
             # To demonstrate rollback, let's assume this second add is the one that fails
             user_fail = User(username="trans_user_1_conflict", user_profile={"info": "potential fail"})
             session.add(user_fail) # This will succeed
             print("  Added second user for explicit transaction demo.")
        else:
            print("  Simulating an error condition for explicit transaction demo...")
            # Let's try to add a user that would violate a constraint IF user1 was "trans_user_1_conflict"
            # For a clearer demo, let's force an error by trying to add user1 again if it had a fixed name.
            # Instead, let's commit user1 and try to add another user in a *new* transaction for error handling demo.
            # For this part, let's assume the happy path.
            pass
        
        transaction.commit() # Commit the transaction
        print(f"  Transaction committed. user1 ID: {user1.user_id}")
    except Exception as e:
        print(f"  ERROR in explicit transaction: {e}")
        if transaction.is_active: # Check if transaction is still active before rolling back
            transaction.rollback()
        print("  Transaction rolled back.")
    finally:
        session.close() # Always close the session

    print("\nb) Using `with Session(engine) as session:` and `with session.begin():` (Recommended)")
    # This pattern handles begin, commit, and rollback automatically.
    try:
        with SessionLocal() as session:
            with session.begin(): # Outer transaction
                user2 = User(username="trans_user_2", user_profile={"info": "with block success"})
                session.add(user2)
                print(f"  Added user2 (ID: {user2.user_id}) within 'with session.begin()' block.")
                # If this block completes without error, transaction is committed.
            print("  'with session.begin()' block completed, transaction committed.")
    except Exception as e:
        print(f"  ERROR in 'with session.begin()' block: {e}")
        print("  Transaction automatically rolled back due to error in 'with session.begin()'.")

    # Demonstrating automatic rollback
    try:
        with SessionLocal() as session:
            with session.begin():
                user3 = User(username="trans_user_3_fail", user_profile={"info": "intended fail"})
                session.add(user3)
                print(f"  Added user3 (ID: {user3.user_id})")
                raise ValueError("Simulated error to trigger automatic rollback!")
            print("  This line should not be reached if error occurs.") # Should not print
    except ValueError as e:
        print(f"  Caught simulated error: {e}")
        print("  Transaction for user3 automatically rolled back.")
    
    # Verify user3 was not committed
    with SessionLocal() as session:
        check_user3 = session.execute(select(User).where(User.username == "trans_user_3_fail")).scalar_one_or_none()
        print(f"  Verification: User 'trans_user_3_fail' exists? {'Yes' if check_user3 else 'No'}")


    print("\nc) Nested Transactions (Savepoints)")
    # SQLAlchemy uses savepoints for nested transactions.
    with SessionLocal() as session:
        with session.begin(): # Outer transaction
            user_outer = User(username="user_outer_savepoint", user_profile={"info": "outer"})
            session.add(user_outer)
            print(f"  Added user_outer (ID: {user_outer.user_id}) in outer transaction.")

            try:
                with session.begin_nested(): # Inner transaction (SAVEPOINT)
                    user_inner_fail = User(username="user_inner_savepoint_fail", user_profile={"info": "inner fail"})
                    session.add(user_inner_fail)
                    print(f"    Added user_inner_fail (ID: {user_inner_fail.user_id}) in nested transaction.")
                    raise ValueError("Simulated error in nested transaction!")
                print("    This line in nested try should not be reached.") # Should not print
            except ValueError as e:
                print(f"    Caught error in nested transaction: {e}")
                # The current session.begin_nested() block handles its own rollback.
                # The outer transaction is still active.
                print("    Nested transaction rolled back (to savepoint). Outer transaction continues.")
            
            user_inner_ok = User(username="user_inner_savepoint_ok", user_profile={"info": "inner ok after failed nested"})
            session.add(user_inner_ok)
            print(f"  Added user_inner_ok (ID: {user_inner_ok.user_id}) in outer transaction after failed nested.")
        # Outer transaction commits here (user_outer and user_inner_ok should be saved)
        print("  Outer transaction committed.")

    # Verify savepoint behavior
    with SessionLocal() as session:
        check_outer = session.execute(select(User).where(User.username == "user_outer_savepoint")).scalar_one_or_none()
        check_inner_fail = session.execute(select(User).where(User.username == "user_inner_savepoint_fail")).scalar_one_or_none()
        check_inner_ok = session.execute(select(User).where(User.username == "user_inner_savepoint_ok")).scalar_one_or_none()
        print(f"  Verification: user_outer_savepoint exists? {'Yes' if check_outer else 'No'}")
        print(f"  Verification: user_inner_savepoint_fail exists? {'Yes' if check_inner_fail else 'No'}") # Should be No
        print(f"  Verification: user_inner_savepoint_ok exists? {'Yes' if check_inner_ok else 'No'}")


# --- 2. Interacting with Advanced PostgreSQL Features ---
# Note on using raw SQL for setup of advanced database objects:
# - Tables: Defined as Python classes (ORM models) and created via `Base.metadata.create_all(engine)`.
#           SQLAlchemy generates the `CREATE TABLE` SQL for you.
# - Advanced DB Objects (Views, Materialized Views, Stored Functions, Triggers):
#   The following function `setup_advanced_pg_features` uses raw SQL strings 
#   (via `connection.execute(text(...))`) to define these specific types of objects.
#   This is generally done because:
#     1. SQLAlchemy's ORM focuses on mapping Python classes to tables. These other advanced
#        DB objects don't always have direct ORM definition counterparts for their *creation*.
#     2. The DDL (Data Definition Language) syntax for these objects is often database-specific.
#        Raw SQL ensures the correct native syntax for PostgreSQL is used.
#     3. For DDL of these non-table objects, raw SQL can be clearer.
#   SQLAlchemy still uses the underlying DBAPI driver (like psycopg3) to execute these raw SQL commands.

def setup_advanced_pg_features(engine):
    print("\n--- Setting up Advanced PostgreSQL Features (Views, Functions, Triggers) ---")
    with engine.connect() as connection:
        # a) View
        connection.execute(text("DROP VIEW IF EXISTS active_users_view_d1;"))
        connection.execute(text("""
            CREATE VIEW active_users_view_d1 AS
            SELECT user_id, username, user_profile->>'email' as email
            FROM users_sqlalchemy
            WHERE is_active = TRUE;
        """))
        print("  Created View: active_users_view_d1")

        # b) Materialized View
        connection.execute(text("DROP MATERIALIZED VIEW IF EXISTS document_stats_mv_d1;"))
        connection.execute(text("""
            CREATE MATERIALIZED VIEW document_stats_mv_d1 AS
            SELECT d.source_name, COUNT(c.chunk_id) as chunk_count
            FROM documents_sqlalchemy d
            LEFT JOIN chunks_sqlalchemy c ON d.doc_id = c.doc_id
            GROUP BY d.source_name;
        """))
        print("  Created Materialized View: document_stats_mv_d1")

        # c) Stored Function
        connection.execute(text("DROP FUNCTION IF EXISTS get_user_activity_summary_d1(integer);"))
        connection.execute(text("""
            CREATE OR REPLACE FUNCTION get_user_activity_summary_d1(p_user_id INTEGER)
            RETURNS TEXT AS $$
            DECLARE
                uname TEXT;
                session_count INTEGER;
            BEGIN
                SELECT username INTO uname FROM users_sqlalchemy WHERE user_id = p_user_id;
                IF NOT FOUND THEN
                    RETURN 'User not found.';
                END IF;
                SELECT COUNT(*) INTO session_count FROM chat_sessions_sqlalchemy WHERE user_id = p_user_id;
                RETURN 'User: ' || uname || ' has ' || session_count || ' chat session(s).';
            END;
            $$ LANGUAGE plpgsql;
        """))
        print("  Created Stored Function: get_user_activity_summary_d1")

        # d) Trigger and Audit Table
        # Audit table is defined as SQLAlchemy model AuditLog and created by Base.metadata.create_all()
        connection.execute(text("DROP TRIGGER IF EXISTS audit_user_inserts_d1 ON users_sqlalchemy;"))
        connection.execute(text("DROP FUNCTION IF EXISTS log_user_insert_d1();"))
        connection.execute(text("""
            CREATE OR REPLACE FUNCTION log_user_insert_d1()
            RETURNS TRIGGER AS $$
            BEGIN
                INSERT INTO audit_log_sqlalchemy_d1 (table_name, row_id, action, details)
                VALUES ('users_sqlalchemy', NEW.user_id, 'INSERT', jsonb_build_object('username', NEW.username));
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """))
        connection.execute(text("""
            CREATE TRIGGER audit_user_inserts_d1
            AFTER INSERT ON users_sqlalchemy
            FOR EACH ROW EXECUTE FUNCTION log_user_insert_d1();
        """))
        print("  Created Trigger: audit_user_inserts_d1 and its logging function.")
        connection.commit()


def demonstrate_advanced_features(SessionLocal):
    print("\n--- 2. Demonstrating Advanced PostgreSQL Features via SQLAlchemy ---")

    with SessionLocal() as session:
        # Create some initial data
        user_adv1 = User(username="adv_user_active", is_active=True, user_profile={"email": "active@example.com"})
        user_adv2 = User(username="adv_user_inactive", is_active=False, user_profile={"email": "inactive@example.com"})
        user_adv3 = User(username="adv_user_for_func", is_active=True) # For function and trigger demo
        
        doc1 = Document(source_name="Advanced PG Doc 1", original_content="Content for doc 1")
        chunk1_1 = Chunk(document=doc1, chunk_text="First chunk of doc 1. Searchable content.", chunk_text_tsv=func.to_tsvector('english', "First chunk of doc 1. Searchable content."))
        chunk1_2 = Chunk(document=doc1, chunk_text="Second chunk. More searchable items.", chunk_text_tsv=func.to_tsvector('english', "Second chunk. More searchable items."))
        
        doc2 = Document(source_name="Advanced PG Doc 2", original_content="Content for doc 2")
        chunk2_1 = Chunk(document=doc2, chunk_text="Only chunk for doc 2. Test FTS.", chunk_text_tsv=func.to_tsvector('english', "Only chunk for doc 2. Test FTS."))

        session.add_all([user_adv1, user_adv2, user_adv3, doc1, doc2]) # doc1 will cascade add chunks
        session.commit()
        print(f"  Added sample users and documents for advanced features demo. User for trigger: {user_adv3.user_id}")

        # a) Querying a View
        print("\na) Querying 'active_users_view_d1':")
        # One way to query a view is using text() if not mapped to a model
        active_users_from_view = session.execute(text("SELECT username, email FROM active_users_view_d1")).fetchall()
        for user_row in active_users_from_view:
            print(f"  View User: {user_row.username}, Email: {user_row.email}")

        # b) Materialized Views
        print("\nb) Querying 'document_stats_mv_d1' (Materialized View):")
        # Initial query (might be empty or based on data before setup if MV wasn't refreshed)
        stats_before_refresh = session.execute(text("SELECT source_name, chunk_count FROM document_stats_mv_d1")).fetchall()
        print("  Stats before refresh:")
        for stat in stats_before_refresh:
            print(f"    {stat.source_name}: {stat.chunk_count} chunks")
        
        print("  Refreshing 'document_stats_mv_d1'...")
        session.execute(text("REFRESH MATERIALIZED VIEW document_stats_mv_d1;"))
        session.commit() # REFRESH MV is transactional in some contexts or needs commit
        
        stats_after_refresh = session.execute(text("SELECT source_name, chunk_count FROM document_stats_mv_d1")).fetchall()
        print("  Stats after refresh:")
        for stat in stats_after_refresh:
            print(f"    {stat.source_name}: {stat.chunk_count} chunks")

        # c) Calling a Stored Function
        print("\nc) Calling Stored Function 'get_user_activity_summary_d1':")
        if user_adv3.user_id:
            summary = session.execute(select(func.get_user_activity_summary_d1(user_adv3.user_id))).scalar_one()
            print(f"  Summary for user ID {user_adv3.user_id}: {summary}")
        
        # d) Observing a Trigger
        print("\nd) Observing Trigger 'audit_user_inserts_d1':")
        # The trigger fired when user_adv3 was inserted. Let's check AuditLog.
        audit_logs = session.execute(
            select(AuditLog).where(AuditLog.table_name == 'users_sqlalchemy', AuditLog.row_id == user_adv3.user_id)
        ).scalars().all()
        print(f"  Audit logs for user ID {user_adv3.user_id} insertion:")
        for log_entry in audit_logs:
            print(f"    Log ID: {log_entry.log_id}, Action: {log_entry.action}, Details: {log_entry.details}")
        if not audit_logs:
            print("    No audit log found for this user insertion (trigger might not have fired as expected or data issue).")

        # e) Full-Text Search (FTS)
        print("\ne) Full-Text Search on Chunks:")
        # Assumes chunk_text_tsv was populated correctly (done during initial add)
        # For new chunks or updates, you'd do:
        # new_chunk.chunk_text_tsv = func.to_tsvector('english', new_chunk.chunk_text)
        
        search_term = "searchable"
        print(f"  Searching for chunks containing '{search_term}' (using .match with default tsquery conversion):")
        # .match() on a TSVECTOR column with a string argument typically uses
        # plainto_tsquery or to_tsquery by default, with the specified regconfig.
        query_fts = select(Chunk.chunk_id, Chunk.chunk_text).where(
            Chunk.chunk_text_tsv.match(search_term, postgresql_regconfig='english')
        )
        
        found_chunks = session.execute(query_fts).fetchall()
        if found_chunks:
            for chk_row in found_chunks:
                print(f"    Found Chunk ID: {chk_row.chunk_id}, Text: '{chk_row.chunk_text[:50]}...'")
        else:
            print(f"    No chunks found matching '{search_term}'.")

        search_expression = "searchable & content" # Phrase search using FTS operators
        print(f"  Searching for chunks matching expression '{search_expression}' (using @@ operator with to_tsquery):")
        # For complex expressions with FTS operators, use to_tsquery explicitly 
        # with the custom '@@' operator.
        query_fts_expression = select(Chunk.chunk_id, Chunk.chunk_text).where(
            Chunk.chunk_text_tsv.op('@@')(func.to_tsquery('english', search_expression))
        )
        found_chunks_expression = session.execute(query_fts_expression).fetchall()
        if found_chunks_expression:
            for chk_row in found_chunks_expression:
                print(f"    Found Chunk ID: {chk_row.chunk_id}, Text: '{chk_row.chunk_text[:50]}...'")
        else:
            print(f"    No chunks found matching expression '{search_expression}'.")


def cleanup_advanced_pg_features(engine):
    print("\n--- Cleaning up Advanced PostgreSQL Features ---")
    with engine.connect() as connection:
        connection.execute(text("DROP VIEW IF EXISTS active_users_view_d1;"))
        connection.execute(text("DROP MATERIALIZED VIEW IF EXISTS document_stats_mv_d1;"))
        connection.execute(text("DROP TRIGGER IF EXISTS audit_user_inserts_d1 ON users_sqlalchemy;"))
        connection.execute(text("DROP FUNCTION IF EXISTS log_user_insert_d1();"))
        connection.execute(text("DROP FUNCTION IF EXISTS get_user_activity_summary_d1(integer);"))
        # AuditLog table will be dropped by Base.metadata.drop_all()
        connection.commit()
    print("  Cleaned up custom DB objects.")


if __name__ == "__main__":
    print("Starting Module D1: Advanced PostgreSQL Features & Transactions with SQLAlchemy")
    db_url_str = ""
    engine = None
    try:
        db_url_str = get_db_url()
        # The 'engine' is the SQLAlchemy object that manages the database connection pool and dialect.
        # It's the entry point for all database interactions.
        engine = create_engine(db_url_str, echo=False) # echo=True to see SQL

        # Create a SessionLocal class to create db sessions
        SessionLocalGlobal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        # For demonstration purposes, drop and recreate all tables to ensure a clean state on each run.
        print("\nDropping all existing tables defined in Base.metadata...")
        Base.metadata.drop_all(bind=engine)
        print("Creating all tables defined in Base.metadata...")
        Base.metadata.create_all(bind=engine)
        
        # Setup PG-specific features (views, functions, etc.)
        setup_advanced_pg_features(engine)

        demonstrate_transactions(SessionLocalGlobal)
        demonstrate_advanced_features(SessionLocalGlobal)

    except ValueError as e:
        print(f"Configuration error: {e}")
    except ImportError as e:
        print(f"Import error: {e}. Ensure required libraries are installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        if engine:
            cleanup_advanced_pg_features(engine) # Clean up views, functions, etc.
            # Base.metadata.drop_all(bind=engine) # Optionally drop tables again after demo
            # print("Final drop of all tables complete.")
            engine.dispose() # Close all connections in the connection pool

    print("\nModule D1 execution finished.")
    print("Review 'D1_advanced_postgres_transactions.py' for details.")