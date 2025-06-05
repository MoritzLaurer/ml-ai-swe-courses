# Module B5: ORMs: SQLAlchemy with PostgreSQL for RAG

# This script introduces Object-Relational Mappers (ORMs) using SQLAlchemy 2.x
# to interact with the PostgreSQL RAG database. It covers defining SQLAlchemy
# models corresponding to our RAG schema (from B3), creating the schema,
# and performing CRUD (Create, Read, Update, Delete) operations, including
# vector similarity searches with the `pgvector` integration. This module
# contrasts ORM usage with the direct `psycopg3` operations shown in B4.


# --- Introduction to Object-Relational Mappers (ORMs) and SQLAlchemy ---
# What is an ORM?
# An Object-Relational Mapper (ORM) is a programming technique for converting data
# between incompatible type systems using object-oriented programming languages.
# Essentially, it allows you to interact with your relational database using
# Python objects and methods, instead of writing raw SQL queries.
# SQLAlchemy here uses psycopg3 as the driver to connect to PostgreSQL and 
# lets you interact with the database using Pythonic code, rather than writing raw SQL.
#
# Benefits of using an ORM:
# - Write more Pythonic code: Work with objects and classes.
# - Reduced SQL boilerplate: ORM handles generating most SQL.
# - Database independence (to an extent): Easier to switch DBs, though advanced
#   features might still require DB-specific code.
# - Improved developer productivity: Faster development for common tasks.
# - Abstraction: Hides some complexities of direct DB interaction.
#
# SQLAlchemy:
# SQLAlchemy is a popular and powerful SQL toolkit and ORM for Python.
# It provides both a "Core" expression language (allowing programmatic SQL
# construction) and a full-blown "ORM" (mapping Python classes to DB tables).
# We will focus on SQLAlchemy 2.x style, which emphasizes a more explicit and
# consistent API, particularly with its `select()` based querying for the ORM.

# Note: To run this script, you also need the sqlalchemy package:
# `uv add pgvector sqlalchemy`  (or `uv pip install sqlalchemy pgvector`)
# And ensure your PostgreSQL database has the pgvector extension enabled,
# as done in Module B3: `CREATE EXTENSION IF NOT EXISTS vector;`

# --- Layered Architecture of Database Interaction: ---
# 1. PostgreSQL: The actual database server that stores and manages data.
#    (The Library & Head Librarian)
# 2. SQL: The language used to query and manipulate data in PostgreSQL.
#    (The Language spoken by the Librarian)
# 3. psycopg3 (or other DBAPI driver): A Python library that sends SQL to PostgreSQL
#    and gets data back. It translates between Python and PostgreSQL.
#    (The Messenger who speaks the Librarian's language)
# 4. SQLAlchemy: A Python library that provides higher-level tools.
#    - SQLAlchemy Core: Helps build SQL queries using Python code.
#    - SQLAlchemy ORM: Lets you work with Python objects that map to database tables.
#    SQLAlchemy *uses* psycopg3 (or another driver) to do the actual database communication.
#    (The Sophisticated Personal Assistant who uses the Messenger)
# ---

# --- Comparison: psycopg3 direct SQL vs. SQLAlchemy ORM ---
#
# `psycopg3` (Direct SQL):
#   Pros:
#     - Full control over SQL: Optimize queries exactly as needed.
#     - Potentially better performance for highly complex/tuned queries if ORM generates sub-optimal SQL.
#     - Less abstraction overhead.
#     - Good for scenarios where you want to write and manage SQL explicitly.
#   Cons:
#     - More verbose: Requires writing SQL strings for all operations.
#     - Prone to SQL injection if not careful with parameterization (though psycopg3 helps).
#     - Data mapping: Manually map rows to Python objects or dictionaries.
#     - Refactoring table/column names can be error-prone (need to update all SQL strings).
#
# `SQLAlchemy ORM`:
#   Pros:
#     - Pythonic: Interact with DB using Python objects and methods.
#     - Abstraction: Reduces SQL boilerplate for common CRUD and query operations.
#     - Database Agnostic (mostly): Easier to switch databases for basic operations.
#     - Productivity: Faster development for many tasks.
#     - Maintainability: Changes to schema (e.g., column names) in models can be more easily managed.
#       Tools like Alembic help with schema migrations.
#     - Relationships: Easily define and navigate relationships between objects (e.g., `doc.chunks`).
#     - Unit of Work: Session manages object state and transactions.
#   Cons:
#     - Learning Curve: SQLAlchemy has a rich API that takes time to master.
#     - Abstraction Overhead: Can sometimes generate less efficient SQL than hand-tuned queries,
#       though it's generally quite good. SQLAlchemy Core can be used for more control.
#     - "Magic": Can sometimes be unclear what SQL is being generated without inspection.
#
# For RAG Operations:
# - Storing documents, chunks, embeddings: ORM can be very convenient for creating and managing these structured objects and their relationships.
# - Vector similarity search: Libraries like `pgvector-sqlalchemy` integrate well, allowing you to use vector operations within the ORM's query language.
# - Complex queries: SQLAlchemy's `select()` construct is very powerful and flexible. For highly specialized queries not easily expressed via ORM, you can still drop down to execute raw SQL strings via `session.execute(text(...))`.
#
# Conclusion: For many applications, including RAG backends, SQLAlchemy ORM (especially the 2.x style) offers a good balance of productivity, maintainability, and power. It allows developers to focus more on the application logic rather than SQL minutiae for common tasks.


import os
import datetime
import random

from dotenv import load_dotenv

from sqlalchemy import create_engine, select, func, String, Text, ForeignKey, Integer, TIMESTAMP, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

# Import Vector type from pgvector-sqlalchemy
# This assumes the pgvector extension is installed in your PostgreSQL database
# and the pgvector-sqlalchemy library is installed in your Python environment.
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    print("pgvector.sqlalchemy not found. Please ensure 'pgvector' package is installed: uv add sqlalchemy pgvector")
    # Define a dummy Vector type if not found, so the script can be parsed
    # but operations requiring it will likely fail.
    class Vector:
        def __init__(self, dimensions):
            self.dimensions = dimensions
        def __call__(self, *args, **kwargs):
            return None # Placeholder

# --- SQLAlchemy 2.x Core Components ---
# These are the fundamental building blocks when working with SQLAlchemy ORM.
# 1. Engine: Manages database connections and dialects (e.g., for PostgreSQL).
#    It's the central source of database connectivity, created once per application.
#    Example: `engine = create_engine("postgresql+psycopg://user:pass@host/db")`
#
# 2. Session: Manages persistence operations (transactions) for ORM-mapped objects.
#    It's the primary interface for querying and saving data. Think of it as a
#    "workspace" for your objects that are linked to the database.
#    You typically create short-lived sessions for specific tasks.
#    Example: `with Session(engine) as session:`
#
# 3. DeclarativeBase: A base class that your own model classes will inherit from.
#    It uses Python metaclass programming to translate your Python class
#    definitions (e.g., `class User(Base): ...`) into database table metadata.
#    This is the foundation of the "declarative" mapping style.
#    Example: `class Base(DeclarativeBase): pass`
#
# 4. Mapped and mapped_column: Used within model classes (that inherit from `DeclarativeBase`)
#    to define attributes that correspond to database columns.
#    - `mapped_column(...)` specifies the column's data type (e.g., `String`, `Integer`, `Vector`),
#      constraints (`primary_key=True`, `ForeignKey(...)`, `nullable=False`), and other properties.
#    - `Mapped[python_type]` is a type hint that works with `mapped_column` to provide
#      static type checking for your model attributes.
#    - How it works: When you define `my_attribute: Mapped[int] = mapped_column(Integer)`,
#      SQLAlchemy doesn't store this as a simple class variable. Instead, it uses
#      Python's descriptor protocol. This means `my_attribute` becomes a special
#      object on the class that manages how values are get/set on *instances* of
#      your model (e.g., `user_instance.my_attribute`). This allows SQLAlchemy to
#      track changes, handle loading data from the DB, etc., without you needing to
#      write `self.my_attribute` in the class definition itself; the mapping is declarative.
#    Example: `username: Mapped[str] = mapped_column(String(100), unique=True)`

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
    # SQLAlchemy URL format: postgresql+psycopg://user:password@host:port/dbname
    # We use 'psycopg' as the driver, assuming psycopg3 is installed,
    # which is what 'psycopg[binary]' from A0 provides.
    return f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# --- Define Declarative Base ---
class Base(DeclarativeBase):
    pass

# --- Define RAG System Models (SQLAlchemy 2.x syntax) ---
# These models correspond to the tables designed in Module B3.

VECTOR_DIMENSION = 1536 # Must match your embedding model and B3 definition

class Document(Base):
    __tablename__ = "documents_sqlalchemy" # Using a different suffix to avoid conflict if B3 tables exist

    # Mapped_column defines a direct mapping to a physical database column.
    # primary_key=True: Designates this column as the table's primary key.
    # autoincrement=True: The database automatically generates a value for new rows.
    doc_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False) # nullable=False means this column cannot be NULL in the database
    doc_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    original_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    upload_timestamp: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now() # Use func.now() for DB-side default
    )

    # Relationship defines a link to another mapped class (table) for object-oriented navigation.
    # It does not create a column in this table but uses ForeignKeys defined elsewhere.
    # 'back_populates' links to the corresponding relationship attribute in the 'Chunk' class.
    # 'cascade' defines how operations (e.g., delete) on this Document affect related Chunks.
    chunks: Mapped[list["Chunk"]] = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(doc_id={self.doc_id}, source_name='{self.source_name}')>"

class Chunk(Base):
    __tablename__ = "chunks_sqlalchemy"

    chunk_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # ForeignKey creates a physical column that links to the 'documents_sqlalchemy' table.
    # ondelete specifies action on child rows (Chunk) when parent row (Document) is deleted (e.g., CASCADE, RESTRICT, SET NULL)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents_sqlalchemy.doc_id", ondelete="CASCADE"), nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationships:
    # This 'document' attribute allows navigating from a Chunk object to its parent Document object.
    # It uses the 'doc_id' foreign key column for the linkage.
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    # This 'embedding' attribute allows navigating from a Chunk object to its (optional) Embedding object.
    # 'uselist=False' indicates a one-to-one relationship.
    # It relies on a ForeignKey in the 'Embedding' table pointing back to this Chunk.
    embedding: Mapped["Embedding | None"] = relationship("Embedding", back_populates="chunk", uselist=False, cascade="all, delete-orphan")


    def __repr__(self):
        return f"<Chunk(chunk_id={self.chunk_id}, doc_id={self.doc_id}, text_preview='{self.chunk_text[:30]}...')>"

class Embedding(Base):
    __tablename__ = "embeddings_sqlalchemy"
    # UniqueConstraint ensures that the specified column(s) have unique values across the table.
    # Here, it enforces a one-to-one relationship between Chunk and Embedding via chunk_id.
    # This means each chunk_id can appear at most once in the embeddings table, ensuring a chunk has at most one embedding.
    __table_args__ = (UniqueConstraint('chunk_id', name='uq_embedding_chunk_id'),)

    embedding_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Unique=True on FK for one-to-one if not using __table_args__ for constraint naming.
    # For a true one-to-one where Embedding cannot exist without a Chunk, and Chunk has one Embedding.
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks_sqlalchemy.chunk_id", ondelete="CASCADE"), nullable=False, unique=True)
    embedding_vector: Mapped[list[float]] = mapped_column(Vector(VECTOR_DIMENSION), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True) # String(100) specifies a VARCHAR column with a maximum length of 100 characters.

    # Relationship to Chunk
    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="embedding")

    def __repr__(self):
        return f"<Embedding(embedding_id={self.embedding_id}, chunk_id={self.chunk_id}, model='{self.model_name}')>"

class User(Base):
    __tablename__ = "users_sqlalchemy"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    # JSONB is used for user_profile because it allows storing flexible, semi-structured data (like a dictionary).
    # It's efficient for querying and indexing compared to TEXT, and avoids creating numerous nullable columns
    # for potentially varying profile attributes. PostgreSQL provides rich functions for querying JSONB.
    user_profile: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationship to ChatSessions
    chat_sessions: Mapped[list["ChatSession"]] = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(user_id={self.user_id}, username='{self.username}')>"

class ChatSession(Base):
    __tablename__ = "chat_sessions_sqlalchemy"

    session_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # ON DELETE SET NULL: if user is deleted, session user_id becomes NULL.
    # If you want to delete sessions when user is deleted, use ondelete="CASCADE" on ForeignKey
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users_sqlalchemy.user_id", ondelete="SET NULL"), nullable=True)
    start_time: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    session_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationship to User
    user: Mapped["User | None"] = relationship("User", back_populates="chat_sessions")
    # Relationship to ChatMessages
    messages: Mapped[list["ChatMessage"]] = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.timestamp")

    def __repr__(self):
        return f"<ChatSession(session_id={self.session_id}, user_id={self.user_id})>"

class ChatMessage(Base):
    __tablename__ = "chat_messages_sqlalchemy"

    message_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions_sqlalchemy.session_id", ondelete="CASCADE"), nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    sender: Mapped[str] = mapped_column(String(50), nullable=False) # E.g., 'user', 'bot', 'system'
    message_text: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[list[int] | None] = mapped_column(ARRAY(Integer), nullable=True)
    llm_response_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationship to ChatSession
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage(message_id={self.message_id}, session_id={self.session_id}, sender='{self.sender}')>"


# --- Main SQLAlchemy ORM Demo Function ---
def sqlalchemy_orm_demo(engine):
    """Demonstrates schema creation and CRUD operations using SQLAlchemy ORM."""

    # Create a SessionLocal class to create db sessions
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    print("--- SQLAlchemy ORM Demo ---")

    # --- Drop and Recreate Schema for a clean demo run ---
    # This will drop tables if they exist and then recreate them.
    # WARNING: This is destructive and should not be used in production
    # without careful consideration. It's suitable for a demo script.
    print("\nDropping existing tables (if any) defined in Base.metadata...")
    Base.metadata.drop_all(bind=engine)
    print("Tables dropped.")
    
    print("\nCreating database schema...")
    Base.metadata.create_all(bind=engine)
    print("Schema creation step complete.")

    # --- CRUD Operations ---
    # Using `with Session(engine) as session:` for SQLAlchemy 2.x style context management
    with SessionLocal() as session:
        print("\n--- CRUD Operations Demo ---")

        # 1. CREATE
        print("\n1. Creating new records...")
        # Create a User
        new_user = User(username="alice_sqlalchemy", user_profile={"theme": "light", "language": "en"})
        session.add(new_user)
        # Note: `new_user` object might not have its ID until committed or flushed if using autoincrement PK.
        # We need to commit to get the ID for subsequent operations.

        # Create a Document
        new_document = Document(
            source_name="SQLAlchemy Guide",
            doc_metadata={"version": "2.0", "author": "SQLAlchemy Team"},
            original_content="SQLAlchemy is a powerful ORM..."
        )
        session.add(new_document)
        session.commit() # Commit to save user and document, and get their IDs
        print(f"Created User: {new_user}")
        print(f"Created Document: {new_document}")

        # Create Chunks for the Document
        chunk1_text = "Introduction to ORM concepts and SQLAlchemy Core."
        chunk2_text = "Working with Sessions and the Declarative Mapping."
        
        new_chunk1 = Chunk(
            document=new_document, # Associate with the document
            chunk_text=chunk1_text,
            token_count=len(chunk1_text.split()),
            chunk_metadata={"page": 1, "section": "1.1"}
        )
        new_chunk2 = Chunk(
            doc_id=new_document.doc_id, # Corrected: Use the mapped attribute name 'doc_id'
            chunk_text=chunk2_text,
            token_count=len(chunk2_text.split()),
            chunk_metadata={"page": 5, "section": "2.3"}
        )
        session.add_all([new_chunk1, new_chunk2])
        session.commit() # Commit to save chunks and get their IDs
        print(f"Created Chunk 1: {new_chunk1}")
        print(f"Created Chunk 2: {new_chunk2}")

        # Create Embedding for Chunk1
        dummy_vector_1 = [random.uniform(-1, 1) for _ in range(VECTOR_DIMENSION)]
        new_embedding1 = Embedding(
            chunk=new_chunk1, # Associate with chunk1
            embedding_vector=dummy_vector_1,
            model_name="dummy_model_v1"
        )
        session.add(new_embedding1)
        session.commit()
        print(f"Created Embedding for Chunk 1: {new_embedding1}")

        # Create a Chat Session and Messages for the User
        new_session = ChatSession(user=new_user, session_metadata={"topic": "SQLAlchemy Basics"})
        session.add(new_session)
        session.commit() # Get session_id

        msg1 = ChatMessage(
            session=new_session,
            sender="user",
            message_text="What is SQLAlchemy?",
        )
        msg2 = ChatMessage(
            session_id=new_session.session_id, # Can also assign by FK
            sender="bot",
            message_text="SQLAlchemy is a Python SQL toolkit and ORM.",
            retrieved_chunk_ids=[new_chunk1.chunk_id] # Example of using array
        )
        session.add_all([msg1, msg2])
        session.commit()
        print(f"Created ChatSession: {new_session} with messages.")



        # 2. READ
        print("\n2. Reading records...")
        # Get a user by primary key
        retrieved_user = session.get(User, new_user.user_id)
        print(f"Retrieved User by ID ({new_user.user_id}): {retrieved_user.username if retrieved_user else 'Not found'}")

        # Query using select (SQLAlchemy 2.x style)
        print("\nQuerying for documents by source name:")
        stmt_docs = select(Document).where(Document.source_name.like("%SQLAlchemy%"))
        documents = session.scalars(stmt_docs).all() # scalars() for single-column entities, execute() for multiple cols/entities
        for doc in documents:
            print(f"  Found Document: {doc.source_name}, Metadata: {doc.doc_metadata}")
            # Access related chunks (lazy-loaded by default)
            print(f"    Chunks for '{doc.source_name}':")
            for chk in doc.chunks: # This will trigger a query if not already loaded
                 print(f"      - Chunk ID: {chk.chunk_id}, Text: '{chk.chunk_text[:20]}...'")
                 if chk.embedding:
                     print(f"        Embedding Model: {chk.embedding.model_name}")

        print("\nQuerying chunks with their document (explicit join):")
        stmt_chunks_join = (
            select(Chunk, Document.source_name.label("doc_source")) # Select all columns from the Chunk model and the source_name from Document (aliased as "doc_source")
            .join(Document, Chunk.doc_id == Document.doc_id)       # Explicitly JOIN Chunk with Document using the foreign key relationship
            .where(Document.source_name == "SQLAlchemy Guide")      # Filter results where the document's source_name is "SQLAlchemy Guide"
            .order_by(Chunk.chunk_id)                               # Order the results by chunk_id
        )
        # Execute the query defined above against the database
        results = session.execute(stmt_chunks_join).all() # Returns Row objects
        for row in results:
            chunk_obj = row.Chunk # The Chunk object
            doc_source_name = row.doc_source # The labeled column
            print(f"  Chunk ID: {chunk_obj.chunk_id}, Text: '{chunk_obj.chunk_text[:30]}...', Document Source: {doc_source_name}")



        # 3. UPDATE
        print("\n3. Updating records...")
        if retrieved_user:
            retrieved_user.user_profile = {"theme": "dark", "language": "en", "status": "active"}
            # The session tracks changes to ORM objects.
            session.commit()
            print(f"Updated user '{retrieved_user.username}' profile: {retrieved_user.user_profile}")

        # Update a chunk's text
        chunk_to_update = session.get(Chunk, new_chunk1.chunk_id)
        if chunk_to_update:
            chunk_to_update.chunk_text = "Updated: An introduction to ORM concepts and the SQLAlchemy Core library."
            session.commit()
            print(f"Updated chunk {chunk_to_update.chunk_id} text.")



        # 4. DELETE
        print("\n4. Deleting records...")
        # Create a temporary user to delete
        temp_user = User(username="temp_user_to_delete")
        session.add(temp_user)
        session.commit()
        user_id_to_delete = temp_user.user_id
        print(f"Created temporary user for deletion: {temp_user.username} (ID: {user_id_to_delete})")

        user_to_delete = session.get(User, user_id_to_delete)
        if user_to_delete:
            session.delete(user_to_delete)
            session.commit()
            print(f"Deleted user with ID: {user_id_to_delete}")
            # Verify deletion
            deleted_user_check = session.get(User, user_id_to_delete)
            print(f"User {user_id_to_delete} exists after delete? {'Yes' if deleted_user_check else 'No'}")

        # Note on cascade deletes:
        # If Document is deleted, its Chunks and their Embeddings should also be deleted
        # due to `cascade="all, delete-orphan"` and `ondelete="CASCADE"` on FKs.
        print("\nTesting cascade delete (Document -> Chunks -> Embeddings):")
        doc_to_delete_id = new_document.doc_id
        # Fetch the document again to ensure it's in the current session context if needed,
        # or rely on the existing `new_document` object if it's still valid.
        doc_to_delete = session.get(Document, doc_to_delete_id)
        if doc_to_delete:
            session.delete(doc_to_delete)
            session.commit()
            print(f"Deleted Document with ID: {doc_to_delete_id}. Related chunks and embeddings should also be gone.")
            # Verify deletion of chunks (associated with the deleted document)
            remaining_chunks_count_stmt = select(func.count(Chunk.chunk_id)).where(Chunk.doc_id == doc_to_delete_id)
            remaining_chunks_count = session.scalar(remaining_chunks_count_stmt)
            print(f"Chunks remaining for deleted doc_id {doc_to_delete_id}: {remaining_chunks_count}")



        # --- Vector Similarity Search with pgvector-sqlalchemy ---
        print("\n5. Vector Similarity Search (L2 distance)...")
        # We need at least one embedding in the database for this.
        # Let's re-add a document, chunk, and embedding if they were deleted.
        
        # Check if any embeddings exist, if not, add one
        existing_embeddings_count_stmt = select(func.count(Embedding.embedding_id))
        if session.scalar(existing_embeddings_count_stmt) == 0:
            print("  No embeddings found. Adding a sample for demo...")
            sample_doc = Document(source_name="Sample Doc for VecSearch")
            session.add(sample_doc)
            session.flush() # Get sample_doc.doc_id
            sample_chunk = Chunk(document=sample_doc, chunk_text="This is a sample chunk for vector search.")
            session.add(sample_chunk)
            session.flush() # Get sample_chunk.chunk_id
            sample_vector = [random.uniform(-0.5, 0.5) for _ in range(VECTOR_DIMENSION)]
            sample_embedding = Embedding(chunk=sample_chunk, embedding_vector=sample_vector, model_name="dummy_sample_model")
            session.add(sample_embedding)
            session.commit()
            print(f"  Added sample embedding with ID: {sample_embedding.embedding_id}")

        query_vector = [random.uniform(-1, 1) for _ in range(VECTOR_DIMENSION)]
        
        # Using l2_distance (Euclidean distance) from pgvector.sqlalchemy
        # Other options: max_inner_product, cosine_distance
        stmt_vector_search = (
            select(Embedding.embedding_id, Chunk.chunk_text, Embedding.embedding_vector.l2_distance(query_vector).label("distance"))
            .join(Chunk, Embedding.chunk_id == Chunk.chunk_id)
            .order_by(Embedding.embedding_vector.l2_distance(query_vector))
            .limit(3)
        )
        
        similar_embeddings = session.execute(stmt_vector_search).all()
        if similar_embeddings:
            print("  Top 3 similar chunks (by L2 distance to a random query vector):")
            for emb_row in similar_embeddings:
                print(f"    Embedding ID: {emb_row.embedding_id}, Chunk Text: '{emb_row.chunk_text[:30]}...', Distance: {emb_row.distance:.4f}")
        else:
            print("  No embeddings found to perform vector search.")
        
        session.commit() # Final commit for any pending changes

    print("\n--- SQLAlchemy ORM Demo Complete ---")



if __name__ == "__main__":
    print("Starting Module B5: ORMs: SQLAlchemy with PostgreSQL for RAG")
    db_url_str = ""
    try:
        db_url_str = get_db_url()
        # Create an engine
        # echo=True will log all SQL statements generated by SQLAlchemy - useful for debugging/learning.
        # echo=False for production or quieter output.
        engine = create_engine(db_url_str, echo=False)

        # Test connection (optional, create_all will also test it)
        with engine.connect() as connection:
            print("Successfully connected to PostgreSQL using SQLAlchemy engine.")
        
        sqlalchemy_orm_demo(engine)

    except ValueError as e: # From get_db_url
        print(f"Configuration error: {e}")
    except ImportError as e: # For pgvector
        print(f"Import error: {e}. Ensure required libraries are installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\nModule B5 execution finished.")
    print("Review the script 'B5_sqlalchemy_orm_for_rag.py' for details.")
    print("SQLAlchemy models corresponding to RAG tables should have been created/interacted with.")
    print("The tables created are suffixed with '_sqlalchemy' (e.g., 'documents_sqlalchemy').")
