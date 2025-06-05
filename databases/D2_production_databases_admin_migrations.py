# Module D2: Productionizing Databases: Admin, Performance & Migrations with SQLAlchemy

# This module covers essential practices for managing databases in a production
# environment. We'll discuss:
# 1. Database Administration: Core concepts for PostgreSQL.
# 2. Performance and Scalability: Techniques with a focus on SQLAlchemy interactions.
# 3. Security: Best practices when using SQLAlchemy.
# 4. Database Migrations: Using Alembic with SQLAlchemy to manage schema changes.
#
# This script will be more conceptual and explanatory in nature for certain topics
# (like psql commands or Alembic CLI usage), using comments to illustrate.
# For SQLAlchemy-specific parts, we will show code examples.

import os
import traceback
from dotenv import load_dotenv

from sqlalchemy import create_engine, text, inspect, Index, select
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

# Assuming models are defined in a similar way to B5/D1
# For this script, we don't need to redefine all models, but we'll
# refer to them conceptually. We might use a simple model for specific demos.
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, func, inspect

class Base(DeclarativeBase):
    pass

class SimpleUser(Base): # A simple model for demonstration
    __tablename__ = "simple_users_prod_d2"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True) # Example of index on model
    email: Mapped[str] = mapped_column(String(100))

    # Example of a multi-column index defined in __table_args__
    __table_args__ = (
        Index('idx_username_email_prod_d2', 'username', 'email'),
    )

    def __repr__(self):
        return f"<SimpleUser(id={self.id}, username='{self.username}')>"


# --- Database Connection URL ---
def get_db_url():
    load_dotenv()
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "myprojectdb")
    db_user = os.getenv("DB_USER", "myprojectuser")
    db_password = os.getenv("DB_PASSWORD", "yoursecurepassword")
    if not all([db_host, db_port, db_name, db_user, db_password]):
        raise ValueError("DB connection env vars not fully set.")
    return f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# --- 1. Database Administration (Conceptual, PostgreSQL-general) ---
# Database administration involves managing the database server, ensuring its
# availability, performance, security, and data integrity.

def database_administration_concepts():
    print("\n--- 1. Database Administration (Conceptual) ---")
    print("\na) Users, Roles, and Permissions in PostgreSQL:")
    print("  - PostgreSQL uses roles to manage database access permissions.")
    print("  - A role can be a user (if it has LOGIN permission) or a group of roles.")
    print("  - Permissions (privileges) define what a role can do (e.g., SELECT, INSERT, CREATE TABLE).")
    print("  - Principle of Least Privilege: Grant only necessary permissions.")
    print("\n  Example PSQL/SQL commands (run in psql or admin tool):")
    print("  ----------------------------------------------------")
    print("  -- CREATE ROLE myapp_user WITH LOGIN PASSWORD 'securepass';")
    print("  -- CREATE ROLE myapp_readonly_group;")
    print("  -- GRANT CONNECT ON DATABASE myprojectdb TO myapp_user;")
    print("  -- GRANT USAGE ON SCHEMA public TO myapp_user;")
    print("  -- GRANT SELECT ON ALL TABLES IN SCHEMA public TO myapp_readonly_group;")
    print("  -- GRANT myapp_readonly_group TO myapp_user; -- myapp_user inherits permissions")
    print("  -- GRANT INSERT, UPDATE, DELETE ON specific_table TO myapp_user;")
    print("  -- REVOKE DELETE ON specific_table FROM myapp_user;")
    print("  -- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO myapp_readonly_group;")
    print("  ----------------------------------------------------")

    print("\nb) Backup and Recovery (pg_dump, pg_restore):")
    print("  - `pg_dump`: Creates logical backups of a PostgreSQL database.")
    print("    - Can dump a single database, schema, or table.")
    print("    - Output formats: plain SQL script, custom archive (recommended for flexibility).")
    print("  - `pg_restore`: Restores a PostgreSQL database from an archive created by `pg_dump`.")
    print("\n  Example command-line usage:")
    print("  ----------------------------------------------------")
    print("  -- Full backup (custom format):")
    print("  -- pg_dump -U db_user -h localhost -Fc -f mydatabase.dump myprojectdb")
    print("\n  -- Restore (needs an existing empty database or use -C to create):")
    print("  -- createdb -U db_user new_restored_db")
    print("  -- pg_restore -U db_user -h localhost -d new_restored_db mydatabase.dump")
    print("\n  -- Backup (plain SQL format):")
    print("  -- pg_dump -U db_user -h localhost -f mydatabase.sql myprojectdb")
    print("\n  -- Restore (plain SQL format):")
    print("  -- psql -U db_user -h localhost -d new_restored_db -f mydatabase.sql")
    print("  ----------------------------------------------------")
    print("  - Regular backups are CRITICAL for production. Test your restore process!")
    print("  - Consider Point-In-Time Recovery (PITR) for more granular recovery options (uses WAL archiving).")


# --- 2. Performance and Scalability (SQLAlchemy Focus) ---

def performance_and_scalability_discussion(engine):
    print("\n--- 2. Performance and Scalability (SQLAlchemy Focus) ---")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    print("\na) Connection Pooling with SQLAlchemy:")
    print("  - SQLAlchemy's `Engine` manages a connection pool by default.")
    print("  - Pooling improves performance by reusing database connections, avoiding overhead of establishing new ones for each request.")
    print("  - Key `create_engine` parameters for pooling:")
    print("    - `pool_size`: Number of connections to keep open in the pool (default 5).")
    print("    - `max_overflow`: Number of extra connections allowed beyond `pool_size` (default 10).")
    print("    - `pool_timeout`: Seconds to wait for a connection if pool is full (default 30).")
    print("    - `pool_recycle`: Seconds after which a connection is recycled (prevents stale connections, e.g., 3600 for 1 hour).")
    print("    - `pool_pre_ping`: Pings connection before use to check liveness (adds slight overhead but robust).")
    
    # Example of creating an engine with custom pool settings:
    # db_url = get_db_url()
    # prod_engine = create_engine(
    #     db_url,
    #     pool_size=10,
    #     max_overflow=20,
    #     pool_timeout=30,
    #     pool_recycle=1800, # Recycle connections every 30 minutes
    #     pool_pre_ping=True
    # )
    # print(f"  Example: Engine configured with pool_size=10, max_overflow=20. Current engine: {engine.pool.status()}")

    print("\nb) Monitoring and Query Analysis:")
    print("  i. Using EXPLAIN and EXPLAIN ANALYZE with SQLAlchemy:")
    print("     - `EXPLAIN`: Shows the query plan (how PostgreSQL intends to execute the query).")
    print("     - `EXPLAIN ANALYZE`: Executes the query and shows the actual plan with timing and row counts.")
    with SessionLocal() as session:
        # Example query to analyze
        stmt_to_explain = select(SimpleUser).where(SimpleUser.username == "test_explain_user")
        
        # To get the EXPLAIN output:
        # 1. Compile the query to SQL
        # 2. Prepend "EXPLAIN (FORMAT JSON, ANALYZE)" or similar
        # 3. Execute the raw SQL
        
        try:
            compiled_stmt = stmt_to_explain.compile(engine, compile_kwargs={"literal_binds": True})
            # Using literal_binds for readability of EXPLAIN; be cautious in other contexts.
            # For EXPLAIN ANALYZE, parameters are usually fine.
            
            explain_sql = f"EXPLAIN (FORMAT JSON, ANALYZE) {compiled_stmt}" # ANALYZE executes it!
            # explain_sql_no_analyze = f"EXPLAIN (FORMAT JSON) {compiled_stmt}"
            
            print(f"     Executing: {explain_sql[:200]}...") # Show part of the query
            
            # Parameters should be passed separately if not using literal_binds.
            # For a query with parameters:
            # explain_stmt = text("EXPLAIN (FORMAT JSON, ANALYZE) SELECT * FROM simple_users_prod_d2 WHERE username = :uname")
            # result = session.execute(explain_stmt, {"uname": "test_explain_user"}).scalar_one()
            
            # For this demo, we'll execute the literal_binds version for simplicity of EXPLAIN output viewing
            result = session.execute(text(explain_sql)).scalar_one() # scalar_one to get the JSON string
            
            # The result is a JSON string (or list of JSON objects if FORMAT JSON)
            print("     EXPLAIN ANALYZE (JSON output example - first few lines):")
            if result:
                # The result is typically a list containing one JSON object string from EXPLAIN (FORMAT JSON)
                # For simplicity, assuming it's a single string or the first element of a list
                json_result = result[0] if isinstance(result, list) else result
                # In newer psycopg3/SQLAlchemy, EXPLAIN (FORMAT JSON) result might be directly a list of dict
                if isinstance(json_result, dict):
                    plan_details = json_result.get("Plan", {})
                    print(f"       Plan Node Type: {plan_details.get('Node Type')}, Actual Total Time: {plan_details.get('Actual Total Time')}")
                else: # if it's a string representation of JSON
                    for line in str(json_result).splitlines()[:5]:
                         print(f"       {line}")
            else:
                print("     No EXPLAIN result.")

        except Exception as e:
            print(f"     Error during EXPLAIN: {e}")
            # traceback.print_exc()

    print("\n  ii. Querying `pg_stat_statements`:")
    print("      - PostgreSQL extension that tracks execution statistics of all SQL statements.")
    print("      - Needs to be enabled in `postgresql.conf` (shared_preload_libraries) and `CREATE EXTENSION pg_stat_statements;` in the DB.")
    print("      - Provides insights into frequently run queries, execution times, I/O, etc.")
    print("      - Can be queried via SQLAlchemy `text()`.")
    print("      Example query (if extension is enabled):")
    print("      `SELECT query, calls, total_exec_time, rows FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 10;`")
    # with SessionLocal() as session:
    #     try:
    #         pg_stat_stmt = text("SELECT query, calls, total_exec_time FROM pg_stat_statements ORDER BY calls DESC LIMIT 1;")
    #         top_query_stats = session.execute(pg_stat_stmt).first()
    #         if top_query_stats:
    #             print(f"      Example from pg_stat_statements: {top_query_stats}")
    #         else:
    #             print("      pg_stat_statements might not be enabled or no queries tracked yet.")
    #     except SQLAlchemyError as e: # Catch DB errors like relation not found if extension isn't there
    #         print(f"      Could not query pg_stat_statements (is it enabled?): {e}")


    print("\nc) Indexing Review and Best Practices:")
    print("  - Indexes speed up data retrieval by allowing the DB to find rows matching query conditions more quickly.")
    print("  - SQLAlchemy can declare indexes on models (single column `index=True`, multi-column via `__table_args__`).")
    print(f"    Example: `username: Mapped[str] = mapped_column(index=True)` on SimpleUser model.")
    print(f"    Example: `Index('idx_username_email_prod_d2', 'username', 'email')` on SimpleUser model.")
    print("  - Common PostgreSQL Index Types:")
    print("    - B-tree: Default, good for equality and range queries on most data types.")
    print("    - GIN (Generalized Inverted Index): Good for composite types (arrays, JSONB), full-text search.")
    print("    - GiST (Generalized Search Tree): Good for geometric data, full-text search, more complex indexing scenarios.")
    print("    - BRIN (Block Range Index): Good for very large tables where values correlate with physical storage order.")
    print("    - HASH: Only useful for equality comparisons (less common than B-tree).")
    print("    - For `pgvector`: HNSW, IVFFlat are common for approximate nearest neighbor search.")
    print("  - Best Practices:")
    print("    - Index columns frequently used in `WHERE` clauses, `JOIN` conditions, and `ORDER BY` clauses.")
    print("    - Avoid over-indexing: Indexes consume storage and slow down writes (INSERT, UPDATE, DELETE).")
    print("    - Analyze query plans (`EXPLAIN`) to see if indexes are being used effectively.")
    print("    - Consider partial indexes (e.g., `CREATE INDEX ... WHERE some_column IS TRUE`).")
    print("    - For FTS, use GIN or GiST on `tsvector` columns.")


    print("\nd) Read Replicas and Sharding (Conceptual):")
    print("  i. Read Replicas:")
    print("     - Copies of the primary database that handle read-only queries.")
    print("     - Offload read traffic from the primary server, improving its performance for writes.")
    print("     - PostgreSQL supports streaming replication (asynchronous by default).")
    print("     - Application (SQLAlchemy) can be configured to route read queries to replicas and write queries to primary.")
    print("       This typically involves multiple engine configurations or custom routing logic.")
    # Example conceptual routing (not fully implemented here):
    # primary_engine = create_engine(get_db_url_for_primary())
    # replica_engine = create_engine(get_db_url_for_replica())
    # Session = sessionmaker()
    # Session.configure(binds={User: primary_engine, SomeReadOnlyModel: replica_engine})
    # Or a custom Session that selects engine based on operation type.

    print("\n  ii. Sharding (Database Partitioning):")
    print("      - Distributing data across multiple databases or servers (shards).")
    print("      - Each shard holds a subset of the data (e.g., users A-M on shard1, N-Z on shard2).")
    print("      - Improves scalability for very large datasets and high write throughput.")
    print("      - Complex to implement and manage. Application needs to be shard-aware.")
    print("      - SQLAlchemy doesn't manage sharding directly; it's an architectural pattern requiring custom logic")
    print("        to route queries to the correct shard's engine.")
    print("      - Some PostgreSQL extensions (e.g., Citus) provide sharding capabilities.")


# --- 3. Security (SQLAlchemy Focus) ---

def security_considerations(engine_url):
    print("\n--- 3. Security (SQLAlchemy Focus) ---")

    print("\na) SQL Injection Prevention:")
    print("  - SQLAlchemy ORM and Core Expression Language inherently protect against SQL injection.")
    print("  - Queries are constructed using parameterized statements; user input is treated as data, not executable SQL.")
    print("  - Example: `session.execute(select(User).where(User.username == user_input))` is safe.")
    print("  - If using `text()` for raw SQL, ALWAYS use parameter binding:")
    print("    `session.execute(text('SELECT * FROM users WHERE username = :name'), {'name': user_input})` is safe.")
    print("    `session.execute(text(f\"SELECT * FROM users WHERE username = '{user_input}'\"))` is UNSAFE and vulnerable.")

    print("\nb) Secure Connections (SSL/TLS):")
    print("  - Encrypt data in transit between the application and PostgreSQL.")
    print("  - PostgreSQL supports SSL connections.")
    print("  - Configure `create_engine` with SSL parameters in `connect_args` for `psycopg`.")
    print("  Example `connect_args` for SSL (paths to certs depend on setup):")
    print("  ```python")
    print("  ssl_args = {")
    print("      'sslmode': 'verify-full',  # or 'require', 'verify-ca'")
    print("      'sslrootcert': '/path/to/server-ca.pem',")
    print("      'sslcert': '/path/to/client-cert.pem',")
    print("      'sslkey': '/path/to/client-key.pem'")
    print("  }")
    print("  engine = create_engine(db_url, connect_args=ssl_args)")
    print("  ```")
    # Actual connection test with SSL would require a DB configured for SSL and certs.
    # We can check if the current URL implies SSL for psycopg
    if "sslmode=" in engine_url.lower():
        print(f"  Current DB URL ({engine_url}) seems to include SSL parameters.")
    else:
        print(f"  Current DB URL ({engine_url}) does not explicitly show SSL parameters for psycopg driver.")


    print("\nc) Principle of Least Privilege (Database User):")
    print("  - The database user account your application uses should only have the permissions it strictly needs.")
    print("  - E.g., if the app only reads data, grant only `SELECT` privileges.")
    print("  - Avoid using superuser accounts (like `postgres`) for application connections.")
    print("  - Create specific roles/users for your application (see Database Administration section).")
    print(f"  Currently connected with user implied by DB URL: {engine_url.split('://')[1].split(':')[0] if '://' in engine_url else 'unknown'}")


# --- 4. Database Migrations with Alembic (for SQLAlchemy) ---

def database_migrations_with_alembic():
    print("\n--- 4. Database Migrations with Alembic (for SQLAlchemy) ---")
    print("Alembic is a database migration tool specifically for SQLAlchemy.")
    print("It allows you to manage and version your database schema changes over time.")

    print("\na) Why Migrations?")
    print("  - As your application evolves, your database schema (table structures, columns, etc.) will likely change.")
    print("  - Migrations provide a controlled, versioned way to apply these changes to your database.")
    print("  - Ensures consistency across different environments (dev, staging, prod).")
    print("  - Allows rollback to previous schema versions if needed.")

    print("\nb) Introduction to Alembic:")
    print("  - Alembic works by comparing your SQLAlchemy models to the current state of the database.")
    print("  - It generates migration scripts (Python files) containing `upgrade()` and `downgrade()` functions.")
    print("    - `upgrade()`: Applies the schema changes.")
    print("    - `downgrade()`: Reverts the schema changes.")

    print("\nc) Setting up Alembic (Conceptual Commands - run in terminal):")
    print("  1. Install Alembic: `uv pip install alembic` (or `pip install alembic`)")
    print("  2. Initialize Alembic in your project: `alembic init alembic`")
    print("     This creates an 'alembic' directory and an `alembic.ini` file.")
    print("  3. Configure `env.py` (in the 'alembic' directory):")
    print("     - Point to your SQLAlchemy models' `Base.metadata`.")
    print("       ```python")
    print("       # In alembic/env.py, around line 20-25:")
    print("       # from myapp.models import Base  # Import your Base")
    print("       # target_metadata = Base.metadata")
    print("       # Make sure your models are imported so Base.metadata is populated.")
    print("       ```")
    print("     - Ensure the database URL is configured for Alembic, often by reading from `alembic.ini` or environment variables.")
    print("       The default `env.py` often includes logic to get `sqlalchemy.url` from `alembic.ini`.")
    print("  4. Configure `alembic.ini`:")
    print("     - Set `sqlalchemy.url` to your database connection string.")
    print("       `sqlalchemy.url = postgresql+psycopg://user:pass@host/dbname`")

    print("\nd) Generating Migrations (Conceptual Commands - run in terminal):")
    print("  - Autogenerate migrations (Alembic compares models to DB and generates changes):")
    print("    `alembic revision -m \"create_users_table\" --autogenerate`")
    print("    `alembic revision -m \"add_email_to_users\" --autogenerate`")
    print("    - Review the generated script in the `alembic/versions/` directory carefully!")
    print("    - Autogenerate might not catch all changes (e.g., server defaults, complex constraints without names).")
    print("  - Manual migrations (for changes autogenerate misses or for data migrations):")
    print("    `alembic revision -m \"add_custom_function\"`")
    print("    - Then manually edit the generated script using Alembic's `op` functions (e.g., `op.create_table()`, `op.add_column()`, `op.execute('SQL SCRIPT')`).")

    print("\ne) Applying and Managing Migrations (Conceptual Commands - run in terminal):")
    print("  - Apply all pending migrations: `alembic upgrade head`")
    print("  - Apply migrations up to a specific revision: `alembic upgrade <revision_id>`")
    print("  - Downgrade (revert) migrations:")
    print("    `alembic downgrade -1` (revert one migration)")
    print("    `alembic downgrade <target_revision_id>`")
    print("  - Show migration history: `alembic history`")
    print("  - Show current revision: `alembic current`")
    print("  - Show details of a revision: `alembic show <revision_id>`")

    print("\nf) Branching and Merging (Conceptual):")
    print("  - Useful when multiple developers work on schema changes in parallel branches.")
    print("  - Alembic supports creating named branches in migration history.")
    print("  - `alembic merge <revision1> <revision2> -m \"merge feature_x and feature_y\"`")
    print("  - Requires careful management to avoid conflicts.")

    print("\nIMPORTANT: Always backup your production database before applying migrations.")
    print("Test migrations thoroughly in a staging environment.")


if __name__ == "__main__":
    print("Starting Module D2: Productionizing Databases")
    engine_url = ""
    engine = None
    try:
        engine_url = get_db_url()
        engine = create_engine(engine_url, echo=False)

        # For demos requiring tables, ensure they exist
        Base.metadata.drop_all(bind=engine) # Clean slate for demo
        Base.metadata.create_all(bind=engine) # Creates SimpleUser table

        database_administration_concepts()
        performance_and_scalability_discussion(engine)
        security_considerations(engine_url)
        database_migrations_with_alembic()

    except ValueError as e:
        print(f"Configuration error: {e}")
    except SQLAlchemyError as e:
        print(f"Database operational error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        if engine:
            # Clean up SimpleUser table after demo
            # Base.metadata.drop_all(bind=engine)
            engine.dispose()

    print("\nModule D2 execution finished.")
    print("Review 'D2_production_databases_admin_migrations.py' for details.")
