# Module B2: PostgreSQL Fundamentals & psycopg3

# This script introduces PostgreSQL as a robust relational database and the
# `psycopg3` library for Python-PostgreSQL interaction. It covers connection setup,
# core PostgreSQL concepts (schemas, data types, keys), DDL (Data Definition Language),
# and DML (Data Manipulation Language) operations using `psycopg3`, building upon
# the SQL fundamentals learned in B1.

import os
import psycopg # psycopg3
from dotenv import load_dotenv

# --- Why PostgreSQL? ---
# PostgreSQL, often simply "Postgres," is a powerful, open-source object-relational
# database system with over 30 years of active development that has earned it a
# strong reputation for reliability, feature robustness, and performance.
# It is more robust and feature-rich than SQLite.
#
# Key Advantages:
# 1.  Reliability and Standards Compliance: Strong adherence to SQL standards.
# 2.  Extensibility: Users can define their own data types, index types,
#     functional languages, etc. This is how extensions like PostGIS (for
#     geospatial data) and pgvector (for vector embeddings) are possible.
# 3.  Data Types: Rich set of native data types, including JSON/JSONB, XML,
#     arrays, hstore (key-value store), and more. JSONB is particularly useful
#     for semi-structured data.
# 4.  Concurrency and Performance: Sophisticated Multi-Version Concurrency
#     Control (MVCC) for handling many concurrent users effectively.
# 5.  Advanced Features: Supports advanced SQL features like window functions,
#     Common Table Expressions (CTEs), full-text search, and has robust
#     transactional integrity (ACID compliance).
# 6.  Vibrant Community: Large, active community ensures ongoing development,
#     support, and a wealth of third-party tools and extensions.
# 7.  Open Source and Cost-Effective: No licensing costs, reducing total cost
#     of ownership.
#
# For our RAG LLM system, PostgreSQL's ability to handle structured relational
# data, semi-structured JSONB data, and vector embeddings (via pgvector)
# makes it a versatile and powerful choice.

# --- Core Relational Concepts (Recap) ---
# -   Schema: A collection of database objects, including tables, views, indexes,
#     sequences, data types, functions, and operators, associated with a
#     particular database. It's like a namespace within a database.
#     By default, objects are created in the 'public' schema.
# -   Table: A collection of related data entries organized in rows and columns.
#     E.g., an 'employees' table.
# -   Row (Record or Tuple): A single entry or record in a table.
#     E.g., one specific employee's data.
#   - Column (Attribute or Field): A named data element of a specific data type
#     that represents a particular attribute of the rows in a table.
#     E.g., 'employee_id', 'first_name', 'hire_date'.
#   - Data Types: Define the kind of data a column can hold (e.g., INTEGER,
#     VARCHAR(255), TEXT, BOOLEAN, DATE, TIMESTAMP, NUMERIC, JSONB, etc.).
#     PostgreSQL has a rich set of data types. (Recent additions are usually
#     incremental improvements or new extensions rather than fundamental new types).

# --- Keys ---
# Keys are one or more columns used to identify or relate rows in tables.
# -   Primary Key (PK): A column (or set of columns) that uniquely identifies
#     each row in a table. Values must be unique and non-null.
#     A table can have only one primary key.
# -   Foreign Key (FK): A column (or set of columns) in one table that references
#     the primary key of another table. It establishes a link between tables
#     and enforces referential integrity (ensuring relationships are valid).
# -   Unique Constraint (or Unique Key): Ensures that all values in a column (or
#     set of columns) are unique. Unlike a PK, it can allow NULL values (though
#     typically only one NULL is permitted, depending on the DB). A table can
#     have multiple unique constraints.
# -   Candidate Key: Any column or set of columns that can qualify as a primary
#     key (i.e., they are unique and non-null). The primary key is chosen from
#     the set of candidate keys.

# --- Indexes ---
# -   What they are: Special lookup tables that the database search engine can
#     use to speed up data retrieval. An index is a pointer to data in a table.
#     It's similar to an index in the back of a book.
# -   Why they are important: Significantly improve query performance (especially
#     for SELECT statements with WHERE clauses, JOIN operations, and ORDER BY).
#     However, they can slow down data modification operations (INSERT, UPDATE,
#     DELETE) because indexes also need to be updated.
# -   Common types: B-tree (default, good for range queries and equality),
#     Hash (good for equality), GiST, GIN (good for complex data types like
#     arrays, full-text search, JSONB), BRIN (good for very large tables where
#     values correlate with physical location). For pgvector, HNSW and IVFFlat
#     are common index types for approximate nearest neighbor search.

# --- Introduction to SQL (Structured Query Language) ---
# SQL is the standard language for managing and querying relational databases.
#
#   1. Data Definition Language (DDL): Used to define and modify database
#      structure (schemas, tables, indexes, etc.).
#      - `CREATE TABLE table_name (column1 datatype constraints, ...);`
#      - `ALTER TABLE table_name ADD COLUMN column_name datatype;`
#      - `ALTER TABLE table_name DROP COLUMN column_name;`
#      - `ALTER TABLE table_name RENAME TO new_table_name;`
#      - `DROP TABLE table_name;` (Deletes the table structure and all its data)
#      - `TRUNCATE TABLE table_name;` (Deletes all data from the table, but keeps structure. Faster than DELETE from.)
#
#   2. Data Manipulation Language (DML): Used to manage data within tables.
#      - `INSERT INTO table_name (column1, column2) VALUES (value1, value2);`
#      - `SELECT column1, column2 FROM table_name WHERE condition ORDER BY column1 LIMIT 10;`
#      - `UPDATE table_name SET column1 = new_value1 WHERE condition;`
#      - `DELETE FROM table_name WHERE condition;`
#
#   Basic `SELECT` queries:
#      - `SELECT * FROM table_name;` (Select all columns and rows)
#      - `SELECT column1, column2 FROM table_name;` (Select specific columns)
#      - `WHERE` clause: Filters rows based on a condition (e.g., `WHERE age > 30`).
#      - `ORDER BY` clause: Sorts the results (e.g., `ORDER BY last_name ASC` or `DESC`).
#      - `LIMIT` clause: Restricts the number of rows returned (e.g., `LIMIT 10`).
#      - `OFFSET` clause: Skips a number of rows before starting to return rows (e.g., `OFFSET 5`).

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

def manage_employees_table():
    """
    Demonstrates DDL and DML operations on an 'employees' table
    using psycopg3.
    """
    conn_string = get_db_connection_string()
    table_name = "employees" # Using a module-specific table name

    # `psycopg.connect()` establishes a new database session.
    # The `with` statement ensures the connection is properly closed.
    try:
        
        with psycopg.connect(conn_string) as conn:
            print("Successfully connected to PostgreSQL!")

            # --- Understanding Connections and Cursors ---
            # A `Connection` object (here, `conn`) represents the actual session
            # with your PostgreSQL database. It handles things like authentication,
            # network communication, and transaction management.
            #
            # A `Cursor` object (here, `cur`) is obtained from a connection.
            # It's the primary tool you use to:
            #   1. Execute SQL statements: Send commands like SELECT, INSERT,
            #      UPDATE, DELETE, CREATE TABLE, etc., to the database.
            #   2. Fetch results: Retrieve data returned by queries (e.g., rows
            #      from a SELECT statement).
            #   3. Manage query context: It keeps track of the current position
            #      within a result set when you're fetching rows.
            #
            # Think of the connection as the phone line to the database, and the
            # cursor as the specific conversation you're having over that line
            # to ask questions (execute SQL) and get answers (fetch results).
            #
            # The `with conn.cursor() as cur:` syntax is highly recommended because
            # it ensures that the cursor is automatically closed (releasing its
            # resources) when the block is exited, even if errors occur.
            with conn.cursor() as cur:

                # --- DDL: Data Definition Language ---
                print(f"\n--- DDL Operations on '{table_name}' table ---")

                # Drop the table if it exists (for a clean run each time)
                cur.execute(f"DROP TABLE IF EXISTS {table_name};")
                print(f"Dropped table '{table_name}' if it existed.")

                # CREATE TABLE
                create_table_sql = f"""
                CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    role VARCHAR(100),
                    hire_date DATE DEFAULT CURRENT_DATE,
                    salary NUMERIC(10, 2)
                );
                """
                cur.execute(create_table_sql)
                print(f"Successfully created table '{table_name}'.")

                # ALTER TABLE (Example: Add a new column)
                # cur.execute(f"ALTER TABLE {table_name} ADD COLUMN department VARCHAR(50);")
                # print(f"Altered table '{table_name}' to add 'department' column.")

                # --- DML: Data Manipulation Language ---
                print(f"\n--- DML Operations on '{table_name}' table ---")

                # INSERT - Parameter binding (%s placeholders) is crucial to prevent SQL injection.
                # psycopg3 handles the conversion of Python types to SQL types.
                print("\nInserting data...")
                employees_data = [
                    ("Alice Wonderland", "Software Engineer", "2023-01-15", 75000.00),
                    ("Bob The Builder", "Project Manager", "2022-07-01", 90000.00),
                    ("Charlie Brown", "QA Tester", "2023-05-20", 60000.00),
                    ("Diana Prince", "DevOps Engineer", None, 85000.00) # None for hire_date will use DEFAULT
                ]
                insert_sql = f"INSERT INTO {table_name} (name, role, hire_date, salary) VALUES (%s, %s, %s, %s) RETURNING id;"

                inserted_ids = []
                for emp in employees_data:
                    cur.execute(insert_sql, emp) # Executes the INSERT, makes 'RETURNING id' available
                    employee_id = cur.fetchone()[0] # Fetches the single row/column from 'RETURNING id'
                    inserted_ids.append(employee_id)
                    print(f"Inserted employee: {emp[0]}, ID: {employee_id}")

                # SELECT - Fetching data
                print("\nFetching all employees (fetchall()):")
                # cur.execute() prepares the query; results are now available through 'cur'
                # Any previous result set on 'cur' from a prior .execute() is discarded.
                cur.execute(f"SELECT id, name, role, hire_date, salary FROM {table_name} ORDER BY name;")
                # cur.fetchall() retrieves all rows from the current result set into 'all_employees'
                all_employees = cur.fetchall() # Returns a list of tuples
                for emp in all_employees:
                    print(emp)

                print("\nFetching one employee (fetchone()):")
                # Using one of the IDs we captured
                if inserted_ids:
                    # This new cur.execute() discards the previous result set (all_employees' source)
                    # and prepares a new one for the specific employee.
                    cur.execute(f"SELECT id, name, role FROM {table_name} WHERE id = %s;", (inserted_ids[0],))
                    # cur.fetchone() retrieves the single row from this new result set.
                    one_employee = cur.fetchone() # Returns a single tuple or None
                    if one_employee:
                        print(one_employee)

                print("\nFetching N employees (fetchmany(N)):")
                # Again, this cur.execute() discards the previous (one_employee's source)
                # and prepares a new result set.
                cur.execute(f"SELECT id, name FROM {table_name} ORDER BY id;")
                # cur.fetchmany(2) retrieves the next 2 available rows from this new result set.
                some_employees = cur.fetchmany(2) # Fetch 2 rows
                for emp in some_employees:
                    print(emp)

                # UPDATE
                print("\nUpdating Bob's salary...")
                if inserted_ids:
                    # Find Bob's ID assuming he was the second one inserted (index 1)
                    # A more robust way would be to query by name.
                    bob_id_to_update = None
                    for i, emp_data in enumerate(employees_data):
                        if emp_data[0] == "Bob The Builder":
                            bob_id_to_update = inserted_ids[i]
                            break
                    
                    if bob_id_to_update:
                        update_sql = f"UPDATE {table_name} SET salary = %s WHERE id = %s;"
                        new_salary = 95000.00
                        cur.execute(update_sql, (new_salary, bob_id_to_update))
                        print(f"Updated Bob's (ID: {bob_id_to_update}) salary to {new_salary}. Rows affected: {cur.rowcount}")

                        # Verify update
                        cur.execute(f"SELECT name, salary FROM {table_name} WHERE id = %s;", (bob_id_to_update,))
                        print(f"Bob's new details: {cur.fetchone()}")
                    else:
                        print("Could not find Bob The Builder by assumed inserted ID to update.")


                # DELETE
                print("\nDeleting Charlie Brown...")
                charlie_id_to_delete = None
                for i, emp_data in enumerate(employees_data):
                    if emp_data[0] == "Charlie Brown":
                        charlie_id_to_delete = inserted_ids[i]
                        break
                
                if charlie_id_to_delete:
                    delete_sql = f"DELETE FROM {table_name} WHERE id = %s;"
                    cur.execute(delete_sql, (charlie_id_to_delete,))
                    print(f"Deleted Charlie Brown (ID: {charlie_id_to_delete}). Rows affected: {cur.rowcount}")

                    # Verify delete
                    cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE id = %s;", (charlie_id_to_delete,))
                    count = cur.fetchone()[0]
                    print(f"Count of Charlie Brown after delete: {count}")
                else:
                    print("Could not find Charlie Brown by assumed inserted ID to delete.")


                # --- Transaction Management ---
                # By default, psycopg3 operates in "autocommit" mode when you use
                # `with conn:` and `with cur:`. This means each `cur.execute()`
                # that modifies data might be committed immediately (especially DDL).
                #
                # For a sequence of DML operations that should be treated as a
                # single atomic unit (all succeed or all fail), you should use
                # explicit transaction blocks:
                #
                # with conn.transaction():
                #     cur.execute("UPDATE ...")
                #     cur.execute("INSERT ...")
                # # If the block exits normally, transaction is committed.
                # # If an exception occurs, transaction is rolled back.
                #
                # Or manually:
                # cur.execute("BEGIN;") # or conn.autocommit = False at connection
                # try:
                #    cur.execute(...)
                #    conn.commit()
                # except Exception:
                #    conn.rollback()
                #
                # Since we're doing DDL (CREATE/DROP TABLE), those are often
                # auto-committed or run in their own transaction outside explicit user control.
                # Our DML operations above are simple and executed one by one.
                # In a later module, we'll look at transactions in more detail.

            print(f"\nOperations on '{table_name}' table complete.")
            # The `with conn:` block automatically commits changes upon successful exit
            # if autocommit is true (default for simple `with conn:`) or if a
            # `conn.transaction()` block is used and completes successfully.
            # DDL statements like CREATE TABLE are typically auto-committed.

    except psycopg.OperationalError as e:
        print(f"Database connection failed: {e}")
        print("Please check:")
        print("1. Is your PostgreSQL server running?")
        print("2. Are the connection details in your .env file correct?")
    except psycopg.Error as e: # Catch other psycopg errors
        print(f"A database error occurred: {e}")
        # If an error occurred within the transaction, psycopg3's `with conn:`
        # would typically prevent a commit if autocommit was off,
        # or a `with conn.transaction():` block would trigger a rollback.
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    print("Starting Module B2: PostgreSQL Fundamentals with psycopg3")
    manage_employees_table()
    print("\nModule B2 execution finished.")
    print("Review the script 'B2_postgresql_intro_psycopg3.py' for detailed explanations and code.")
    print(f"Consider keeping the table '{'employees'}' for subsequent modules or drop it manually if desired.")
