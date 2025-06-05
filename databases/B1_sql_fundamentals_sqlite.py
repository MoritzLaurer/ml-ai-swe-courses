# Module B1: SQL Language Fundamentals with SQLite

# This script introduces fundamental SQL commands (DDL, DML, DQL) using Python's
# built-in `sqlite3` module. It covers table creation, data insertion, querying,
# joins, aggregations, and basic transaction control with SQLite, providing a
# foundation before moving to PostgreSQL.

import sqlite3
import os # For managing the SQLite file path

# --- Introduction to SQLite ---
# SQLite is a C-language library that implements a small, fast, self-contained,
# high-reliability, full-featured, SQL database engine.
# - Serverless: SQLite does not have a separate server process. It reads and
#   writes directly to ordinary disk files.
# - File-based: A complete SQL database with multiple tables, indices, triggers,
#   and views, is contained in a single disk file.
# - Python's `sqlite3` module: Python has built-in support for SQLite,
#   making it very easy to integrate SQL capabilities into Python applications
#   without external dependencies for the database driver itself.
# - Advantages for learning: Simplicity, no setup overhead, great for embedding.

# --- SQL Command Categories (Recap) ---
# DDL (Data Definition Language): Defines database structure (CREATE, ALTER, DROP)
# DML (Data Manipulation Language): Manages data within tables (INSERT, UPDATE, DELETE)
# DQL (Data Query Language): Retrieves data (SELECT)
# TCL (Transaction Control Language): Manages transactions (COMMIT, ROLLBACK)
# (DCL - Data Control Language - is less relevant for typical SQLite usage as it's often single-user or managed at the file system level)

# --- Common SQL Data Types in SQLite ---
# SQLite uses a more general, dynamic typing system. A value stored in a column
# determines its data type, not the column's declared type (mostly).
# However, column type affinity is used to prefer a certain type.
# Common affinities:
# - TEXT: For strings. Can also store dates/times as ISO8601 strings.
# - INTEGER: For whole numbers. `INTEGER PRIMARY KEY AUTOINCREMENT` is common for auto-incrementing IDs.
# - REAL: For floating-point numbers.
# - NUMERIC: Can store values of any type. Tries to convert to INTEGER or REAL if possible.
# - BLOB (Binary Large Object): For storing data exactly as it was input (e.g., images, files).

def sql_fundamentals_sqlite_demo(db_name="sql_fundamentals.db"):
    """Demonstrates core SQL commands using Python's sqlite3 module."""

    # Connect to SQLite database. If db_name is ":memory:", it's an in-memory DB.
    # Otherwise, it's a file. The file will be created if it doesn't exist.
    conn = None # Initialize conn to None for robust finally block
    try:
        conn = sqlite3.connect(db_name)
        # To get results as dictionary-like objects (access columns by name)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        print(f"--- SQL Fundamentals with SQLite (using '{db_name}') ---")

        # Enable Foreign Key support (important for relational integrity)
        # This needs to be done for each connection if FKs are used.
        cur.execute("PRAGMA foreign_keys = ON;")

        # --- DDL: CREATE TABLE ---
        print("\n--- DDL: Creating Tables ---")
        cur.execute("DROP TABLE IF EXISTS products_sqlite_demo;")
        cur.execute("DROP TABLE IF EXISTS categories_sqlite_demo;")

        create_categories_table_sql = """
        CREATE TABLE categories_sqlite_demo (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        );
        """
        cur.execute(create_categories_table_sql)
        print("Created 'categories_sqlite_demo' table.")

        create_products_table_sql = """
        CREATE TABLE products_sqlite_demo (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            category_id INTEGER,
            price REAL NOT NULL CHECK (price >= 0),
            stock_quantity INTEGER DEFAULT 0,
            release_date TEXT, -- Store dates as TEXT (e.g., 'YYYY-MM-DD')
            FOREIGN KEY (category_id) REFERENCES categories_sqlite_demo(category_id) ON DELETE SET NULL
        );
        """
        cur.execute(create_products_table_sql)
        print("Created 'products_sqlite_demo' table.")

        # DDL: CREATE INDEX (Example)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_product_name_sqlite ON products_sqlite_demo (product_name);")
        print("Created index 'idx_product_name_sqlite' on products_sqlite_demo(product_name).")
        conn.commit()

        # --- DML: INSERT INTO ---
        print("\n--- DML: Inserting Data ---")
        # Using '?' as placeholders
        categories_data = [("Electronics",), ("Books",), ("Home Goods",)]
        cur.executemany("INSERT INTO categories_sqlite_demo (category_name) VALUES (?);", categories_data)
        print(f"Inserted {cur.rowcount} categories.")

        # Get category_ids for products
        cur.execute("SELECT category_id, category_name FROM categories_sqlite_demo;")
        categories_map = {cat['category_name']: cat['category_id'] for cat in cur.fetchall()}

        products_data = [
            ("Laptop Pro X", categories_map["Electronics"], 1299.99, 50, "2024-01-15"),
            ("The SQLite Handbook", categories_map["Books"], 39.95, 120, "2023-11-01"),
            ("Smart Coffee Maker", categories_map["Home Goods"], 89.50, 75, "2024-03-10"),
            ("Wireless Mouse", categories_map["Electronics"], 25.00, 200, None),
            ("Gardening Tools Set", categories_map["Home Goods"], 45.75, 0, "2023-05-20")
        ]
        insert_product_sql = """
        INSERT INTO products_sqlite_demo (product_name, category_id, price, stock_quantity, release_date)
        VALUES (?, ?, ?, ?, ?);
        """
        cur.executemany(insert_product_sql, products_data)
        print(f"Inserted {cur.rowcount} products.")
        # Example: Get last inserted ID (only works reliably for single inserts or if AUTOINCREMENT is used)
        # cur.execute(insert_product_sql, ("Test Product", categories_map["Books"], 10.0, 1, "2024-01-01"))
        # last_id = cur.lastrowid
        # print(f"Last inserted product ID: {last_id}")
        conn.commit()


        # --- DQL: SELECT Statement ---
        print("\n--- DQL: SELECT Statements ---")
        print("\n1. Selecting specific products (Electronics, ordered by price, limit 2):")
        cur.execute("""
            SELECT product_id, product_name, price
            FROM products_sqlite_demo
            WHERE category_id = ?
            ORDER BY price DESC
            LIMIT 2;
        """, (categories_map["Electronics"],))
        for product in cur.fetchall():
            print(f"  ID: {product['product_id']}, Name: {product['product_name']}, Price: {product['price']}")

        # Filtering with LIKE (default case-insensitive for ASCII in SQLite)
        print("\n2. Filtering with LIKE (product name contains 'SQL' or 'Pro' - case insensitive for ASCII):")
        cur.execute("""
            SELECT product_name, price FROM products_sqlite_demo
            WHERE product_name LIKE ? OR product_name LIKE ?;
        """, ('%SQL%', '%Pro%'))
        for product in cur.fetchall():
            print(f"  Name: {product['product_name']}, Price: {product['price']}")

        # --- Joins ---
        print("\n--- Joins ---")
        print("\n1. INNER JOIN (products with their categories):")
        cur.execute("""
            SELECT p.product_name, c.category_name, p.price
            FROM products_sqlite_demo p
            INNER JOIN categories_sqlite_demo c ON p.category_id = c.category_id
            ORDER BY c.category_name, p.product_name;
        """)
        for row in cur.fetchall():
            print(f"  Product: {row['product_name']}, Category: {row['category_name']}, Price: {row['price']}")

        # --- Aggregate Functions ---
        print("\n--- Aggregate Functions ---")
        cur.execute("SELECT COUNT(*) AS total_products FROM products_sqlite_demo;")
        print(f"1. Total products (COUNT): {cur.fetchone()['total_products']}")

        # --- GROUP BY and HAVING ---
        print("\n--- GROUP BY and HAVING ---")
        print("\n1. Number of products and average price per category:")
        cur.execute("""
            SELECT c.category_name, COUNT(p.product_id) AS num_products, AVG(p.price) AS avg_price
            FROM categories_sqlite_demo c
            JOIN products_sqlite_demo p ON c.category_id = p.category_id
            GROUP BY c.category_name
            ORDER BY c.category_name;
        """)
        for row in cur.fetchall():
            avg_price_val = row['avg_price'] if row['avg_price'] is not None else 0.0
            print(f"  Category: {row['category_name']}, Products: {row['num_products']}, Avg Price: {avg_price_val:.2f}")

        # --- DML: UPDATE and DELETE ---
        print("\n--- DML: UPDATE and DELETE ---")
        print("\n1. Updating stock for 'Wireless Mouse':")
        update_sql = "UPDATE products_sqlite_demo SET stock_quantity = ? WHERE product_name = ?;"
        cur.execute(update_sql, (180, "Wireless Mouse"))
        conn.commit()
        print(f"  Updated stock. Rows affected: {cur.rowcount}")

        print("\n2. Deleting products that are out of stock (using `DELETE FROM`):")
        # In SQLite, TRUNCATE TABLE doesn't exist. Use DELETE FROM.
        delete_sql = "DELETE FROM products_sqlite_demo WHERE stock_quantity = 0;"
        cur.execute(delete_sql)
        conn.commit()
        print(f"  Deleted out-of-stock products. Rows affected: {cur.rowcount}")

        # --- Transaction Control ---
        # The `sqlite3` connection object can be used as a context manager.
        # If the `with` block completes successfully, changes are committed.
        # If an exception occurs, changes are rolled back.
        print("\n--- Transaction Control Example ---")
        try:
            with conn: # `conn` itself is the context manager for transactions
                print("  Attempting transaction...")
                # This will succeed
                conn.execute("INSERT INTO categories_sqlite_demo (category_name) VALUES (?);", ("Temporary Category",))
                print("  Inserted 'Temporary Category'.")
                # This would fail if uncommented due to UNIQUE constraint, causing a rollback of "Temporary Category"
                # conn.execute("INSERT INTO categories_sqlite_demo (category_name) VALUES (?);", ("Books",))
                print("  Transaction block successful (implicitly committed).")
        except sqlite3.IntegrityError as e:
            print(f"  Transaction block failed and was rolled back: {e}")
        except Exception as e:
            print(f"  An unexpected error in transaction block: {e}")


        # Verify 'Temporary Category' exists (or not, if the error path was taken)
        cur.execute("SELECT category_name FROM categories_sqlite_demo WHERE category_name = 'Temporary Category';")
        temp_cat = cur.fetchone()
        if temp_cat:
            print(f"  '{temp_cat['category_name']}' exists in the database.")
        else:
            print("  'Temporary Category' does not exist (likely rolled back).")


        # DDL: ALTER TABLE (Example: ADD COLUMN)
        print("\n--- DDL: ALTER TABLE (ADD COLUMN) ---")
        try:
            cur.execute("ALTER TABLE products_sqlite_demo ADD COLUMN last_checked TEXT;")
            conn.commit()
            print("Added 'last_checked' column to 'products_sqlite_demo'.")
        except sqlite3.OperationalError as e:
            print(f"Could not alter table (column might already exist): {e}")


    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}")
        if conn:
            conn.rollback() # Rollback any pending changes if an error occurs outside a 'with conn:' block
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print(f"\nSQLite connection to '{db_name}' closed.")
            # If it's a file-based DB and you want to clean up after demo:
            if db_name != ":memory:" and os.path.exists(db_name):
                # os.remove(db_name) # Uncomment to delete the .db file after run
                # print(f"Removed database file: {db_name}")
                pass


if __name__ == "__main__":
    print("Starting Module B1: SQL Language Fundamentals with SQLite")
    # You can use ":memory:" for an in-memory database that's gone when script ends,
    # or a filename like "my_sqlite_database.db" to persist it.
    db_file = "B1_sqlite_demo.db"
    sql_fundamentals_sqlite_demo(db_file)
    print("\nModule B1 execution finished.")
    print("Review the script 'B1_sql_fundamentals_sqlite.py' for details.")
    if os.path.exists(db_file):
        print(f"A database file '{db_file}' was created/used. You can inspect it with a SQLite browser or delete it if desired.")
