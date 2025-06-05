import os
import psycopg
from dotenv import load_dotenv

def verify_connection():
    load_dotenv() # Load variables from .env file

    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")

    if not all([db_host, db_port, db_name, db_user, db_password]):
        print("One or more database connection environment variables are not set.")
        print("Please check your .env file and ensure it's being loaded.")
        return

    conn_string = f"host='{db_host}' port='{db_port}' dbname='{db_name}' user='{db_user}' password='{db_password}'"

    try:
        # Attempt to connect to the database
        with psycopg.connect(conn_string) as conn:
            # Create a cursor object
            with conn.cursor() as cur:
                # Execute a simple query to get the PostgreSQL version
                cur.execute("SELECT version();")
                db_version = cur.fetchone()
                if db_version:
                    print("Successfully connected to PostgreSQL!")
                    print(f"Database version: {db_version[0]}")
                else:
                    print("Could not retrieve database version.")
    except psycopg.OperationalError as e:
        print(f"Connection failed: {e}")
        print("Please check:")
        print("1. Is your PostgreSQL server running?")
        print("2. Are the connection details in your .env file correct?")
        print(f"   Host: {db_host}, Port: {db_port}, DB: {db_name}, User: {db_user}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    verify_connection()