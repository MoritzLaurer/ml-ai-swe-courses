# Module C1: Deep Dive - Document Databases (MongoDB) for RAG

# This script is an educational exploration of MongoDB for RAG, demonstrating
# its potential use. It is not a definitive recommendation, as alternatives like
# PostgreSQL (with JSONB) are often well-suited for RAG. The optimal database
# choice is always project-specific.

import pymongo
from pymongo import MongoClient
from bson import ObjectId # For working with MongoDB's unique IDs
import json
import time
import datetime

# --- Introduction to MongoDB ---
# MongoDB is a popular NoSQL document-oriented database.
# Core Concepts:
#   - Documents: Data is stored in BSON (Binary JSON) documents, which are flexible,
#     JSON-like structures. Documents are analogous to rows in relational databases.
#     The maximum BSON document size is 16 megabytes.
#   - Collections: Documents are organized into collections. Collections are analogous
#     to tables in relational databases. A collection does not enforce a schema by
#     default; documents within a collection can have different fields.
#   - Databases: Collections are grouped within databases. A single MongoDB server
#     can host multiple databases.
#   - _id: Every document in a collection requires a unique `_id` field, which acts
#     as its primary key. If you don't provide one, MongoDB automatically generates
#     an ObjectId for `_id`.
#
# Python Driver: `pymongo` (ensure it's installed: `uv pip install pymongo`)

# --- MongoDB's Schema Flexibility ---
# Unlike SQL databases that require a strict, predefined schema, MongoDB offers
# schema flexibility. This means documents within a collection can have varying
# structures, fields, and data types. This aids rapid development and evolving
# data models.
#
# Managing Schema Evolution:
# While flexible, applications often have an implicit schema. Consistency and
# backward compatibility are typically handled through:
#   - Application Logic: Defensive code handles varied document structures.
#   - Document Versioning: A 'schema_version' field helps track and manage changes.
#   - Migration Strategies: Scripts for batch updates or on-the-fly data
#     transformations when read.
#   - MongoDB Schema Validation: Optional rules can be set on collections to
#     guide towards a desired structure for new/updated documents, without the
#     upfront rigidity of SQL.
# This approach contrasts with SQL's DDL (`ALTER TABLE`) for schema changes.
# ---

# --- JSON, BSON, and JSONB: A brief comparison ---
#
# JSON (JavaScript Object Notation):
#   - A lightweight, text-based data interchange format. Human-readable.
#   - Represents data as objects (key/value pairs) and arrays.
#   - Standard data types: strings, numbers, booleans, arrays, objects, null.
#   - Primarily a textual format for data exchange, not an optimized internal database storage format.
#
# BSON (Binary JSON):
#   - MongoDB's native binary data format. Not directly human-readable.
#   - Extends JSON with more data types (e.g., ObjectId, Date, Binary data, specific number types).
#   - Designed for efficient storage, fast encoding/decoding, and easy traversal within MongoDB.
#   - Includes type and length information for efficient processing.
#
# JSONB (PostgreSQL):
#   - PostgreSQL's binary representation of JSON data. Not directly human-readable.
#   - Stores standard JSON data types (dates often as ISO8601 strings).
#   - Optimized for efficient querying and indexing (e.g., GIN indexes) within PostgreSQL.
#   - Key order is not preserved, and duplicate object keys are removed (last one wins).
#
# Key Distinction:
#   - JSON is the textual standard.
#   - BSON and JSONB are specialized binary formats used internally by MongoDB and PostgreSQL,
#     respectively, for performance, efficiency, and (in BSON's case) extended type support.
#
# Example Sketch:
#   If JSON (text) is `{"date": "2024-01-01", "value": 100}`,
#   BSON might store 'date' as a native Date type and 'value' as a specific number type.
#   JSONB would store it in an optimized binary structure for fast lookups in PostgreSQL.

print("Starting Module C1: Deep Dive - Document Databases (MongoDB) for RAG")
print("=" * 70)
print("IMPORTANT: Ensure MongoDB is running (e.g., on localhost:27017).")
print("  `docker run -d -p 27017:27017 mongo` (if using Docker)")
print("  `uv add pymongo` (or `uv pip install pymongo`) (to install the driver)")
print("=" * 70)


def connect_to_mongodb(host="localhost", port=27017):
    """Establishes a connection to MongoDB."""
    try:
        client = MongoClient(f"mongodb://{host}:{port}/")
        client.admin.command('ping') # Verify connection
        print(f"Successfully connected to MongoDB at {host}:{port}.")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"MongoDB connection failed: {e}")
        print("Ensure MongoDB is running and accessible.")
        return None

def mongodb_rag_examples(client):
    """Demonstrates MongoDB operations relevant to RAG systems."""
    if not client:
        return

    print("\n--- MongoDB for RAG Examples ---")

    # Use a specific database and collection for RAG chat logs
    db = client["rag_system_module_c1"]
    chat_logs_collection = db["chat_logs_detailed"]

    # Clear previous demo data for idempotency
    chat_logs_collection.delete_many({})
    print(f"Cleared previous data from '{chat_logs_collection.name}' collection.")

    # --- CRUD Operations ---

    # 1. CREATE (Inserting Documents)
    print("\n1. CREATE: Inserting detailed chat logs...")
    log_1 = {
        "session_id": "sess_rag_001",
        "user_id": "user_alpha",
        "start_timestamp": datetime.datetime.now(datetime.timezone.utc),
        "messages": [
            {"sender": "user", "text": "What is retrieval augmented generation?", "timestamp": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=60)},
            {"sender": "bot", "text": "RAG combines pre-trained LLMs with external knowledge retrieval.", "timestamp": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=50), "retrieved_doc_ids": ["doc_abc", "doc_def"], "confidence": 0.92}
        ],
        "user_feedback": {"rating": 5, "tags": ["helpful", "clear"]},
        "llm_details": {"model_used": "gpt-4-turbo", "prompt_tokens": 150, "completion_tokens": 80},
        "schema_version": 1
    }
    result_1 = chat_logs_collection.insert_one(log_1)
    print(f"Inserted log 1 with _id: {result_1.inserted_id}")

    log_2 = {
        "session_id": "sess_rag_002",
        "user_id": "user_beta",
        "start_timestamp": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=10),
        "messages": [
            {"sender": "user", "text": "How does MongoDB handle schema?", "timestamp": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=9)},
            {"sender": "bot", "text": "MongoDB is schema-flexible, but schema validation is available.", "timestamp": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=8), "retrieved_doc_ids": ["mongo_doc_01"]},
            {"sender": "user", "text": "Thanks!", "timestamp": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=7)}
        ],
        "tags": ["mongodb", "schema"],
        "schema_version": 1
    }
    # `insert_many` can be used for multiple documents
    result_many = chat_logs_collection.insert_many([log_2,
        {
            "session_id": "sess_rag_001", # Same session as log_1, different interaction
            "user_id": "user_alpha",
            "start_timestamp": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5),
            "messages": [{"sender": "user", "text": "Follow-up question...", "timestamp": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=4)}],
            "schema_version": 1
        }
    ])
    print(f"Inserted {len(result_many.inserted_ids)} more logs.")


    # 2. READ (Querying Documents)
    print("\n2. READ: Querying chat logs...")

    # Find one document
    a_log = chat_logs_collection.find_one({"session_id": "sess_rag_001"})
    print("\nFound one log for session 'sess_rag_001':")
    if a_log:
        print(f"  _id: {a_log['_id']}, User: {a_log['user_id']}")

    # Find multiple documents
    print("\nLogs for user 'user_alpha':")
    user_alpha_logs = chat_logs_collection.find({"user_id": "user_alpha"})
    for log in user_alpha_logs:
        print(f"  Session ID: {log['session_id']}, Start: {log['start_timestamp']}")

    # Querying with filters (e.g., feedback rating)
    # The dot notation "user_feedback.rating" accesses nested fields.
    print("\nLogs with 5-star feedback rating:")
    high_feedback_logs = chat_logs_collection.find({"user_feedback.rating": 5})
    for log in high_feedback_logs:
        print(f"  Session ID: {log['session_id']}, Feedback: {log['user_feedback']}")

    # Projection (selecting specific fields)
    # 0 means exclude, 1 means include. _id is included by default unless excluded.
    print("\nSession IDs and LLM models used (projection):")
    projected_logs = chat_logs_collection.find(
        {"llm_details.model_used": {"$exists": True}}, # Filter for logs that have llm_details
        {"session_id": 1, "llm_details.model_used": 1, "_id": 0}
    )
    for log in projected_logs:
        print(f"  {log}")


    # 3. UPDATE (Modifying Documents)
    print("\n3. UPDATE: Modifying chat logs...")

    # Update one document: Add a 'resolved' tag to user_beta's session
    update_result_one = chat_logs_collection.update_one(
        {"session_id": "sess_rag_002"},
        {"$addToSet": {"tags": "resolved"}} # $addToSet adds item if not present
    )
    print(f"Updated one log (sess_rag_002): Matched {update_result_one.matched_count}, Modified {update_result_one.modified_count}")

    # Update multiple documents: Increment schema_version for all logs (example)
    # This is often part of a migration script.
    update_result_many = chat_logs_collection.update_many(
        {}, # Empty filter matches all documents
        {"$inc": {"schema_version": 1}}
    )
    print(f"Updated many logs: Matched {update_result_many.matched_count}, Modified {update_result_many.modified_count}")

    # Verify an update
    updated_log_beta = chat_logs_collection.find_one({"session_id": "sess_rag_002"})
    if updated_log_beta:
        print(f"  Log 'sess_rag_002' after update: Tags={updated_log_beta.get('tags')}, SchemaVer={updated_log_beta.get('schema_version')}")


    # 4. DELETE (Removing Documents)
    print("\n4. DELETE: Removing chat logs...")
    # For safety, let's create a temporary log to delete
    temp_log_id = chat_logs_collection.insert_one({"session_id": "temp_to_delete", "user_id":"temp_user"}).inserted_id
    print(f"Inserted temporary log with _id: {temp_log_id}")

    delete_result = chat_logs_collection.delete_one({"_id": temp_log_id})
    print(f"Deleted temporary log: {delete_result.deleted_count} document(s).")

    # delete_many({}) would delete all documents in the collection.


    # --- Indexing in MongoDB (Basics) ---
    print("\n--- Indexing ---")
    # Indexes improve query performance. MongoDB automatically creates an index on `_id`.
    # Create an index on `session_id` (if it doesn't exist)
    chat_logs_collection.create_index([("session_id", pymongo.ASCENDING)], unique=False) # Can be unique=True if session_id should be unique
    print("Ensured index exists on 'session_id'.")
    # Create a compound index
    chat_logs_collection.create_index([("user_id", pymongo.ASCENDING), ("start_timestamp", pymongo.DESCENDING)])
    print("Ensured compound index exists on 'user_id' and 'start_timestamp'.")
    # Index on a nested field
    chat_logs_collection.create_index([("user_feedback.rating", pymongo.ASCENDING)])
    print("Ensured index exists on 'user_feedback.rating'.")

    # List existing indexes
    print("Existing indexes on collection:")
    for index in chat_logs_collection.list_indexes():
        print(f"  {index['name']}: {index['key']}")


    # --- Schema Design and Evolution ---
    print("\n--- Schema Design and Evolution ---")
    # MongoDB is schema-flexible, but applications usually have an implicit schema.
    # Strategies:
    #   - Application-level validation: Check data structure in your code.
    #   - Document versioning: Add a `schema_version` field (as done above).
    #     Your application can then handle different versions accordingly.
    #   - Migration scripts: For larger changes, write scripts to update documents.
    #
    # MongoDB Schema Validation (Example - not enforced here, just conceptual)
    # You can define validation rules when creating a collection or add them later.
    # This example is for illustration; run `collMod` in mongo shell or specific driver commands.
    conceptual_schema_validation_rules = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["session_id", "user_id", "start_timestamp", "messages"],
            "properties": {
                "session_id": {"bsonType": "string", "description": "must be a string and is required"},
                "user_id": {"bsonType": "string", "description": "must be a string and is required"},
                "start_timestamp": {"bsonType": "date", "description": "must be a date and is required"},
                "messages": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "object",
                        "required": ["sender", "text", "timestamp"],
                        "properties": {
                            "sender": {"bsonType": "string"},
                            "text": {"bsonType": "string"},
                            "timestamp": {"bsonType": "date"}
                        }
                    }
                },
                "user_feedback": {
                    "bsonType": "object",
                    "properties": {
                        "rating": {"bsonType": "int", "minimum": 1, "maximum": 5},
                        "tags": {"bsonType": "array", "items": {"bsonType": "string"}}
                    }
                },
                "schema_version": {"bsonType": "int"}
            }
        }
    }
    # To apply in mongo shell (example):
    # db.runCommand({
    #   collMod: "chat_logs_detailed",
    #   validator: <conceptual_schema_validation_rules>,
    #   validationLevel: "moderate" // Or "strict"
    # })
    print("Conceptual schema validation rules defined (see comments).")
    print("Schema validation helps enforce structure for new/updated documents.")


    # --- Specific RAG Use Cases ---
    print("\n--- Specific RAG Use Cases for MongoDB ---")
    print("1. Storing detailed, evolving chat logs:")
    print("   - Rich metadata per message (confidence, retrieved docs).")
    print("   - Flexible user feedback structures.")
    print("   - LLM parameters, diagnostics can be easily added/changed.")
    print("2. Managing highly variable source document metadata:")
    print("   - If source documents for RAG have vastly different metadata schemas,")
    print("     MongoDB can store this metadata alongside document IDs without")
    print("     requiring a unified relational schema for all metadata types.")

    # Clean up (optional: drop the test database)
    # client.drop_database("rag_system_module_c1")
    # print("\nDropped test database 'rag_system_module_c1'.")


def main():

    mongo_client = connect_to_mongodb()
    if mongo_client:
        mongodb_rag_examples(mongo_client)
        mongo_client.close()
        print("\nMongoDB connection closed.")

    print("\nModule C1 execution finished.")
    print("Review the script 'C1_mongodb_for_rag.py' for details.")

if __name__ == "__main__":
    main()