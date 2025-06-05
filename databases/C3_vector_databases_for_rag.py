# Module C3: Deep Dive - Standalone Vector Databases for RAG (with ChromaDB)

# This script introduces standalone vector databases as an alternative or complement
# to solutions like PostgreSQL's pgvector extension, focusing on ChromaDB for a practical, local-first demonstration.
# We'll cover when to consider dedicated vector DBs, their core concepts, and how to
# use ChromaDB for storing, indexing, and querying embeddings with metadata.

# --- When to Consider Dedicated Vector Databases Over pgvector ---
# While pgvector is a great extension that integrates vector search directly
# into PostgreSQL, there are scenarios where a dedicated vector database might be
# more suitable or offer advantages:
#
# 1.  Extreme Scale & Performance:
#     -   Massive Datasets: Handling tens of millions to billions of embeddings.
#         Dedicated solutions are often optimized for sharding and distributed
#         architectures from the ground up.
#     -   Ultra-Low Latency: When p99 latencies for vector search need to be
#         consistently in the sub-millisecond range for very high throughput.
#     -   Specialized Hardware: Some dedicated vector DBs might have better
#         integrations or optimizations for specific hardware accelerators (though
#         this is an evolving space).
#
# 2.  Advanced Vector Search Features & Algorithms:
#     -   Cutting-Edge Indexing: May offer more varieties or more mature/tuned
#         implementations of advanced indexing algorithms (e.g., specific types of
#         HNSW, IVFADC, or proprietary methods).
#     -   Advanced Filtering Capabilities: Sophisticated pre-filtering or post-filtering
#         logic that might be more expressive or performant than what's easily
#         achievable with pgvector in complex scenarios.
#     -   Quantization & Memory Optimization: More built-in options for vector
#         quantization (reducing vector size for memory efficiency) which can be
#         critical at scale.
#
# 3.  Decoupled Architecture & Operational Simplicity (for the vector workload):
#     -   Microservice Approach: You may want to isolate the vector search
#         functionality as a separate service for independent scaling, updates,
#         and resource management.
#     -   Simplified Vector DB Operations: If the team managing the vector search
#         is different from the team managing the primary relational database,
#         a dedicated solution might offer them a simpler operational model focused
#         solely on vector data.
#
# 4.  Feature Velocity:
#     -   Dedicated vector databases are singularly focused on vector search and
#         may incorporate new research and features in this domain more rapidly.
#
# --- Downsides of Dedicated Vector Databases (vs. pgvector) ---
# -   Increased System Complexity: You now have another database system to deploy,
#     manage, monitor, backup, and secure.
# -   Data Synchronization: Keeping metadata (often stored in a primary DB like
#     PostgreSQL) in sync with vectors in the dedicated vector DB can be complex.
#     This usually requires ETL pipelines or event-driven updates.
# -   Transactional Integrity: ACID transactions typically won't span across your
#     primary DB and the vector DB.
# -   Joins & Relational Operations: You lose the ability to easily JOIN vector
#     search results with other relational data in a single query, a key strength
#     of pgvector. Complex queries might require multiple steps and application-side joins.
# -   Cost: Managed services for dedicated vector DBs can add to expenses, while
#     self-hosting requires operational expertise.

# --- Core Concepts in Vector Databases ---
# -   Embeddings (Vectors): Numerical representations of data (text, images, audio)
#     in a high-dimensional space, where similarity in meaning or content
#     corresponds to proximity in the vector space.
# -   Collections (or Indexes in some DBs): A named group of embeddings, often
#     with associated metadata. Analogous to a table in SQL or a collection in MongoDB.
# -   Indexing Algorithms:
#     -   Purpose: To speed up similarity search. Searching through billions of
#         vectors naively (exact search) is computationally expensive.
#     -   Approximate Nearest Neighbor (ANN): Most vector databases use ANN
#         algorithms (e.g., HNSW, IVFADC, Product Quantization) to find "close enough"
#         matches quickly, trading a small amount of accuracy for significant speed gains.
#     -   ChromaDB uses HNSW (Hierarchical Navigable Small World) by default, which is a
#         popular and effective graph-based ANN algorithm.
# -   Similarity Metrics: How "closeness" between vectors is measured. Common ones:
#     -   Cosine Similarity (measures the cosine of the angle between two vectors;
#         good for orientation, common for text embeddings). Ranges from -1 to 1.
#     -   Euclidean Distance (L2 distance; straight-line distance between two points).
#         Ranges from 0 to infinity.
#     -   Dot Product (Inner Product).
#     ChromaDB defaults to `l2` (Euclidean distance) but also supports `cosine` and `ip` (inner product).
# -   Metadata Filtering: The ability to filter search results based on metadata
#     associated with the embeddings (e.g., find similar documents created after a
#     certain date, or with a specific tag). This is crucial for practical RAG.

# --- Introduction to ChromaDB ---
# Chroma is an open-source embedding database designed to make it easy to build
# LLM apps by making knowledge, facts, and skills pluggable for LLMs.
# Key Features:
#   - Simple API: Python-native and easy to get started with.
#   - Local-First: Can run in-memory or persist to disk with minimal setup.
#     Also supports a client-server model for more robust deployments.
#   - Automatic Embedding (Optional): Can integrate with embedding functions.
#     (We will provide embeddings manually for clarity in this module).
#   - Metadata Storage & Filtering: Strong support for attaching and filtering by metadata.
#
# Installation: `uv add chromadb` (or `uv pip install chromadb`)

import chromadb
import random # For generating dummy embeddings



# --- ChromaDB Client and Collection Setup ---

def setup_chroma_client(mode="persistent", path="./C3_chroma_db_example"):
    """
    Sets up a ChromaDB client.
    :param mode: "in-memory" or "persistent".
    :param path: Path for persistent storage (if mode is "persistent").
    :return: ChromaDB client instance.
    """
    if mode == "in-memory":
        print("Setting up ChromaDB client (in-memory)...")
        client = chromadb.Client() # Ephemeral client
    elif mode == "persistent":
        print(f"Setting up ChromaDB client (persistent at '{path}')...")
        client = chromadb.PersistentClient(path=path)
    else:
        raise ValueError("Invalid mode. Choose 'in-memory' or 'persistent'.")
    return client

def rag_with_chromadb_demo():
    """Demonstrates core ChromaDB functionalities for RAG."""

    print("\n--- ChromaDB RAG Demo ---")
    # For this demo, we'll use a persistent client so data can be inspected
    # after the script runs. Re-running will use existing data.
    # To start fresh, delete the "./C3_chroma_db_example" directory.
    client = setup_chroma_client(mode="persistent", path="./C3_chroma_db_example")

    # --- Collections in ChromaDB ---
    # A collection is where you store your embeddings, documents, and metadata.
    # You can specify the distance metric when creating a collection.
    # Common distance functions: "l2" (default), "cosine", "ip" (inner product)
    # For text embeddings from models like Sentence Transformers or OpenAI, "cosine"
    # is often a good choice, or "l2" if embeddings are normalized.
    # Let's use 'cosine' for this example.
    collection_name = "rag_documents_c3"
    embedding_dimension = 768 # Example dimension (e.g., from a sentence-transformer model)

    try:
        print(f"\nCreating or getting collection: '{collection_name}' with cosine distance...")
        # get_or_create_collection is idempotent
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Specifies the distance function for HNSW
        )
        print(f"Collection '{collection.name}' ready. Current count: {collection.count()} items.")
    except Exception as e:
        print(f"Error creating/getting collection: {e}")
        return

    # --- Adding Data to a Collection ---
    # You need:
    #   - embeddings (list of lists of floats)
    #   - documents (list of strings, optional but recommended for RAG)
    #   - metadatas (list of dictionaries, optional)
    #   - ids (list of strings, unique identifiers for each item)
    print("\n--- Adding Items to Collection ---")

    # Let's prepare some sample data (as if from our RAG pipeline)
    # In a real RAG system, embeddings would come from an embedding model.
    # Documents would be the text chunks.
    sample_items = [
        {
            "id": "doc_chunk_001",
            "text": "ChromaDB is a vector database for AI applications.",
            "embedding": [random.uniform(-0.1, 0.1) for _ in range(embedding_dimension)],
            "metadata": {"source": "Chroma Docs", "category": "database", "year": 2023, "is_public": True}
        },
        {
            "id": "doc_chunk_002",
            "text": "It supports metadata filtering and various distance metrics.",
            "embedding": [random.uniform(-0.1, 0.1) for _ in range(embedding_dimension)],
            "metadata": {"source": "Chroma Blog", "category": "feature", "year": 2024, "is_public": True}
        },
        {
            "id": "doc_chunk_003",
            "text": "pgvector integrates vector search into PostgreSQL.",
            "embedding": [random.uniform(0.8, 1.0) for _ in range(embedding_dimension)], # Intentionally different
            "metadata": {"source": "Postgresql Docs", "category": "database", "year": 2022, "is_public": True}
        },
        {
            "id": "doc_chunk_004",
            "text": "RAG systems combine retrieval with generative models.",
            "embedding": [random.uniform(-0.2, 0.0) for _ in range(embedding_dimension)],
            "metadata": {"source": "AI Research Paper", "category": "concept", "year": 2023, "is_public": False}
        }
    ]

    # Prepare lists for ChromaDB's add method
    # We'll only add if the collection is empty for this demo to avoid duplicates on re-runs.
    if collection.count() == 0:
        print("Collection is empty. Adding sample items...")
        item_ids = [item["id"] for item in sample_items]
        item_embeddings = [item["embedding"] for item in sample_items]
        item_documents = [item["text"] for item in sample_items]
        item_metadatas = [item["metadata"] for item in sample_items]

        try:
            collection.add(
                ids=item_ids,
                embeddings=item_embeddings,
                documents=item_documents,
                metadatas=item_metadatas
            )
            print(f"Successfully added {len(item_ids)} items to the collection.")
        except Exception as e:
            print(f"Error adding items: {e}")
    else:
        print(f"Collection already has {collection.count()} items. Skipping add operation for this demo.")


    # --- Querying (Similarity Search) ---
    print("\n--- Querying for Similar Items ---")
    # To query, you provide one or more query_embeddings.
    # query_embeddings: The vector(s) you want to find matches for.
    # n_results: The number of similar items to return.

    # Let's generate a query vector similar to our first item
    query_embedding_similar_to_001 = [v + random.uniform(-0.01, 0.01) for v in sample_items[0]["embedding"]]

    try:
        query_results = collection.query(
            query_embeddings=[query_embedding_similar_to_001],
            n_results=2 # Get top 2 results
            # include=['documents', 'metadatas', 'distances'] # Specify what to return
        )

        print("\nQuery results for a vector similar to 'doc_chunk_001':")
        if query_results:
            ids = query_results.get('ids', [[]])[0]
            documents = query_results.get('documents', [[]])[0]
            metadatas = query_results.get('metadatas', [[]])[0]
            distances = query_results.get('distances', [[]])[0]

            for i in range(len(ids)):
                print(f"  ID: {ids[i]}, Distance: {distances[i]:.4f}")
                print(f"    Doc: {documents[i]}")
                print(f"    Meta: {metadatas[i]}")
    except Exception as e:
        print(f"Error during query: {e}")


    # --- Querying with Metadata Filters ---
    print("\n--- Querying with Metadata Filters ---")
    # ChromaDB supports filtering using a MongoDB-like query syntax for metadata.
    # Example: Find documents similar to query_embedding_similar_to_001,
    # but only those from "Chroma Docs" or "Chroma Blog" AND with year 2023 or later.

    # Filter for items where category is "database"
    print("\nQuerying for 'database' category items similar to 'doc_chunk_001' vector:")
    try:
        filtered_results_category = collection.query(
            query_embeddings=[query_embedding_similar_to_001],
            n_results=3,
            where={"category": "database"} # Simple equality filter
            # include=['documents', 'metadatas', 'distances']
        )
        print("Results (category='database'):")
        if filtered_results_category and filtered_results_category.get('ids', [[]])[0]:
            for i in range(len(filtered_results_category['ids'][0])):
                print(f"  ID: {filtered_results_category['ids'][0][i]}, Doc: {filtered_results_category['documents'][0][i][:40]}..., Meta: {filtered_results_category['metadatas'][0][i]}")
        else:
            print("  No results found with this filter.")
    except Exception as e:
        print(f"Error during filtered query (category): {e}")

    # Filter for items with year >= 2023
    print("\nQuerying for items with year >= 2023, similar to 'doc_chunk_001' vector:")
    try:
        filtered_results_year = collection.query(
            query_embeddings=[query_embedding_similar_to_001],
            n_results=3,
            where={"year": {"$gte": 2023}} # Using $gte (greater than or equal)
            # include=['documents', 'metadatas', 'distances']
        )
        print("Results (year >= 2023):")
        if filtered_results_year and filtered_results_year.get('ids', [[]])[0]:
            for i in range(len(filtered_results_year['ids'][0])):
                print(f"  ID: {filtered_results_year['ids'][0][i]}, Doc: {filtered_results_year['documents'][0][i][:40]}..., Meta: {filtered_results_year['metadatas'][0][i]}")
        else:
            print("  No results found with this filter.")
    except Exception as e:
        print(f"Error during filtered query (year): {e}")

    # Complex filter: source is "AI Research Paper" AND is_public is False
    print("\nQuerying for items from 'AI Research Paper' AND is_public is False:")
    try:
        # For this type of query, the query_embeddings are still used for similarity ranking,
        # but the pool of candidates is first reduced by the 'where' clause.
        # Let's use a generic query vector for this example.
        generic_query_vector = [random.uniform(-0.5, 0.5) for _ in range(embedding_dimension)]
        filtered_results_complex = collection.query(
            query_embeddings=[generic_query_vector],
            n_results=2,
            where={
                "$and": [
                    {"source": "AI Research Paper"},
                    {"is_public": False}
                ]
            }
            # include=['documents', 'metadatas', 'distances']
        )
        print("Results (complex filter: 'AI Research Paper' AND not public):")
        if filtered_results_complex and filtered_results_complex.get('ids', [[]])[0]:
            for i in range(len(filtered_results_complex['ids'][0])):
                 print(f"  ID: {filtered_results_complex['ids'][0][i]}, Doc: {filtered_results_complex['documents'][0][i][:40]}..., Meta: {filtered_results_complex['metadatas'][0][i]}")
        else:
            print("  No results found with this complex filter.")
    except Exception as e:
        print(f"Error during filtered query (complex): {e}")


    # --- Other Operations (Briefly) ---
    # - Get items by ID: `collection.get(ids=["doc_chunk_001", "doc_chunk_002"])`
    # - Update items: `collection.update(...)` (can update embeddings, documents, metadatas)
    # - Upsert items: `collection.upsert(...)` (adds if ID doesn't exist, updates if it does)
    # - Delete items: `collection.delete(ids=["doc_chunk_004"])` or by `where` filter.
    # - Count items: `collection.count()`
    # - Modify collection (e.g., name, metadata): `collection.modify(...)`

    print("\n--- Getting a specific item by ID ---")
    try:
        item_data = collection.get(ids=["doc_chunk_002"], include=['documents', 'metadatas'])
        if item_data and item_data['ids']:
            print(f"Data for 'doc_chunk_002':")
            print(f"  ID: {item_data['ids'][0]}")
            print(f"  Document: {item_data['documents'][0]}")
            print(f"  Metadata: {item_data['metadatas'][0]}")
        else:
            print("Item 'doc_chunk_002' not found.")
    except Exception as e:
        print(f"Error getting item by ID: {e}")

    # --- Clean up (Optional) ---
    # If you want to delete the collection or reset the client:
    # client.delete_collection(name=collection_name)
    # print(f"\nDeleted collection: '{collection_name}'")
    # client.reset() # Resets the entire database (deletes all collections for this client/path)
    # print("ChromaDB client has been reset (all data deleted for this path).")

    print("\nChromaDB demo finished.")


# --- Architectural Considerations and Pros/Cons (Recap) ---
# Using a dedicated vector DB like ChromaDB:
# Pros:
#   - Simplicity for vector-focused tasks.
#   - Can be very fast and efficient for pure vector search.
#   - Good for decoupling if vector search is a distinct service.
# Cons (vs. integrated solutions like pgvector):
#   - Data Synchronization: Keeping metadata in Chroma aligned with a primary
#     database (e.g., PostgreSQL containing full document details) requires effort.
#   - Transaction Management: No distributed transactions across Chroma and other DBs.
#   - Relational Capabilities: Lacks the rich relational querying and JOINs of SQL DBs.
#
# --- Other Standalone Vector Database Options ---
# -   Qdrant: High-performance, written in Rust, rich filtering, good for production.
# -   Weaviate: GraphQL API, modular, supports various media types.
# -   Milvus: Highly scalable, designed for massive datasets.
# -   Pinecone: Popular managed vector database service.
# Each has its own strengths, API, and deployment model. ChromaDB is excellent
# for getting started quickly and for many local or smaller-scale deployments.

if __name__ == "__main__":
    print("Starting Module C3: Deep Dive - Standalone Vector Databases (ChromaDB)")
    try:
        rag_with_chromadb_demo()
    except ImportError:
        print("ChromaDB library not found. Please install it: `uv pip install chromadb`")
    except Exception as e:
        print(f"An unexpected error occurred in the demo: {e}")
        import traceback
        traceback.print_exc()

    print("\nModule C3 execution finished.")
    print("Review the script 'C3_vector_databases_for_rag.py' for details.")
    print("If you used a persistent ChromaDB client, data is stored in './C3_chroma_db_example'.")
    print("Delete this directory to start fresh on the next run.")
