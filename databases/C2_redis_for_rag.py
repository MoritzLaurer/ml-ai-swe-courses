# Module C2: Deep Dive - Key-Value Stores (Redis) for RAG

# This script explores Redis, an in-memory data structure store, often used as a
# database, cache, and message broker. We'll cover its core concepts, common
# data types, and demonstrate its utility in a RAG (Retrieval Augmented Generation)
# system context, particularly for caching, session management, and rate limiting.

import redis
import time
import json # For serializing complex data structures like dictionaries

# --- Introduction to Redis ---
# Redis (Remote Dictionary Server) is an open-source, in-memory data structure store.
# Key Characteristics:
#   - In-Memory: Primarily stores data in RAM, leading to very fast read and write operations.
#   - Data Structures: Supports various data structures like strings, hashes, lists,
#     sets, sorted sets, streams, HyperLogLogs, bitmaps, and geospatial indexes.
#   - Versatility: Can be used as a high-performance cache, database, message broker,
#     and for real-time analytics (e.g., leaderboards, counters).
#   - Persistence: Offers options to persist data to disk, though its primary mode is in-memory.
#   - Atomic Operations: Many Redis operations are atomic, ensuring data integrity.
#   - Scalability: Supports replication (master-slave) and clustering for scalability and high availability.
#
# Python Driver: `redis-py` (ensure it's installed: `uv pip install redis`)

# --- Redis-py Client: Main Classes and Methods Used ---
# `redis-py`: Python's interface to Redis.
#
# Key Class:
#   - `redis.Redis(host, port, db, decode_responses)`: The primary class to create a
#     connection client to a Redis server.
#
# Common Method Categories (examples):
#   - Connection: `ping()` (check connection).
#   - Strings: `set(key, value)`, `get(key)`, `incr(key)` (for counters).
#   - Hashes (objects): `hset(hash_key, field, value)`, `hget(hash_key, field)`, `hgetall(hash_key)`.
#   - Lists (ordered): `rpush(list_key, value)`, `lpop(list_key)`.
#   - Sets (unique, unordered): `sadd(set_key, value)`, `smembers(set_key)`.
#   - Sorted Sets (unique, ordered by score): `zadd(zset_key, {member: score})`, `zrevrange(zset_key, start, end)`.
#   - Key Management: `expire(key, seconds)`, `ttl(key)`, `delete(key)`.


print("Starting Module C2: Deep Dive - Key-Value Stores (Redis) for RAG")
print("=" * 70)
print("IMPORTANT: Ensure Redis is running (e.g., on localhost:6379).")
print("  `docker run -d -p 6379:6379 redis` (if using Docker)")
print("  `uv add redis` (or `uv pip install redis`) (to install the Python driver)")
print("=" * 70)

def connect_to_redis(host="localhost", port=6379, db=0):
    """Establishes a connection to Redis."""
    try:
        # `decode_responses=True` makes redis-py return Python strings instead of bytes.
        r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        r.ping()  # Verify connection
        print(f"Successfully connected to Redis at {host}:{port}, db={db}.")
        return r
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection failed: {e}")
        print("Ensure Redis server is running and accessible.")
        return None

def redis_rag_examples(r):
    """Demonstrates Redis operations relevant to RAG systems."""
    if not r:
        return

    print("\n--- Redis for RAG Examples ---")

    # For RAG, Redis keys might be prefixed to organize data, e.g., "llm_cache:", "session:"
    # Clear previous demo data (optional, for clean runs)
    # Be careful with FLUSHDB in a real environment! It clears the current database.
    # r.flushdb()
    # print("Cleared all keys from the current Redis database (for demo purposes).")


    # --- Common Redis Data Types and Commands ---
    print("\n--- Common Redis Data Types and Commands ---")

    # 1. STRINGS
    # The most basic Redis data type.
    # Use Cases:
    #   - Storing simple textual data or serialized objects (like JSON strings).
    #   - Atomic counters for metrics (e.g., page views, API calls).
    #   - Storing flags or status indicators.
    #   - Caching small, frequently accessed pieces of data.
    print("\n1. STRINGS:")
    r.set("rag_system_name", "AwesomeRAG")
    system_name = r.get("rag_system_name")
    print(f"  Get 'rag_system_name': {system_name}")

    # Strings can be used as counters
    r.set("api_requests_today", "0") # Initialize as string for incr
    r.incr("api_requests_today") # Increment by 1
    r.incrby("api_requests_today", 5) # Increment by 5
    requests_count = r.get("api_requests_today")
    print(f"  API requests today (counter): {requests_count}")

    # Setting with expiration (TTL - Time To Live)
    # 'ex=60' means expire in 60 seconds
    r.set("temporary_data", "This will vanish soon", ex=60)
    print(f"  Set 'temporary_data' with 60s TTL. Current TTL: {r.ttl('temporary_data')}s")


    # 2. HASHES
    # Used to store structured data, representing objects with multiple fields and values.
    # Each hash key can have its own set of field-value pairs.
    # Use Cases:
    #   - Storing user profiles or object attributes where you might want to access individual fields.
    #   - Representing session data with various attributes.
    #   - Grouping related data under a single key for better organization.
    print("\n2. HASHES (for storing object-like data):")
    user_id = "user:123"
    user_data = {
        "username": "rag_dev",
        "email": "dev@rag.com",
        "preferences": json.dumps({"theme": "dark", "notifications": True}) # Serialize complex types
    }
    # r.hmset(user_id, user_data) # hmset is deprecated in Redis, though redis-py might polyfill.
    # For redis-py 4.0+, use hset with the 'mapping' argument for multiple fields:
    r.hset(user_id, mapping=user_data)
    
    r.hset(user_id, "last_login", str(time.time())) # Add/update a single field

    retrieved_username = r.hget(user_id, "username")
    retrieved_all_user_data = r.hgetall(user_id) # Returns a dictionary
    print(f"  Retrieved username for '{user_id}': {retrieved_username}")
    print(f"  Retrieved all data for '{user_id}': {retrieved_all_user_data}")
    if retrieved_all_user_data and "preferences" in retrieved_all_user_data:
        user_prefs = json.loads(retrieved_all_user_data["preferences"])
        print(f"  Decoded preferences: {user_prefs}")


    # 3. LISTS
    # An ordered sequence of strings, maintained in insertion order.
    # Supports adding/removing elements from both ends (head/tail).
    # Use Cases:
    #   - Implementing message queues (e.g., background task processing).
    #   - Storing timelines or activity feeds (e.g., "latest 100 actions").
    #   - Buffering logs or events before batch processing.
    print("\n3. LISTS (for queues, recent items):")
    log_queue_key = "system_logs"
    r.rpush(log_queue_key, "User login: user_alpha") # Add to the right (end)
    r.rpush(log_queue_key, "Document processed: doc_xyz")
    r.lpush(log_queue_key, "System startup sequence initiated") # Add to the left (start)

    print(f"  Current log queue (all items): {r.lrange(log_queue_key, 0, -1)}")
    latest_log = r.lpop(log_queue_key) # Remove and get from the left (start)
    print(f"  Processed (LPOP) latest log: {latest_log}")
    print(f"  Log queue after LPOP: {r.lrange(log_queue_key, 0, -1)}")
    print(f"  Length of log queue: {r.llen(log_queue_key)}")


    # 4. SETS
    # An unordered collection of unique string values.
    # Adding the same element multiple times has no effect.
    # Use Cases:
    #   - Tracking unique visitors or items (e.g., unique users online).
    #   - Storing tags associated with an item.
    #   - Performing set operations like union, intersection, difference (e.g., common friends).
    print("\n4. SETS (for unique items, tags):")
    online_users_key = "online_users"
    r.sadd(online_users_key, "user:alice", "user:bob", "user:charlie")
    r.sadd(online_users_key, "user:alice") # Adding existing member has no effect
    print(f"  Online users: {r.smembers(online_users_key)}")
    print(f"  Is 'user:bob' online? {'Yes' if r.sismember(online_users_key, 'user:bob') else 'No'}")
    print(f"  Number of online users: {r.scard(online_users_key)}")


    # 5. SORTED SETS (ZSETs)
    # Similar to Sets, but each member is associated with a floating-point score.
    # Members are unique and are ordered by their scores.
    # Use Cases:
    #   - Implementing leaderboards or ranking systems.
    #   - Priority queues where the score determines priority.
    #   - Time-series data where the score could be a timestamp, allowing retrieval by time range.
    #   - Storing items with weights for weighted selection.
    print("\n5. SORTED SETS (for leaderboards, ranked items):")
    leaderboard_key = "rag_query_scores"
    r.zadd(leaderboard_key, {"query_abc": 95.5, "query_def": 88.0, "query_xyz": 99.1})
    r.zadd(leaderboard_key, {"query_abc": 96.0}) # Updates score for existing member
    
    # Get top N queries by score (highest first)
    top_queries = r.zrevrange(leaderboard_key, 0, 2, withscores=True) # Top 3 (0, 1, 2)
    print(f"  Top queries (with scores): {top_queries}")
    print(f"  Score of 'query_def': {r.zscore(leaderboard_key, 'query_def')}")
    print(f"  Rank of 'query_def' (0-based, highest score is rank 0): {r.zrevrank(leaderboard_key, 'query_def')}")

    # --- RAG Use Cases for Redis ---
    print("\n--- RAG Use Cases for Redis ---")

    # A. Caching LLM Responses
    print("\nA. Caching LLM Responses:")
    # LLM calls can be expensive and slow. Caching frequent/identical queries can save resources.
    # Key: hash of the query or a canonical query string.
    # Value: LLM response (often JSON string).
    # TTL: Set an appropriate expiration (e.g., hours, days).
    
    def get_llm_response_with_cache(r_conn, query, ttl_seconds=3600):
        cache_key = f"llm_cache:{hash(query)}" # Simple hash for demo; consider more robust hashing
        
        cached_response = r_conn.get(cache_key)
        if cached_response:
            print(f"  LLM CACHE HIT for query: '{query}'")
            return json.loads(cached_response) # Deserialize from JSON
        
        print(f"  LLM CACHE MISS for query: '{query}'. Simulating LLM call...")
        # Simulate an actual LLM call
        time.sleep(0.1) # Simulate network latency and processing
        llm_response_data = {"answer": f"This is the LLM's answer to '{query}'.", "model": "gpt-dummy", "timestamp": time.time()}
        
        # Store in cache with TTL
        r_conn.set(cache_key, json.dumps(llm_response_data), ex=ttl_seconds)
        print(f"  Stored LLM response in cache for query: '{query}' with TTL: {ttl_seconds}s")
        return llm_response_data

    query1 = "What is the capital of France?"
    query2 = "Explain Redis data structures."
    
    response1_call1 = get_llm_response_with_cache(r, query1)
    print(f"    Response 1 (Call 1): {response1_call1['answer']}")
    response1_call2 = get_llm_response_with_cache(r, query1) # Should be a cache hit
    print(f"    Response 1 (Call 2): {response1_call2['answer']}")
    
    response2 = get_llm_response_with_cache(r, query2)
    print(f"    Response 2: {response2['answer']}")


    # B. Caching Frequently Accessed Data (e.g., document metadata)
    print("\nB. Caching Frequently Accessed Data (e.g., document metadata from primary DB):")
    # If your RAG system frequently looks up metadata for specific document IDs from
    # a slower primary database (like PostgreSQL), caching this in Redis can speed things up.
    
    def get_document_metadata_with_cache(r_conn, doc_id):
        cache_key = f"doc_meta_cache:{doc_id}"
        cached_meta = r_conn.get(cache_key)
        if cached_meta:
            print(f"  DOC META CACHE HIT for doc_id: '{doc_id}'")
            return json.loads(cached_meta)
            
        print(f"  DOC META CACHE MISS for doc_id: '{doc_id}'. Simulating fetch from primary DB...")
        time.sleep(0.05) # Simulate DB query
        # Example metadata from primary DB
        doc_metadata_from_db = {"title": f"Document {doc_id}", "author": "AI Team", "tags": ["rag", "ai"], "length_pages": 10}
        
        r_conn.set(cache_key, json.dumps(doc_metadata_from_db), ex=600) # Cache for 10 minutes
        print(f"  Stored doc metadata in cache for doc_id: '{doc_id}'")
        return doc_metadata_from_db

    meta1 = get_document_metadata_with_cache(r, "doc_001")
    print(f"    Metadata for doc_001: {meta1['title']}")
    meta1_again = get_document_metadata_with_cache(r, "doc_001") # Cache hit
    print(f"    Metadata for doc_001 (again): {meta1_again['title']}")


    # C. Session Management
    print("\nC. Session Management (basic example):")
    # Storing user session data.
    # Key: session_id
    # Value: Hash containing user_id, last_active_time, chat_history_ids (pointers), etc.
    
    def create_user_session(r_conn, user_id_val):
        session_id_val = f"session:{hash(user_id_val + str(time.time()))}" # Simplistic session ID
        session_data = {
            "user_id": user_id_val,
            "created_at": str(time.time()),
            "last_active": str(time.time())
            # "chat_history_preview": "User asked about X..."
        }
        r_conn.hset(session_id_val, mapping=session_data)
        r_conn.expire(session_id_val, 7200) # Session expires in 2 hours
        print(f"  Created session '{session_id_val}' for user '{user_id_val}'.")
        return session_id_val

    def get_session_data(r_conn, session_id_val):
        return r_conn.hgetall(session_id_val)

    def update_session_activity(r_conn, session_id_val):
        if r_conn.exists(session_id_val):
            r_conn.hset(session_id_val, "last_active", str(time.time()))
            r_conn.expire(session_id_val, 7200) # Refresh TTL
            print(f"  Updated last_active for session '{session_id_val}'.")
            return True
        return False

    user_session = create_user_session(r, "user_rag_fan")
    session_details = get_session_data(r, user_session)
    print(f"    Initial session details: {session_details}")
    time.sleep(1) # Simulate some activity
    update_session_activity(r, user_session)
    session_details_updated = get_session_data(r, user_session)
    print(f"    Updated session details: {session_details_updated}")


    # D. Rate Limiting
    print("\nD. Rate Limiting (simple fixed window counter):")
    # Prevent abuse of API endpoints.
    # Key: e.g., `ratelimit:{user_id}:{api_endpoint}`
    # Value: Counter (String type)
    # TTL: Window size (e.g., 60 seconds)
    
    def is_rate_limited(r_conn, user_id_val, endpoint, limit_per_window, window_seconds=60):
        key = f"ratelimit:{user_id_val}:{endpoint}"
        
        # Atomically increment and get the new value.
        # If the key does not exist, it's created and set to 1.
        # The pipeline ensures these operations are atomic.
        pipe = r_conn.pipeline()
        pipe.incr(key)
        pipe.ttl(key) # Get current TTL to see if it's a new key
        current_count, key_ttl = pipe.execute()

        if key_ttl == -2 or key_ttl == -1 : # Key didn't exist or had no TTL (should be -2 if new from INCR)
            # This is the first request in this window (or key expired and this is the first new one)
            r_conn.expire(key, window_seconds)
            # current_count is already 1 from INCR

        if int(current_count) > limit_per_window:
            print(f"  RATE LIMITED: User '{user_id_val}' for endpoint '{endpoint}'. Count: {current_count}, Limit: {limit_per_window}")
            # Optionally, you might not want to expire the key here if you want to keep them blocked
            # or you might implement a more sophisticated blocking mechanism.
            return True # Limited
            
        print(f"  Request allowed for user '{user_id_val}', endpoint '{endpoint}'. Count: {current_count}")
        return False

    user_for_rate_limit = "test_user_rl"
    api_endpoint_rl = "query_llm"
    request_limit = 3 # Max 3 requests per window

    for i in range(request_limit + 2): # Try 5 requests
        print(f"  Attempting request {i+1} for rate limiting test...")
        if not is_rate_limited(r, user_for_rate_limit, api_endpoint_rl, request_limit):
            print(f"    Request {i+1} processed.")
        else:
            print(f"    Request {i+1} blocked by rate limiter.")
        time.sleep(0.1) # Small delay between attempts

    key_to_check_ttl = f"ratelimit:{user_for_rate_limit}:{api_endpoint_rl}"
    ttl_val = r.ttl(key_to_check_ttl)
    if ttl_val and ttl_val > 0 :
        print(f"  Rate limit key '{key_to_check_ttl}' will expire in {ttl_val}s. Waiting for it to pass...")
        # time.sleep(ttl_val + 1) # Wait for window to expire
        # print("  Window passed. Retrying...")
        # if not is_rate_limited(r, user_for_rate_limit, api_endpoint_rl, request_limit):
        #      print(f"    Request after window processed.")
    else:
        print(f"  Rate limit key '{key_to_check_ttl}' has no TTL or doesn't exist (current TTL: {ttl_val}).")


    # --- Redis Persistence Overview ---
    print("\n--- Redis Persistence Overview ---")
    # While Redis is primarily in-memory, it offers persistence options to save data to disk:
    # 1. RDB (Redis Database Backup):
    #    - Point-in-time snapshots of your dataset at specified intervals.
    #    - Good for backups, disaster recovery, faster restarts with large datasets.
    #    - Configuration: `save <seconds> <changes>` (e.g., `save 900 1` means save if 1 key changed in 900s).
    #    - Can lead to some data loss between snapshots if Redis crashes.
    #
    # 2. AOF (Append Only File):
    #    - Logs every write operation received by the server. Replays these on restart.
    #    - More durable than RDB. Configurable fsync policies (e.g., every second, every write).
    #    - AOF file can grow large; Redis supports automatic rewriting in the background.
    #    - Can be slower to restart than RDB if AOF file is very large (though rewrite helps).
    #
    # 3. No Persistence:
    #    - Data exists only in memory and is lost on restart. Suitable for pure caching scenarios.
    #
    # 4. RDB + AOF:
    #    - Can combine both for greater data safety. On restart, Redis will use AOF as it's generally more complete.
    #
    # Choice depends on your application's durability requirements. For RAG caching,
    # some data loss might be acceptable, making RDB or even no persistence viable.
    # For session data or rate limiting counters that need to survive restarts, AOF is often preferred.
    # Configuration is typically done in the `redis.conf` file.
    print("  Redis offers RDB (snapshots) and AOF (append-only file) for persistence.")
    print("  The choice depends on durability needs vs. performance.")


def main():
    redis_client = connect_to_redis()
    if redis_client:
        # Clean up keys from previous runs for a cleaner demo, if desired
        # Be careful with this in a shared Redis instance.
        # keys_to_delete = redis_client.keys("llm_cache:*") + \
        #                  redis_client.keys("doc_meta_cache:*") + \
        #                  redis_client.keys("session:*") + \
        #                  redis_client.keys("ratelimit:*") + \
        #                  redis_client.keys("rag_system_name") + \
        #                  redis_client.keys("api_requests_today") + \
        #                  redis_client.keys("temporary_data") + \
        #                  redis_client.keys("user:123") + \
        #                  redis_client.keys("system_logs") + \
        #                  redis_client.keys("online_users") + \
        #                  redis_client.keys("rag_query_scores")
        # if keys_to_delete:
        #    redis_client.delete(*keys_to_delete)
        #    print(f"Deleted {len(keys_to_delete)} demo keys.")
            
        redis_rag_examples(redis_client)
        redis_client.close() # Good practice, though connections are often pooled in apps
        print("\nRedis client usage finished (connection implicitly closed by Python's GC if not explicitly).")

    print("\nModule C2 execution finished.")
    print("Review the script 'C2_redis_for_rag.py' for details.")

if __name__ == "__main__":
    main()