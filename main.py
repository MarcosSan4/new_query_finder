"""
Automatic Query Review System

This script automates the process of reviewing user queries by leveraging cosine similarity and a vector database (Qdrant).
It identifies new queries by comparing incoming queries with previous ones stored in the vector database.

Dependencies:
- psycopg2-binary
- python-dotenv
- qdrant-client
- langchain
- openai

Make sure to install the required packages and set up your environment variables before running the script.
"""

import os
import psycopg2
from datetime import datetime
import uuid
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration variables

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')

# Qdrant configuration
QDRANT_HOST = os.getenv('QDRANT_HOST')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_COLLECTION_NAME = 'message_logs'  # You can change this to your preferred collection name

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Embedding model configuration
EMBEDDING_MODEL = 'text-embedding-ada-002'  # Or any other model you prefer

# Similarity threshold for classifying new queries
SIMILARITY_THRESHOLD = 0.8  # Adjust this value based on your needs

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)

def get_database_connection():
    """Establish a connection to the PostgreSQL database."""
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def get_qdrant_client():
    """Initialize and return a Qdrant client."""
    client = QdrantClient(
        host=QDRANT_HOST,
        api_key=QDRANT_API_KEY
    )
    return client

def ensure_qdrant_collection(client, vector_size=1536):
    """Ensure that the Qdrant collection exists."""
    collections = client.get_collections().collections
    if QDRANT_COLLECTION_NAME in [collection.name for collection in collections]:
        print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists.")
    else:
        print(f"Creating collection '{QDRANT_COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Collection '{QDRANT_COLLECTION_NAME}' created successfully.")

def get_embedding(text):
    """Generate an embedding for the given text."""
    embedding = embeddings.embed_query(text)
    return embedding

def fetch_unprocessed_logs(conn, cutoff_date=None):
    """Fetch unprocessed logs from the database."""
    with conn.cursor() as cur:
        if cutoff_date:
            cur.execute("""
                SELECT log_id, message_content, response_content, timestamp
                FROM message_logs
                WHERE similarity_score IS NULL
                AND message_type = 'query'
                AND timestamp <= %s
                ORDER BY timestamp ASC
            """, (cutoff_date,))
        else:
            cur.execute("""
                SELECT log_id, message_content, response_content, timestamp
                FROM message_logs
                WHERE similarity_score IS NULL
                AND message_type = 'query'
                ORDER BY timestamp ASC
            """)
        logs = cur.fetchall()
    return logs

def add_log_to_qdrant(client, log_id, message_content, response_content):
    """Add a log entry to the Qdrant vector database."""
    try:
        embedding = get_embedding(message_content)
        point_id = str(uuid.uuid4())
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "log_id": log_id,
                "message_content": message_content,
                "response_content": response_content
            }
        )
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[point]
        )
        print(f"Log {log_id} added to Qdrant.")
        return True
    except Exception as e:
        print(f"Error adding log to Qdrant: {e}")
        return False

def search_similar_queries(client, embedding, top_k=3):
    """Search for similar queries in the Qdrant database."""
    try:
        search_results = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=embedding,
            limit=top_k
        )
        return search_results
    except Exception as e:
        print(f"Error searching in Qdrant: {e}")
        return []

def update_log_with_similarity(conn, log_id, similarity_score, similar_queries_str):
    """Update the log with similarity score and similar queries."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE message_logs
            SET similarity_score = %s, similar_queries = %s
            WHERE log_id = %s
        """, (similarity_score, similar_queries_str, log_id))
        conn.commit()

def classify_query(similarity_score):
    """
    Classify the query as new or not based on the similarity score.
    Returns True if it's a new query, False otherwise.
    """
    if similarity_score < SIMILARITY_THRESHOLD:
        return True  # It's a new query
    else:
        return False  # Not a new query

def update_log_with_classification(conn, log_id, is_new_query):
    """Update the log with the classification result."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE message_logs
            SET is_new_query = %s
            WHERE log_id = %s
        """, (is_new_query, log_id))
        conn.commit()

def process_logs(cutoff_date=None):
    """Main function to process logs and perform automatic query review."""
    conn = get_database_connection()
    client = get_qdrant_client()
    ensure_qdrant_collection(client)
    try:
        logs = fetch_unprocessed_logs(conn, cutoff_date)
        if not logs:
            print("No unprocessed logs found.")
            return

        for log in logs:
            log_id, message_content, response_content, timestamp = log
            print(f"\nProcessing log_id {log_id}...")

            # Generate embedding
            embedding = get_embedding(message_content)

            # Search for similar queries
            search_results = search_similar_queries(client, embedding, top_k=3)

            # Process search results
            similarity_scores = []
            similar_queries = []

            for result in search_results:
                metadata = result.payload
                if metadata and 'log_id' in metadata and 'message_content' in metadata:
                    # Qdrant returns distance; convert it to similarity
                    # For cosine similarity, the distance is (1 - cosine_similarity)
                    # So similarity = 1 - distance
                    distance = result.score  # This is the cosine distance
                    similarity = 1 - distance
                    similarity_scores.append(similarity)
                    similar_queries.append(f"{metadata['message_content']} (log_id: {metadata['log_id']})")

            # Determine the highest similarity score
            if similarity_scores:
                top_similarity_score = max(similarity_scores)
            else:
                top_similarity_score = 0.0

            similar_queries_str = "; ".join(similar_queries) if similar_queries else "No similar queries found"

            # Add the current log to Qdrant
            success = add_log_to_qdrant(client, log_id, message_content, response_content)
            if not success:
                print(f"Skipping log_id {log_id} due to error.")
                continue

            # Update the database with similarity info
            update_log_with_similarity(conn, log_id, top_similarity_score, similar_queries_str)

            # Classify the query
            is_new_query = classify_query(top_similarity_score)
            update_log_with_classification(conn, log_id, is_new_query)

            print(f"Log {log_id} processed successfully.")

    except Exception as e:
        print(f"Error processing logs: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    # Optional: Get cutoff date from user input
    cutoff_date_str = input("Enter cutoff date (YYYY-MM-DD) or press Enter to process all logs: ").strip()
    if cutoff_date_str:
        try:
            cutoff_date = datetime.strptime(cutoff_date_str, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Processing all logs.")
            cutoff_date = None
    else:
        cutoff_date = None

    process_logs(cutoff_date)
