# Automatic Query Review System

This Python script automates the process of reviewing user queries by leveraging cosine similarity and a vector database (Qdrant). It identifies new queries by comparing incoming queries with previous ones stored in the vector database.

## Overview

The script performs the following steps:

1. **Fetch Unprocessed Logs**: Retrieves user queries from your message logs database that haven't been processed yet.
2. **Generate Embeddings**: Uses OpenAI's embedding model to generate vector representations of the queries.
3. **Search Similar Queries**: Searches for similar queries in the Qdrant vector database.
4. **Calculate Similarity Scores**: Determines the similarity between the new query and existing ones.
5. **Classify Queries**: Classifies the query as new or existing based on a similarity threshold.
6. **Update Database**: Updates your message logs database with the similarity scores and classification results.
7. **Add to Qdrant**: Adds the new query to the Qdrant database for future comparisons.

## Prerequisites

- **Python 3.7 or higher**
- **PostgreSQL Database**: For storing message logs.
- **Qdrant Vector Database**: For storing and searching vector embeddings.
- **OpenAI API Key**: For generating embeddings.

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/yourusername/automatic-query-review.git
cd automatic-query-review

### 2. Install Dependencies

Create and activate a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

Install the required packages:

pip install -r requirements.txt

### 3. Configure Environment Variables

Create a .env file in the project root directory with the following content:

DATABASE_URL=your_postgresql_database_url
QDRANT_HOST=your_qdrant_host
QDRANT_API_KEY=your_qdrant_api_key
OPENAI_API_KEY=your_openai_api_key

Replace the placeholder values with your actual credentials.

### 4. Adjust Configuration (Optional)

Review and adjust the configuration variables in the script (automatic_query_review.py) if necessary:

	•	QDRANT_COLLECTION_NAME: The name of the collection in Qdrant.
	•	EMBEDDING_MODEL: The OpenAI model used for embeddings.
	•	SIMILARITY_THRESHOLD: The threshold for classifying a query as new.

### 5. Prepare Your Database

Ensure your PostgreSQL database (message_logs table) has the following columns:

	•	log_id (Primary Key)
	•	message_content (Text)
	•	response_content (Text)
	•	timestamp (Timestamp)
	•	similarity_score (Float, nullable)
	•	similar_queries (Text, nullable)
	•	is_new_query (Boolean, nullable)
	•	message_type (Text) - Used to filter queries.

### 6. Run the Script

Execute the script:

python automatic_query_review.py

You will be prompted to enter a cutoff date. Enter a date in YYYY-MM-DD format or press Enter to process all unprocessed logs.

How It Works

	•	Embeddings: The script uses OpenAI’s embedding model to convert text queries into high-dimensional vectors.
	•	Vector Database: Qdrant stores these vectors and allows for efficient similarity search.
	•	Similarity Calculation: By calculating the cosine similarity between vectors, the script determines how similar a new query is to existing ones.
	•	Classification: Based on a predefined similarity threshold, the script classifies queries as new or existing.

Customization

	•	Similarity Threshold: Adjust the SIMILARITY_THRESHOLD variable in the script to fine-tune the sensitivity of new query detection.
	•	Embedding Model: Change the EMBEDDING_MODEL variable to use different OpenAI models.
	•	Database Queries: Modify SQL queries in the script to match your database schema.

Dependencies

Install the required packages using:

pip install psycopg2-binary python-dotenv qdrant-client langchain openai

Alternatively, you can use the requirements.txt file:

requirements.txt

psycopg2-binary
python-dotenv
qdrant-client
langchain
openai

License

This project is licensed under the MIT License.

Contact

For any questions or issues, please open an issue on GitHub or contact me directly.

---

**Note**: Ensure you have the necessary API keys and access to the required services (PostgreSQL database, Qdrant, OpenAI API) before running the script.

**Dependencies**:

- **psycopg2-binary**: PostgreSQL database adapter for Python.
- **python-dotenv**: For loading environment variables from a `.env` file.
- **qdrant-client**: Client library for interacting with Qdrant.
- **langchain**: Framework for working with language models.
- **openai**: OpenAI API client.

You can install all dependencies using:

pip install -r requirements.txt

How to Use the Script:

	1.	Set Up Environment Variables: Create a .env file with your database URL, Qdrant host and API key, and OpenAI API key.
	2.	Install Dependencies: Run pip install -r requirements.txt.
	3.	Adjust Configuration: Modify the variables in the script if needed.
	4.	Run the Script: Execute python automatic_query_review.py and follow the prompts.

Understanding the Script:

	•	Functions: The script is modular, with functions handling database connections, Qdrant interactions, embeddings, and classification.
	•	Comments: Each function and significant code block includes comments explaining its purpose and functionality.
	•	Main Execution: The process_logs function orchestrates the entire process, fetching logs, processing them, and updating the database.
