from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
from openai import OpenAI

# Set OpenAI API key
# Ensure your OpenAI API key is set in environment variables
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# Connect to the ICD-10 index
index_name = "icd-10"  # Replace with your index name
index = pc.Index(index_name)

# Connect to the ICD-10 index
index_name = "icd10-openai"  # Replace with your index name
openai_index = pc.Index(index_name)

# Load the SentenceTransformer model
# Same model used for embedding generation
model = SentenceTransformer("all-MiniLM-L6-v2")


# Function to generate embeddings using OpenAI
def generate_openai_embedding(text):
    response = client.embeddings.create(model="text-embedding-ada-002",
                                        input=text)
    return response.data[0].embedding

# Query function


def query_icd10(user_query, top_k=5):
    """
    Query the Pinecone index to retrieve ICD-10 codes based on semantic similarity.

    Args:
        user_query (str): The user input query.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: List of top matching ICD-10 codes and descriptions.
    """
    # Generate embedding for the user query
    query_embedding = model.encode(user_query).tolist()

    # Search the Pinecone index using keyword arguments
    results = index.query(vector=query_embedding,
                          top_k=top_k, include_metadata=True)

    # Parse and return results
    return [
        {"Code": match["id"], "Name": match["metadata"]["name"],
            "Description": match["metadata"]["description"], "Score": match["score"]}
        for match in results["matches"]
    ]

# Function to query the Pinecone index


def query_openai_index(user_query, top_k=5):
    """
    Query the Pinecone index to retrieve ICD-10 codes based on semantic similarity.

    Args:
        user_query (str): The user input query.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: List of top matching ICD-10 codes and descriptions.
    """
    # Generate the query embedding using OpenAI
    query_embedding = generate_openai_embedding(user_query)

    # Query the Pinecone index
    results = openai_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Parse and return results
    return [
        {"Code": match["id"], "Name": match["metadata"].get(
            "name", ""), "Description": match["metadata"].get("description", ""), "Score": match["score"]}
        for match in results["matches"]
    ]


# Test the query function
if __name__ == "__main__":
    # user_query = "cholera symptoms due to Vibrio cholerae"
    # user_query = "sugar problem with blurry eyes"
    user_query = "High BP"
    # user_query = "piles"
    results = query_icd10(user_query)
    openai_results = query_openai_index(user_query)

    print(f"Query Results for {user_query}:")
    for res in results:
        print(
            f"Code: {res['Code']}, Name: {res['Name']}, Description: {res['Description']}, Score: {res['Score']:.2f}")

    print("\nOpenAI Results:")
    for res in openai_results:
        print(
            f"Code: {res['Code']}, Name: {res['Name']}, Description: {res['Description']}, Score: {res['Score']:.2f}")
