from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os

# Initialize Pinecone
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

# Connect to the ICD-10 index
index_name = "icd-10"  # Replace with your index name
index = pc.Index(index_name)


# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Same model used for embedding generation

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
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Parse and return results
    return [
        {"Code": match["id"], "Name": match["metadata"]["name"], "Description": match["metadata"]["description"], "Score": match["score"]}
        for match in results["matches"]
    ]

# Test the query function
if __name__ == "__main__":
    # user_query = "cholera symptoms due to Vibrio cholerae"
    # user_query = "sugar problem with blurry eyes"
    # user_query = "High BP"
    user_query = "piles"
    results = query_icd10(user_query)

    print(f"Query Results for {user_query}:")
    for res in results:
        print(f"Code: {res['Code']}, Name: {res['Name']}, Description: {res['Description']}, Score: {res['Score']:.2f}")