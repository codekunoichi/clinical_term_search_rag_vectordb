from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import os

# Initialize Pinecone
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

# Connect to the ICD-10 index
index_name = "icd-10"  # Replace with your index name
index = pc.Index(index_name)

# Load the ICD-10 data
# Replace 'ICD10.csv' with the actual path to your CSV file
icd10_data = pd.read_csv("ICD10.csv")
# Ensure the DataFrame contains 'Code', 'Name', and 'Description' columns

# Initialize the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and efficient for embeddings

# Generate embeddings and prepare data for upsertion
vectors_to_upsert = []
for _, row in icd10_data.iterrows():
    # Generate an embedding using the 'Description' column
    embedding = model.encode(row['Description']).tolist()  # Convert to a list for Pinecone
    
    # Prepare a vector for upsertion
    vectors_to_upsert.append({
        "id": row["Code"],  # Use the ICD-10 code as the unique ID
        "values": embedding,
        "metadata": {
            "name": row["Name"],              # Add the 'Name' column to metadata
            "description": row["Description"] # Add the 'Description' column to metadata
        }
    })

# Upsert the embeddings into the index
# Pinecone recommends upserting in batches to improve performance
batch_size = 100
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i + batch_size]
    index.upsert(vectors=batch)

print(f"Successfully upserted {len(vectors_to_upsert)} vectors into the Pinecone index!")