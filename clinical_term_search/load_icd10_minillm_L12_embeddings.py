from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import pandas as pd
from tqdm import tqdm  # For progress tracking
import os
# Initialize Pinecone
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# Connect to your Pinecone index
index_name = "icd10-minilm-l12"  # New index name for MiniLM-L12 embeddings
index = pc.Index(index_name)

# Load the ICD-10 data
# Replace 'ICD10.csv' with the actual path to your CSV file
icd10_data = pd.read_csv("ICD10.csv")
# Ensure the DataFrame contains 'Code', 'Name', and 'Description' columns

# Initialize the SentenceTransformer model (MiniLM-L12)
# Larger MiniLM model for better embeddings
model = SentenceTransformer("all-MiniLM-L12-v2")

# Generate embeddings and prepare data for upsertion
vectors_to_upsert = []
print("Generating embeddings...")
for _, row in tqdm(icd10_data.iterrows(), total=len(icd10_data)):
    # Generate an embedding using the 'Description' column
    # Convert to a list for Pinecone
    embedding = model.encode(row['Description']).tolist()

    # Prepare a vector for upsertion
    vectors_to_upsert.append({
        "id": row["Code"],  # Use the ICD-10 code as the unique ID
        "values": embedding,
        "metadata": {
            # Add the 'Name' column to metadata
            "name": row["Name"],
            # Add the 'Description' column to metadata
            "description": row["Description"]
        }
    })

# Upsert the embeddings into the index
# Pinecone recommends upserting in batches to improve performance
batch_size = 100
print("Upserting embeddings into Pinecone...")
for i in tqdm(range(0, len(vectors_to_upsert), batch_size)):
    batch = vectors_to_upsert[i:i + batch_size]
    index.upsert(vectors=batch)

print(
    f"Successfully upserted {len(vectors_to_upsert)} vectors into the Pinecone index!")
print(index.describe_index_stats())
