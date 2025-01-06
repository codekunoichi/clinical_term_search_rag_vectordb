from openai import OpenAI


from pinecone import Pinecone
import pandas as pd
import os
import time
from tqdm import tqdm  # Progress bar library
import time  # To measure execution time


# Set OpenAI API key
# Ensure your OpenAI API key is set in environment variables
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# Connect to the ICD-10 index
index_name = "icd10-openai"  # Replace with your index name
index = pc.Index(index_name)

# Load the ICD-10 data
icd10_data = pd.read_csv("ICD10.csv")

# Start timer
start_time = time.time()

# Function to generate embeddings using OpenAI


def generate_openai_embedding(text):
    response = client.embeddings.create(model="text-embedding-ada-002",
                                        input=text)
    return response.data[0].embedding


# Generate embeddings and prepare data for upsertion
vectors_to_upsert = []
print("Generating embeddings and preparing data for upsertion...")

for _, row in tqdm(icd10_data.iterrows(), total=len(icd10_data), desc="Embedding Generation"):
    embedding = generate_openai_embedding(row['Description'])
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

# Upsert the embeddings into the Pinecone index
batch_size = 300
print(f"Upserting {len(vectors_to_upsert)} vectors into Pinecone...")

for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc="Upsertion Progress"):
    batch = vectors_to_upsert[i:i + batch_size]
    index.upsert(vectors=batch)

# End timer
end_time = time.time()

# Calculate and print total execution time
total_time = end_time - start_time
print(
    f"Total time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

print(
    f"Successfully upserted {len(vectors_to_upsert)} vectors into the Pinecone index!")
