import openai
import pinecone
import pandas as pd
from pinecone import Pinecone
import os
import time  # To measure execution time

# Start timer
start_time = time.time()


# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")  # Ensure your OpenAI API key is set in environment variables

# Initialize Pinecone
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

# Connect to the ICD-10 index
index_name = "icd10-openai"  # Replace with your index name
index = pinecone.Index(index_name)

print("Read the CSV...")
# Load the ICD-10 data
# Replace 'ICD10.csv' with the actual path to your CSV file
icd10_data = pd.read_csv("ICD10.csv")
# Ensure the DataFrame contains 'Code', 'Name', and 'Description' columns

print("Generating embeddings using openai")
# Function to generate embeddings using OpenAI
def generate_openai_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

# Generate embeddings and prepare data for upsertion
vectors_to_upsert = []
print("Generating embeddings and preparing data for upsertion...")
for _, row in icd10_data.iterrows():
    # Generate an embedding using the 'Description' column
    embedding = generate_openai_embedding(row['Description'])
    
    # Prepare a vector for upsertion
    vectors_to_upsert.append({
        "id": row["Code"],  # Use the ICD-10 code as the unique ID
        "values": embedding,
        "metadata": {
            "name": row["Name"],              # Add the 'Name' column to metadata
            "description": row["Description"] # Add the 'Description' column to metadata
        }
    })

# Upsert the embeddings into the Pinecone index
# Pinecone recommends upserting in batches to improve performance
batch_size = 200
print(f"Upserting {len(vectors_to_upsert)} vectors into Pinecone...")
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i + batch_size]
    index.upsert(vectors=batch)

print(f"Successfully upserted {len(vectors_to_upsert)} vectors into the Pinecone index!")

print(index)

# End timer
end_time = time.time()

# Calculate and print total execution time
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

