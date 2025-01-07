from openai import OpenAI
from pinecone import Pinecone
import pandas as pd
from tqdm import tqdm
import os
import csv
import time

# Initialize clients
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Connect to your Pinecone index
index_name = "cpt-openai"
index = pc.Index(index_name)

# Load the CPT data
print("Reading CSV file...")
valid_rows = []
skipped_rows = []

with open("CPTList-New.csv", 'r', encoding='utf-8-sig', newline='') as file:
    reader = csv.reader(file, quotechar='"', delimiter=',')
    for i, row in enumerate(reader, 1):
        if len(row) == 3:
            # Clean the Code field of any non-ASCII characters
            cleaned_code = row[0].strip()
            if cleaned_code and cleaned_code.isascii():
                valid_rows.append([cleaned_code, row[1], row[2]])
        else:
            skipped_rows.append((i, row))

# Save skipped rows to file
if skipped_rows:
    with open('cpt_skipped_rows.txt', 'w', encoding='utf-8') as f:
        for line_num, row in skipped_rows:
            f.write(f"Line {line_num}: {row}\n")

# Convert to DataFrame
cpt_data = pd.DataFrame(valid_rows, columns=['Code', 'Name', 'Description'])
print(
    f"Loaded {len(cpt_data)} valid CPT codes. Skipped {len(skipped_rows)} rows.")

# Generate embeddings and prepare data for upsertion
vectors_to_upsert = []
print("Generating embeddings...")

# Combine Name and Description for better semantic understanding
cpt_data['combined_text'] = cpt_data['Name'] + ". " + cpt_data['Description']

for _, row in tqdm(cpt_data.iterrows(), total=len(cpt_data)):
    # Generate embedding using OpenAI
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=row['combined_text']
    )
    embedding = response.data[0].embedding

    # Prepare a vector for upsertion
    vectors_to_upsert.append({
        "id": str(row["Code"]),
        "values": embedding,
        "metadata": {
            "code": str(row["Code"]),
            "name": row["Name"],
            "description": row["Description"],
            "combined_text": row["combined_text"]
        }
    })

    # Rate limiting for OpenAI API
    time.sleep(0.1)

# Upsert the embeddings into the index
batch_size = 300
print("Upserting embeddings into Pinecone...")
for i in tqdm(range(0, len(vectors_to_upsert), batch_size)):
    batch = vectors_to_upsert[i:i + batch_size]
    index.upsert(vectors=batch)

print(
    f"Successfully upserted {len(vectors_to_upsert)} vectors into the Pinecone index!")
print(index.describe_index_stats())
