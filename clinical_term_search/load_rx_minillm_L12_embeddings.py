from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import pandas as pd
from tqdm import tqdm
import os
import csv

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Connect to your Pinecone index
index_name = "rxcode-minillm-l12"
index = pc.Index(index_name)

# Load the medication data
print("Reading CSV file...")
valid_rows = []
skipped_rows = []

with open("Medication-New.csv", 'r', encoding='utf-8-sig', newline='') as file:
    reader = csv.reader(file, quotechar='"', delimiter=',')
    next(reader)  # Skip header row
    for i, row in enumerate(reader, 1):
        if len(row) == 7:  # NDC,DrugName,DrugTradeName,DosageForm,Route,Strength,RxNorm
            # Clean the NDC code of any non-ASCII characters
            cleaned_code = row[0].strip()
            if cleaned_code and cleaned_code.isascii():
                valid_rows.append(row)
        else:
            skipped_rows.append((i, row))

# Save skipped rows to file
if skipped_rows:
    with open('rx_skipped_rows.txt', 'w', encoding='utf-8') as f:
        for line_num, row in skipped_rows:
            f.write(f"Line {line_num}: {row}\n")

# Convert to DataFrame
rx_data = pd.DataFrame(valid_rows, columns=[
    'NDC', 'DrugName', 'DrugTradeName', 'DosageForm', 'Route', 'Strength', 'RxNorm'])
print(
    f"Loaded {len(rx_data)} valid medication codes. Skipped {len(skipped_rows)} rows.")

# Initialize the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L12-v2")

# Generate embeddings and prepare data for upsertion
vectors_to_upsert = []
print("Generating embeddings...")

# Combine fields for better semantic understanding
rx_data['combined_text'] = rx_data.apply(
    lambda x: f"{x['DrugName']}. {x['DrugTradeName']}. {x['DosageForm']} {x['Route']} {x['Strength']}", axis=1)

for _, row in tqdm(rx_data.iterrows(), total=len(rx_data)):
    # Generate embedding using the combined text
    embedding = model.encode(row['combined_text']).tolist()

    # Prepare a vector for upsertion
    vectors_to_upsert.append({
        "id": str(row["NDC"]),
        "values": embedding,
        "metadata": {
            "ndc": str(row["NDC"]),
            "drug_name": row["DrugName"],
            "trade_name": row["DrugTradeName"],
            "dosage_form": row["DosageForm"],
            "route": row["Route"],
            "strength": row["Strength"],
            "rxnorm": row["RxNorm"],
            "combined_text": row["combined_text"]
        }
    })

# Upsert the embeddings into the index
batch_size = 300
print("Upserting embeddings into Pinecone...")
for i in tqdm(range(0, len(vectors_to_upsert), batch_size)):
    batch = vectors_to_upsert[i:i + batch_size]
    index.upsert(vectors=batch)

print(
    f"Successfully upserted {len(vectors_to_upsert)} vectors into the Pinecone index!")
print(index.describe_index_stats())
