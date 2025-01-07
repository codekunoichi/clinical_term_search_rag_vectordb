from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone import Pinecone
import csv
import time
from tqdm import tqdm
import os

# Initialize clients
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Initialize models and indexes
minilm_model = SentenceTransformer("all-MiniLM-L12-v2")
minilm_index = pc.Index("cpt-minilm-l12")
openai_index = pc.Index("cpt-openai")


def search_across_models(query, top_k=10):
    # Get MiniLM-L12 embeddings and search
    minilm_embedding = minilm_model.encode(query).tolist()
    minilm_results = minilm_index.query(
        vector=minilm_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Get OpenAI embeddings and search
    openai_response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    openai_embedding = openai_response.data[0].embedding
    openai_results = openai_index.query(
        vector=openai_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return {
        'MiniLM-L12': minilm_results.matches,
        'OpenAI': openai_results.matches
    }


def generate_comparison_csv():
    # Common procedure search terms
    search_terms = [
        "appendectomy",
        "knee replacement",
        "colonoscopy",
        "cataract surgery",
        "heart bypass",
        "hip replacement",
        "tonsillectomy",
        "gallbladder removal",
        "hernia repair",
        "dental cleaning",
        "x-ray chest",
        "mri brain",
        "ct scan abdomen",
        "ultrasound pregnancy",
        "physical therapy",
        "vaccination",
        "blood test",
        "ekg",
        "endoscopy",
        "biopsy"
    ]

    # Prepare CSV data
    csv_data = []
    headers = [
        'Search Term',
        'Rank',
        'MiniLM-L12 Code',
        'MiniLM-L12 Name',
        'MiniLM-L12 Description',
        'MiniLM-L12 Score',
        'OpenAI Code',
        'OpenAI Name',
        'OpenAI Description',
        'OpenAI Score'
    ]

    print("Generating comparison results...")
    for term in tqdm(search_terms):
        print(f"Processing: {term}")
        results = search_across_models(term)

        # Process all 10 results for each term
        for rank in range(10):
            row = {
                'Search Term': term,
                'Rank': rank + 1,
                'MiniLM-L12 Code': results['MiniLM-L12'][rank].metadata['code'],
                'MiniLM-L12 Name': results['MiniLM-L12'][rank].metadata['name'],
                'MiniLM-L12 Description': results['MiniLM-L12'][rank].metadata['description'],
                'MiniLM-L12 Score': results['MiniLM-L12'][rank].score,
                'OpenAI Code': results['OpenAI'][rank].metadata['code'],
                'OpenAI Name': results['OpenAI'][rank].metadata['name'],
                'OpenAI Description': results['OpenAI'][rank].metadata['description'],
                'OpenAI Score': results['OpenAI'][rank].score
            }
            csv_data.append(row)

        time.sleep(1)  # Prevent rate limiting

    # Write to CSV
    output_file = 'cpt_model_comparison_detailed.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"\nDetailed comparison CSV generated: {output_file}")


if __name__ == "__main__":
    generate_comparison_csv()
