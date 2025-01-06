from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone import Pinecone
import pandas as pd
import os
from tabulate import tabulate
import time

# Initialize clients
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Initialize models
minilm_l6_model = SentenceTransformer("all-MiniLM-L6-v2")
minilm_l12_model = SentenceTransformer("all-MiniLM-L12-v2")

# Connect to indices
index_minilm_l6 = pc.Index("icd-10")
index_minilm_l12 = pc.Index("icd10-minilm-l12")
index_openai = pc.Index("icd10-openai")


def generate_openai_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


def search_across_models(query, top_k=3):
    results = {
        "Query": query,
        "MiniLM-L6": [],
        "MiniLM-L12": [],
        "OpenAI": []
    }

    # Generate embeddings for each model
    minilm_l6_embedding = minilm_l6_model.encode(query).tolist()
    minilm_l12_embedding = minilm_l12_model.encode(query).tolist()
    openai_embedding = generate_openai_embedding(query)

    # Query each index
    l6_results = index_minilm_l6.query(
        vector=minilm_l6_embedding, top_k=top_k, include_metadata=True)
    l12_results = index_minilm_l12.query(
        vector=minilm_l12_embedding, top_k=top_k, include_metadata=True)
    openai_results = index_openai.query(
        vector=openai_embedding, top_k=top_k, include_metadata=True)

    # Process results
    for match in l6_results["matches"]:
        results["MiniLM-L6"].append({
            "code": match.id,
            "description": match.metadata["description"],
            "score": round(match.score, 3)
        })

    for match in l12_results["matches"]:
        results["MiniLM-L12"].append({
            "code": match.id,
            "description": match.metadata["description"],
            "score": round(match.score, 3)
        })

    for match in openai_results["matches"]:
        results["OpenAI"].append({
            "code": match.id,
            "description": match.metadata["description"],
            "score": round(match.score, 3)
        })

    return results


def format_results_table(results):
    output = []
    headers = ["Model", "Code", "Description", "Score"]

    # Add header
    output.append("-" * 100)
    output.append(
        f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<50} {headers[3]:<10}")
    output.append("-" * 100)

    for model in ["MiniLM-L6", "MiniLM-L12", "OpenAI"]:
        for idx, result in enumerate(results[model]):
            model_name = model if idx == 0 else ""
            line = f"{model_name:<15} {result['code']:<10} {result['description'][:47]:<50} {result['score']:<10}"
            output.append(line)
        if model != "OpenAI":
            output.append("-" * 100)

    return "\n".join(output)


def main():
    # Sample colloquial medical terms
    search_terms = [
        "broken arm",
        "pink eye",
        "stomach flu",
        "chicken pox",
        "kidney stones",
        "lung infection",
        "blood poisoning",
        "UTI",
        "Acid reflux",
        "bed sores",
        "blood clot",
        "bone spurs",
        "cold sores",
        "collapsed lungs",
        "dry eyes",
        "hives",
        "kidney infections",
        "sinus infection",
        "shingles",
        "ringing in ears",
        "stomach ulcer",
        "yeast infection",
        "cancer",
        "cough",
        "diabetes",
        "diarrhea",
        "ear infection",
        "high blood pressure",
        "heart attack",
        "diabetes type 2",
        "chest pain",
        "migraine headache",
        "anxiety attack",
        "sugar problem with blurry eyes",
        "back pain",
        "piles",
        "headache",
        "cough",
        "cancer",
        "diabetes",
        "stroke",
        "asthma"
    ]

    print("\nComparing semantic search results across different embedding models\n")

    for term in search_terms:
        print(f"\n{'='*80}")
        print(f"Search Term: {term}")
        print(f"{'='*80}\n")

        results = search_across_models(term)
        print(format_results_table(results))
        time.sleep(1)  # Prevent rate limiting


if __name__ == "__main__":
    main()
