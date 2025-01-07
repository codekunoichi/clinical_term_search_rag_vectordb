from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import csv
import time
from tqdm import tqdm
import os

# Initialize Pinecone and model
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
model = SentenceTransformer("all-MiniLM-L12-v2")
index = pc.Index("rxcode-minillm-l12")


def search_medications(query, top_k=10):
    embedding = model.encode(query).tolist()
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches


def generate_rx_results_csv():

    search_terms = [
        # Primary Care
        "Metformin",  # Metformin
        "Glucophage",  # Metformin
        "Lisinopril",  # Lisinopril
        "Prinivil",    # Lisinopril
        "Amlodipine",  # Amlodipine
        "Norvasc",     # Amlodipine
        "Prilosec",    # Omeprazole
        "Synthroid",   # Levothyroxine
        "Zoloft",      # Sertraline
        "Amoxil",      # Amoxicillin
        "ozempic",     # Semaglutide
        "Semaglutide",  # Semaglutide
        "Victoza",     # Semaglutide
        "Byetta",      # Exenatide
        "Bydureon",    # Exenatide
        "Bydureon",    # Exenatide

        # Pain Management
        "Neurontin",   # Gabapentin
        "Ultram",      # Tramadol
        "Cymbalta",    # Duloxetine
        "Lyrica",      # Pregabalin
        "Mobic",       # Meloxicam
        "Flexeril",    # Cyclobenzaprine
        "Iron Supplements",  # Iron Supplements
        "Calcium Supplements",  # Calcium Supplements
        "Magnesium Supplements",  # Magnesium Supplements
        "Vitamin D Supplements",  # Vitamin D Supplements
        "Vitamin B12 Supplements",  # Vitamin B12 Supplements
        "Vitamin A Supplements",  # Vitamin A Supplements
        "Vitamin C Supplements",  # Vitamin C Supplements
        "Vitamin E Supplements",  # Vitamin E Supplements
        "Vitamin K Supplements",  # Vitamin K Supplements

        # Internal Medicine
        "Lipitor",     # Atorvastatin
        "Lopressor",   # Metoprolol
        "Lasix",       # Furosemide
        "Deltasone",   # Prednisone
        "Protonix",    # Pantoprazole
        "Plavix",      # Clopidogrel

        # Cardiology
        "Coumadin",    # Warfarin
        "Eliquis",     # Apixaban
        "Coreg",       # Carvedilol
        "Aldactone",   # Spironolactone
        "Crestor",     # Rosuvastatin
        "Diovan",      # Valsartan

        # Pediatrics
        "Augmentin",   # Amoxicillin/Clavulanate
        "Ventolin",    # Albuterol
        "Flonase",     # Fluticasone
        "Ritalin",     # Methylphenidate
        "Zyrtec",      # Cetirizine
        "Zithromax",   # Azithromycin

        # Additional Common Medications
        "Lexapro",     # Escitalopram
        "Microzide",   # Hydrochlorothiazide (HCTZ)
        "Singulair",   # Montelukast
        "Wellbutrin",  # Bupropion
        "Decadron",    # Dexamethasone
        "Lantus"       # Insulin Glargine
    ]

    # Prepare CSV data
    csv_data = []
    headers = [
        'Search Term',
        'Rank',
        'NDC',
        'Drug Name',
        'Trade Name',
        'Dosage Form',
        'Route',
        'Strength',
        'RxNorm',
        'Score'
    ]

    print("Generating medication search results...")
    for term in tqdm(search_terms):
        print(f"\nProcessing: {term}")
        results = search_medications(term)

        for rank, result in enumerate(results, 1):
            row = {
                'Search Term': term,
                'Rank': rank,
                'NDC': result.metadata['ndc'],
                'Drug Name': result.metadata['drug_name'],
                'Trade Name': result.metadata['trade_name'],
                'Dosage Form': result.metadata['dosage_form'],
                'Route': result.metadata['route'],
                'Strength': result.metadata['strength'],
                'RxNorm': result.metadata['rxnorm'],
                'Score': result.score
            }
            csv_data.append(row)

        time.sleep(0.5)  # Small delay between searches

    # Write to CSV
    output_file = 'rx_search_results_detailed_1.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"\nDetailed medication search results generated: {output_file}")


if __name__ == "__main__":
    generate_rx_results_csv()
