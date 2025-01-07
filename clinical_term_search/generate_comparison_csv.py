from compare_embeddings_search import search_across_models
import csv
import time


def generate_comparison_csv():
    # Using the same search terms as in compare_embeddings_search.py
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
        "diabetes type 2"
    ]

    # Prepare CSV data
    csv_data = []
    headers = [
        'Search Term',
        'Rank',
        'MiniLM-L6 ICD10',
        'MiniLM-L6 Description',
        'MiniLM-L6 Score',
        'MiniLM-L12 ICD10',
        'MiniLM-L12 Description',
        'MiniLM-L12 Score',
        'OpenAI ICD10',
        'OpenAI Description',
        'OpenAI Score'
    ]

    print("Generating comparison results...")
    for term in search_terms:
        print(f"Processing: {term}")
        results = search_across_models(term)

        # Process all 10 results for each term
        for rank in range(10):
            row = {
                'Search Term': term,
                'Rank': rank + 1,
                'MiniLM-L6 ICD10': results['MiniLM-L6'][rank]['code'],
                'MiniLM-L6 Description': results['MiniLM-L6'][rank]['description'],
                'MiniLM-L6 Score': results['MiniLM-L6'][rank]['score'],
                'MiniLM-L12 ICD10': results['MiniLM-L12'][rank]['code'],
                'MiniLM-L12 Description': results['MiniLM-L12'][rank]['description'],
                'MiniLM-L12 Score': results['MiniLM-L12'][rank]['score'],
                'OpenAI ICD10': results['OpenAI'][rank]['code'],
                'OpenAI Description': results['OpenAI'][rank]['description'],
                'OpenAI Score': results['OpenAI'][rank]['score']
            }
            csv_data.append(row)

        time.sleep(1)  # Prevent rate limiting

    # Write to CSV
    output_file = 'embedding_model_comparison_detailed.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"\nDetailed comparison CSV generated: {output_file}")


if __name__ == "__main__":
    generate_comparison_csv()
