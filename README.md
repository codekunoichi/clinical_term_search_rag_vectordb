# Clinical Term Search using RAG and Vector Databases
Experiments with RAG, VectorDB, and Semantic Search for Clinical Terms (ICD-10, CPT, and Medications)

## Overview
This project implements semantic search capabilities for three key medical coding systems:
- ICD-10 Diagnosis Codes
- CPT Procedure Codes 
- Medication/Drug Codes (NDC & RxNorm)

## Workflow
![alt text](image.png)

## Models Used
- OpenAI Embeddings (text-embedding-ada-002)
- Sentence Transformers (all-MiniLM-L12-v2)

## Components

### 1. ICD-10 Diagnosis Code Search
- Index names: `icd10-minillm-l12` and `icd10-openai`
- Example search terms:
  - "chest pain"
  - "type 2 diabetes"
  - "high blood pressure"
  - "anxiety"
  - "depression"

### 2. CPT Procedure Code Search
- Index names: `cpt-minillm-l12` and `cpt-openai`
- Example search terms:
  - "appendectomy"
  - "knee replacement"
  - "colonoscopy" 
  - "cataract surgery"
  - "heart bypass"

### 3. Medication Search
- Index name: `rxcode-minillm-l12`
- Example search terms:
  - Brand names: "Glucophage", "Prinivil", "Norvasc"
  - Generic names: "Metformin", "Lisinopril", "Amlodipine"
  - Common names: "blood pressure medicine", "diabetes medicine"

## Setup Instructions

### Environment Setup
1. Clone the repository
2. Set up virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Set PYTHONPATH:
```bash
export PYTHONPATH=$(pwd)
```

### API Keys Setup
Set environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Loading Data

### Load CPT Data
```bash
python3 clinical_term_search/load_cpt_minillm_L12_embeddings.py
python3 clinical_term_search/load_cpt_openai_embeddings.py
```

### Load ICD-10 Data
```bash
python3 clinical_term_search/load_icd10_minillm_L12_embeddings.py
python3 clinical_term_search/load_icd10_openai_embeddings.py
```

### Load Medication Data
```bash
python3 clinical_term_search/load_rx_minillm_L12_embeddings.py
```

## Generating Comparison Results
```bash
python3 clinical_term_search/generate_cpt_comparison_csv.py
python3 clinical_term_search/generate_rx_search_results.py
```

## Troubleshooting Tips
- CSV file formatting:
  - Avoid double quotes in content if using double quotes as field delimiters
  - Avoid commas in content if using commas as field separators
  - Avoid newlines in content
- Data cleaning SQL example:
```sql
SELECT
   Code,
   REPLACE(Name, '"', '') AS CleanedName,
   REPLACE(Description, '"', '') AS CleanedDescription
FROM
   CPTTable;
```

## Progress Tracking
Using `tqdm` for progress bars:
- tqdm means "progress" in Arabic (taqadum, تقدّم)
- Documentation: [tqdm.github.io](https://tqdm.github.io/)
