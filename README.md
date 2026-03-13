# Research Paper Harvester

Research Paper Harvester is a small Python project that collects research papers from multiple academic sources, expands results through citations, removes redundant entries, scores relevance, and returns a ranked JSON corpus.

## Features

- Fetches seed papers from arXiv, PubMed, and Semantic Scholar
- Expands the corpus using Semantic Scholar citation traversal
- Detects exact and semantic duplicates
- Scores papers against a natural-language query using embeddings
- Ranks papers and outputs a JSON result

## Project Structure

```text
harvester/
├── config/
├── connectors/
├── core/
├── filters/
├── utils/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

- Python 3.10+
- Internet access for API calls

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

Run with the default query:

```powershell
python -m main
```

Run with a custom query:

```powershell
python -m main "transformers for medical image segmentation"
```

## Output

The program prints JSON with:

- `session_id`
- `original_query`
- `corpus`

Each paper in `corpus` includes:

- `paper_id`
- `title`
- `authors`
- `year`
- `source_url`
- `semantic_similarity_score`
- `is_redundant`
- `deflection_flag`

## Notes

- Do not commit `.venv`, `__pycache__`, or other local cache files.
- Install dependencies before running the project.
- The project currently uses assertions in `main.py` for basic runtime validation.
