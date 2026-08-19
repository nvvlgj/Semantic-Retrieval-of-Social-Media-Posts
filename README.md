# Semantic Retrieval of Social Media Posts

A semantic search pipeline for finding documents that are most relevant to a given theme.

The project uses a pretrained Sentence Transformer model to represent documents as embeddings and retrieves relevant documents using cosine similarity.

## Approach

The pipeline has four main stages:

```text
Raw Dataset
    ↓
Preprocessing & Deduplication
    ↓
Embedding Generation
    ↓
Semantic Retrieval
```

### 1. Preprocessing

The dataset is cleaned before generating embeddings.

The preprocessing includes:

- Removing missing or empty documents
- Unicode and whitespace normalization
- URL removal
- Exact duplicate removal
- Lexical near-duplicate detection using TF-IDF and cosine similarity
- Semantic deduplication using sentence embeddings

The semantic deduplication step helps reduce highly similar documents that may describe the same story using different wording.

### 2. Embedding Generation

Documents are converted into dense vector representations using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings are normalized and stored along with the corresponding document metadata.

### 3. Semantic Retrieval

The search theme is converted into an embedding using the same model.

`scikit-learn`'s `NearestNeighbors` is then used with cosine distance to find the most relevant documents. Results are ranked using cosine similarity.

### 4. Evaluation

The retrieval pipeline saves the Top-K results along with basic similarity statistics:

- Highest similarity
- Lowest similarity
- Average similarity
- Median similarity

The retrieved documents can also be manually inspected to evaluate their relevance to the search theme.

## Project Structure

```text
├── data/
├── results/
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── semantic_deduplicator.py
│   ├── embedding.py
│   ├── retrieval.py
│   └── pipeline.py
├── run.py
├── requirements.txt
└── README.md
```

## Running the Project

Clone the repository:

```bash
git clone https://github.com/nvvlgj/Semantic-Retrieval-of-Social-Media-Posts.git
cd Semantic-Retrieval-of-Social-Media-Posts
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python run.py
```

This runs preprocessing, deduplication, embedding generation, and retrieval in sequence.

## Output

The pipeline generates:

```text
data/
├── processed/
├── embeddings/
└── analysis/

results/
```

The embedding stage stores the embedding bank and metadata, while the retrieval stage produces the ranked results and evaluation statistics.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- PyTorch

## Future Improvements

Some areas I'd explore further include:

- Using the article title and an excerpt instead of the full article for embeddings
- Replacing nearest-neighbor-based semantic deduplication with radius-based search or clustering
- Selecting a better representative when multiple documents describe the same story
- Adding diversity-aware ranking to reduce similar results in the Top-K
- Making the search theme configurable
- Adding more structured logging and error handling

## Author

**Lalitha Nandiraju**

M.S. Computer Science  
University of Missouri–Kansas City
