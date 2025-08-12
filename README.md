# Hybrid Search with ColBERT Reranking

This project implements and compares two methods for hybrid search with late-interaction (ColBERT) reranking in Qdrant. The goal is to evaluate the trade-offs between indexing speed and query speed.

## Core Concept: Two Reranking Approaches

The implementation compares two distinct approaches to integrating a ColBERT reranker:

1.  **Native Reranking (`reranking-search` collection)**
    *   **Indexing**: Pre-computes and uploads all embeddings: dense, sparse, and the ColBERT late-interaction multi-vectors.
    *   **Querying**: Uses a single Qdrant `query` API call. Qdrant handles the initial hybrid search and the final ColBERT reranking internally.

2.  **Manual Reranking (`hybrid-search` collection)**
    *   **Indexing**: Pre-computes and uploads only the dense and sparse embeddings.
    *   **Querying**:
        1.  Retrieves candidate documents from Qdrant using a standard hybrid search (dense + sparse).
        2.  The Python client then applies the ColBERT model to rerank these candidates.

## Performance Trade-Off

The primary difference between the two approaches is a trade-off between indexing and query performance.

| Approach         | Indexing Speed (per 1M docs) | Query Speed (per query) |
| ---------------- | ---------------------------- | ----------------------- |
| **Native**       | ~39 hours (~0.14s / doc)      | **~0.06 seconds**       |
| **Manual**       | **~19 hours (~0.07s / doc)** | ~0.75 seconds            |

**Conclusion:** This project proceeds with the **Manual Reranking** approach. The task did not specify query performance requirements, and a 2x faster indexing time was prioritized.

## System Components

*   **Dense Vectors**: `sentence-transformers/all-MiniLM-L6-v2` (384d) with Binary Quantization.
*   **Sparse Vectors**: `Qdrant/bm25` for keyword scoring.
*   **Reranking Model**: `answerdotai/answerai-colbert-small-v1` (late interaction).
*   **Dataset**: `AIR-Bench/qa_msmarco_en` (1M documents).
*   **Qdrant Cluster**: 2 nodes, 3 shards, replication factor of 2.

## How to Run

### 1. Prerequisites

*   Python 3.8+
*   Docker & Docker Compose

### 2. Setup Environment

```bash
# Clone the repository
git clone https://github.com/holyMolyTolli/qdrant-coding-task.git
cd qdrant-coding-task

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Qdrant Cluster

```bash
docker-compose up -d
```

Wait for the cluster to be ready at `http://localhost:6333`.

### 4. Run the Application

The main script will build both collections, run a comparison, and log the results.

```bash
python -m app.main
```

## Code Structure

*   `app/main.py`: Main script to build indexes and run the search comparison.
*   `app/hybrid_search_operations.py`: Contains the core logic for both search approaches:
    *   `hybrid_search_with_native_reranking()`
    *   `hybrid_search_with_manual_reranking()`
*   `app/config.py`: Configuration for models, dataset, and cluster settings.
*   `docker-compose.yml`: Defines the 2-node Qdrant cluster.