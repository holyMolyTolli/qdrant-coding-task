HOST = "localhost"
PORT = 6333
COLLECTION_NAME = "hybrid-search"
UPSERT_BATCH_SIZE = 128  # 128
SHARD_NUMBER = 3
REPLICATION_FACTOR = 2

MAX_DOCUMENTS = 1280  # 1280
DATASET_NAME = "squad"
# DATASET_NAME = "Qdrant/arxiv-titles-instructorxl-embeddings"

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKING_MODEL_NAME = "colbert-ir/colbertv2.0"

HNSW_M = 16
