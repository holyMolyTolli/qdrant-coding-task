HOST = "localhost"
PORT = 6333

COLLECTION_NAME = "hybrid-search"
COLLECTION_NAME_SIMPLIFIED = "hybrid-search-simplified"
BATCH_SIZE = int(128 / 2)
REPLICATION_FACTOR = 2
SHARD_NUMBER = REPLICATION_FACTOR * 2

MAX_DOCUMENTS = BATCH_SIZE * 100
DATASET_NAME = "squad"

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"
# RERANKING_MODEL_NAME = "colbert-ir/colbertv2.0" # 1.2GB model
RERANKING_MODEL_NAME = "answerdotai/answerai-colbert-small-v1"  # 130MB model

HNSW_M = 16

PREFETCH_LIMIT = 20
