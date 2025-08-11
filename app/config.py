HOST = "localhost"
PORT = 6333

RERANKING_COLLECTION_NAME = "reranking-search"
HYBRID_COLLECTION_NAME = "hybrid-search"

DATASET_NAME = "AIR-Bench/qa_msmarco_en"
SUBSET_NAME = "AIR-Bench_24.04"
SPLIT_NAME = "corpus_default"
BATCH_SIZE = 128
MAX_DOCUMENTS = 1000000

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKING_MODEL_NAME = "answerdotai/answerai-colbert-small-v1"

REPLICATION_FACTOR = 2
SHARD_NUMBER = REPLICATION_FACTOR * 2
HNSW_M = 16

PREFETCH_LIMIT = 20

N_CPU = 12
