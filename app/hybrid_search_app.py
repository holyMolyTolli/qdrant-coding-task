#!/usr/bin/env python3

import time
from typing import Any, Dict, List

from datasets import load_dataset
from fastembed import (LateInteractionTextEmbedding, SparseTextEmbedding,
                       TextEmbedding)
from qdrant_client import QdrantClient
from qdrant_client.models import (BinaryQuantization, BinaryQuantizationConfig,
                                  Distance, HnswConfigDiff, Modifier,
                                  MultiVectorComparator, MultiVectorConfig,
                                  PointStruct, Prefetch, SparseVector,
                                  SparseVectorParams, VectorParams)
from tqdm import tqdm

from app.config import (COLLECTION_NAME, DATASET_NAME, DENSE_MODEL_NAME, HOST,
                        RERANKING_MODEL_NAME, MAX_DOCUMENTS, PORT,
                        REPLICATION_FACTOR, SHARD_NUMBER, SPARSE_MODEL_NAME,
                        UPSERT_BATCH_SIZE)


def load_and_prep_documents(dataset_name: str = DATASET_NAME, max_docs: int = MAX_DOCUMENTS) -> List[Dict[str, Any]]:
    print(f"Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split="train")

    documents = []
    for i, example in enumerate(dataset):
        if i >= max_docs:
            break
        text_content = f"Question: {example['question']} Context: {example['context']}"

        answers = example.get("answers", {})
        answer_text = answers.get("text", [""])[0] if answers else ""

        documents.append(
            {
                "id": i,
                "text": text_content,
                "metadata": {
                    "question": example["question"],
                    "context": example["context"],
                    "answer": answer_text,
                    "title": example.get("title", ""),
                    "dataset": dataset_name,
                },
            }
        )
    print(f"Prepared {len(documents)} documents.")
    return documents


def initialize_embedding_models() -> Dict[str, Any]:
    print("Initializing embedding models...")
    return {
        DENSE_MODEL_NAME: TextEmbedding(DENSE_MODEL_NAME),
        SPARSE_MODEL_NAME: SparseTextEmbedding(SPARSE_MODEL_NAME),
        RERANKING_MODEL_NAME: LateInteractionTextEmbedding(RERANKING_MODEL_NAME),
    }


def generate_embeddings(documents: List[Dict[str, Any]], embedding_models: Dict[str, Any]) -> Dict[str, List[Any]]:
    print("Generating embeddings for all documents...")
    texts = [doc["text"] for doc in documents]

    embeddings = {}
    for model_name, model in embedding_models.items():
        embeddings[model_name] = list(model.embed(texts))

    return embeddings


def setup_qdrant_collection(client: QdrantClient, embedding_models: Dict[str, Any]):
    print(f"Setting up Qdrant collection '{COLLECTION_NAME}'...")
    dense_model = embedding_models[DENSE_MODEL_NAME]
    reranking_model = embedding_models[RERANKING_MODEL_NAME]

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_MODEL_NAME: VectorParams(
                size=dense_model.embedding_size,
                distance=Distance.COSINE,
                quantization_config=BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True)),
            ),
            RERANKING_MODEL_NAME: VectorParams(
                size=reranking_model.embedding_size,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
                hnsw_config=HnswConfigDiff(m=0),  # Disable HNSW for this vector
            ),
        },
        sparse_vectors_config={SPARSE_MODEL_NAME: SparseVectorParams(modifier=Modifier.IDF)},
        shard_number=SHARD_NUMBER,
        replication_factor=REPLICATION_FACTOR,
    )
    print("Done!\n")


def _create_point(doc: Dict, embeddings: Dict, index: int) -> PointStruct:

    return PointStruct(
        id=int(doc["id"]),
        vector={
            DENSE_MODEL_NAME: embeddings[DENSE_MODEL_NAME][index],
            SPARSE_MODEL_NAME: embeddings[SPARSE_MODEL_NAME][index].as_object(),
            RERANKING_MODEL_NAME: embeddings[RERANKING_MODEL_NAME][index],
        },
        payload={"text": doc["text"], **doc["metadata"]},
    )


def index_to_qdrant(client: QdrantClient, documents: List[Dict[str, Any]], embeddings: Dict[str, List[Any]], batch_size: int = UPSERT_BATCH_SIZE):
    print("Indexing documents into Qdrant...")
    for i in tqdm(range(0, len(documents), batch_size), desc="Indexing Batches"):
        batch_docs = documents[i : i + batch_size]

        points = [_create_point(doc, embeddings, i + doc_idx) for doc_idx, doc in enumerate(batch_docs)]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("Done!\n")


def hybrid_search_with_reranking(query: str, embedding_models: Dict[str, Any], client: QdrantClient, limit: int = 10):
    dense_query = next(embedding_models[DENSE_MODEL_NAME].query_embed(query))
    sparse_query = next(embedding_models[SPARSE_MODEL_NAME].query_embed(query))
    late_query = next(embedding_models[RERANKING_MODEL_NAME].query_embed(query))

    prefetch = [
        Prefetch(query=dense_query, using=DENSE_MODEL_NAME, limit=20),
        Prefetch(query=SparseVector(**sparse_query.as_object()), using=SPARSE_MODEL_NAME, limit=20),
    ]

    results = client.query_points(
        collection_name=COLLECTION_NAME, prefetch=prefetch, query=late_query, using=RERANKING_MODEL_NAME, with_payload=True, limit=limit
    )

    return results


def search_examples(client: QdrantClient, embedding_models: Dict[str, Any]):
    example_queries = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
        "When was the Declaration of Independence signed?",
        "What is the chemical symbol for gold?",
    ]

    for query in example_queries:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print(f"{'='*60}")

        start_time = time.time()
        results = hybrid_search_with_reranking(query, embedding_models, client, limit=5)
        search_time = time.time() - start_time

        print(f"⏱️  Search completed in {search_time:.3f} seconds")
        print(f"📄 Found {len(results.points)} results:")

        for i, result in enumerate(results.points, 1):
            print(f"\n{i}. Score: {result.score:.4f}")
            print(f"   Question: {result.payload.get('question', 'N/A')}")
            print(f"   Answer: {result.payload.get('answer', 'N/A')}")
            print(f"   Context: {result.payload.get('context', 'N/A')[:200]}...")


def setup_and_index(client: QdrantClient, embedding_models: Dict[str, Any]):
    setup_qdrant_collection(client, embedding_models)
    documents = load_and_prep_documents()
    embeddings = generate_embeddings(documents, embedding_models)
    index_to_qdrant(client, documents, embeddings)


def main():
    client = QdrantClient(url=f"http://{HOST}:{PORT}")
    embedding_models = initialize_embedding_models()

    setup_and_index(client, embedding_models)
    search_examples(client, embedding_models)


if __name__ == "__main__":
    main()
