#!/usr/bin/env python3

import asyncio
import logging
import time
from itertools import islice
from typing import Any, Dict, Generator, List

import numpy as np
from datasets import load_dataset
from fastembed import (LateInteractionTextEmbedding, SparseTextEmbedding,
                       TextEmbedding)
from fastembed.late_interaction import LateInteractionTextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (BinaryQuantization, BinaryQuantizationConfig,
                                  Distance, HnswConfigDiff, Modifier,
                                  MultiVectorComparator, MultiVectorConfig,
                                  PointStruct, Prefetch, SparseIndexParams,
                                  SparseVector, SparseVectorParams,
                                  VectorParams)
from tqdm.asyncio import tqdm

from app.config import (BATCH_SIZE, DATASET_NAME, DENSE_MODEL_NAME, HNSW_M,
                        HOST, HYBRID_COLLECTION_NAME, MAX_DOCUMENTS, PORT,
                        PREFETCH_LIMIT, REPLICATION_FACTOR,
                        RERANKING_COLLECTION_NAME, RERANKING_MODEL_NAME,
                        SHARD_NUMBER, SPARSE_MODEL_NAME)

logger = logging.getLogger(__file__.split("/")[-1].split(".")[0])
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)


def initialize_embedding_models() -> Dict[str, Any]:
    logger.info("Initializing embedding models...")
    return {
        DENSE_MODEL_NAME: TextEmbedding(DENSE_MODEL_NAME),
        SPARSE_MODEL_NAME: SparseTextEmbedding(SPARSE_MODEL_NAME),
        RERANKING_MODEL_NAME: LateInteractionTextEmbedding(RERANKING_MODEL_NAME),
    }


async def recreate_reranking_collection(client: AsyncQdrantClient, embedding_models: Dict[str, Any], collection_name: str):
    logger.info(f"Checking for existing Qdrant collection '{collection_name}'...")

    if await client.collection_exists(collection_name=collection_name):
        logger.warning(f"Collection '{collection_name}' already exists. Deleting it.")
        await client.delete_collection(collection_name=collection_name)

    logger.info(f"Creating new Qdrant collection '{collection_name}'...")
    dense_model = embedding_models[DENSE_MODEL_NAME]
    reranking_model = embedding_models[RERANKING_MODEL_NAME]

    await client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_MODEL_NAME: VectorParams(
                size=dense_model.embedding_size,
                distance=Distance.COSINE,
                quantization_config=BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True)),
                hnsw_config=HnswConfigDiff(m=0),
                on_disk=True,
            ),
            RERANKING_MODEL_NAME: VectorParams(
                size=reranking_model.embedding_size,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
                hnsw_config=HnswConfigDiff(m=0),
                on_disk=True,
            ),
        },
        sparse_vectors_config={SPARSE_MODEL_NAME: SparseVectorParams(modifier=Modifier.IDF, index=SparseIndexParams(on_disk=True))},
        shard_number=SHARD_NUMBER,
        replication_factor=REPLICATION_FACTOR,
    )


async def recreate_hybrid_collection(client: AsyncQdrantClient, embedding_models: Dict[str, Any], collection_name: str):
    logger.info(f"Checking for existing Qdrant collection '{collection_name}'...")

    if await client.collection_exists(collection_name=collection_name):
        logger.warning(f"Collection '{collection_name}' already exists. Deleting it.")
        await client.delete_collection(collection_name=collection_name)

    logger.info(f"Creating new Qdrant collection '{collection_name}'...")
    dense_model = embedding_models[DENSE_MODEL_NAME]

    await client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_MODEL_NAME: VectorParams(
                size=dense_model.embedding_size,
                distance=Distance.COSINE,
                quantization_config=BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True)),
                hnsw_config=HnswConfigDiff(m=0),
                on_disk=True,
            )
        },
        sparse_vectors_config={SPARSE_MODEL_NAME: SparseVectorParams(modifier=Modifier.IDF, index=SparseIndexParams(on_disk=True))},
        shard_number=SHARD_NUMBER,
        replication_factor=REPLICATION_FACTOR,
    )


def stream_and_prep_documents(dataset_name: str = DATASET_NAME, max_docs: int = MAX_DOCUMENTS) -> Generator[Dict[str, Any], None, None]:
    logger.info(f"Loading and streaming dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    for i, example in enumerate(islice(dataset, max_docs)):
        answer_text = " | ".join(example.get("answers", {}).get("text", [""]))
        text_content = f"Question: {example['question']} Answer: {answer_text}"

        yield {"id": i, "text": text_content, "metadata": {"question": example["question"], "answer": answer_text, "dataset": dataset_name}}


def batch_generator(generator: Generator, batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
    while batch := list(islice(generator, batch_size)):
        yield batch


async def index_docs_to_collection(client: AsyncQdrantClient, embedding_models: Dict[str, Any], collection_name: str):
    overall_start_time = time.time()
    document_stream = stream_and_prep_documents()
    batches = batch_generator(document_stream, BATCH_SIZE)
    num_batches = (MAX_DOCUMENTS + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_docs in tqdm(batches, total=int(num_batches), desc="Indexing Batches"):
        texts_to_embed = [doc["text"] for doc in batch_docs]

        start_time = time.time()
        embeddings = {model_name: list(embedding_model.embed(texts_to_embed, parallel=0)) for model_name, embedding_model in embedding_models.items()}
        logger.info(f"Embedding time: {time.time() - start_time:.2f} seconds")

        points_to_upload = []
        for i, doc in enumerate(batch_docs):
            points_to_upload.append(
                PointStruct(
                    id=doc["id"],
                    payload={"text": doc["text"], **doc["metadata"]},
                    vector={
                        model_name: embedding[i].as_object() if model_name == SPARSE_MODEL_NAME else embedding[i]
                        for model_name, embedding in embeddings.items()
                    },
                )
            )

        start_time = time.time()
        client.upload_points(collection_name=collection_name, points=points_to_upload, wait=False, parallel=8, batch_size=BATCH_SIZE)
        logger.info(f"Upload time: {time.time() - start_time:.2f} seconds")

    total_duration = time.time() - overall_start_time
    logger.info(
        f"✅ Successfully uploaded {MAX_DOCUMENTS} to {collection_name} in {total_duration:.2f} seconds ({total_duration/MAX_DOCUMENTS:.2f} seconds per document)"
    )




async def finalize_indexing(client: AsyncQdrantClient, collection_name: str):
    logger.info("Resuming indexing for the collection...")
    await client.update_collection(collection_name=collection_name, hnsw_config=HnswConfigDiff(m=HNSW_M))
    logger.info("Indexing is now active and will build in the background.")


async def native_hybrid_search_with_reranking(
    query_text: str,
    embedding_models: Dict[str, Any],
    client: AsyncQdrantClient,
    collection_name: str,
    limit: int = 10,
    prefetch_limit: int = PREFETCH_LIMIT,
):
    dense_query = next(embedding_models[DENSE_MODEL_NAME].query_embed(query_text))
    sparse_query = next(embedding_models[SPARSE_MODEL_NAME].query_embed(query_text))
    late_query = next(embedding_models[RERANKING_MODEL_NAME].query_embed(query_text))

    prefetch = [
        Prefetch(query=dense_query, using=DENSE_MODEL_NAME, limit=prefetch_limit),
        Prefetch(query=SparseVector(**sparse_query.as_object()), using=SPARSE_MODEL_NAME, limit=prefetch_limit),
    ]

    results = await client.query_points(
        collection_name=collection_name, prefetch=prefetch, query=late_query, using=RERANKING_MODEL_NAME, with_payload=True, limit=limit
    )

    return [dict(result) for result in results.points]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    return vectors / norms


async def manual_hybrid_search_with_reranking(
    query_text: str,
    embedding_models: Dict[str, Any],
    client: AsyncQdrantClient,
    collection_name: str,
    limit: int = 10,
    prefetch_limit: int = PREFETCH_LIMIT,
):
    dense_query = next(embedding_models[DENSE_MODEL_NAME].query_embed(query_text))
    sparse_query = next(embedding_models[SPARSE_MODEL_NAME].query_embed(query_text))
    late_query = next(embedding_models[RERANKING_MODEL_NAME].query_embed(query_text))

    prefetch_results_dense = await client.query_points(
        collection_name=collection_name, query=dense_query, with_payload=True, limit=prefetch_limit, using=DENSE_MODEL_NAME
    )
    prefetch_results_sparse = await client.query_points(
        collection_name=collection_name,
        query=SparseVector(**sparse_query.as_object()),
        with_payload=True,
        limit=prefetch_limit,
        using=SPARSE_MODEL_NAME,
    )

    prefetch_results = prefetch_results_dense.points
    for point in prefetch_results_sparse.points:
        if point.id not in [p.id for p in prefetch_results]:
            prefetch_results.append(point)

    texts_to_embed = [result.payload["text"] for result in prefetch_results]
    reranking_embeddings = list(embedding_models[RERANKING_MODEL_NAME].embed(texts_to_embed, parallel=0))

    scored_results = []
    for i, (text, doc_embedding) in enumerate(zip(texts_to_embed, reranking_embeddings)):
        normalized_query = normalize_vectors(late_query)
        normalized_doc = normalize_vectors(doc_embedding)
        similarity_matrix = normalized_query @ normalized_doc.T
        max_sim_scores = similarity_matrix.max(axis=1)
        final_score = float(max_sim_scores.sum())
        scored_results.append({"text": text, "score": final_score, "id": prefetch_results[i].id, "payload": prefetch_results[i].payload})

    results = sorted(scored_results, key=lambda x: x["score"], reverse=True)[:limit]

    return results


# --- Helper Functions for logger ---


def _log_search_results(title: str, results: List[Dict[str, Any]], search_time: float):
    """Logs the title, time, and formatted results of a search."""
    logger.info(f"\n--- {title} (took {search_time:.3f}s) ---")
    if not results:
        logger.info("  No results found.")
        return
    for i, result in enumerate(results, 1):
        score = result.get("score", 0.0)
        question = result.get("payload", {}).get("question", "N/A")
        result.get("payload", {}).get("answer", "N/A")
        logger.info(f"  {i}. Score: {score:<6.4f} | Q: {question}")


def _log_summary(total_native_search_time: float, total_manual_search_time: float, non_matching_results: List[Dict], query_count: int):
    """Logs the final summary report comparing the two search methods."""
    logger.info("=" * 60)
    logger.info("📊 FINAL SUMMARY")
    logger.info("=" * 60)

    logger.info("Average Search Times:")
    logger.info(f"  - Native: {total_native_search_time / query_count:.3f} seconds")
    logger.info(f"  - Manual: {total_manual_search_time / query_count:.3f} seconds")
    logger.info("-" * 60)

    logger.info("Result Comparison:")
    if not non_matching_results:
        logger.info("  ✅ All results match.")
        return

    logger.warning(f"  ⚠️ Found {len(non_matching_results)} non-matching results:")
    current_query = None
    for result in non_matching_results:
        if current_query != result["query"]:
            current_query = result["query"]
            logger.info(f"Query: '{current_query}'")

        logger.info(f"    - Rank {result['rank']+1}:")
        logger.info(f"        Native: (ID: {result['native_results']['id']}) {result['native_results']['payload']['question']}")
        logger.info(f"        Manual: (ID: {result['manual_results']['id']}) {result['manual_results']['payload']['question']}")


async def search_examples(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
    example_queries = ["What is the capital of France?", "Who wrote Romeo and Juliet?", "What is the largest planet in our solar system?"]

    non_matching_results = []
    total_native_search_time = 0
    total_manual_search_time = 0

    for query_text in example_queries:
        logger.info(f"\n{'='*25} 🔍 Query: {query_text} {'='*25}")

        start_time = time.time()
        native_results = await native_hybrid_search_with_reranking(query_text, embedding_models, client, RERANKING_COLLECTION_NAME, limit=5)
        native_search_time = time.time() - start_time
        total_native_search_time += native_search_time
        _log_search_results("Native Search Results", native_results, native_search_time)

        start_time = time.time()
        manual_results = await manual_hybrid_search_with_reranking(query_text, embedding_models, client, HYBRID_COLLECTION_NAME, limit=5)
        manual_search_time = time.time() - start_time
        total_manual_search_time += manual_search_time
        _log_search_results("Manual Search Results", manual_results, manual_search_time)

        for i, (res_nati, res_manu) in enumerate(zip(native_results, manual_results)):
            if res_nati["id"] != res_manu["id"]:
                non_matching_results.append({"query": query_text, "rank": i, "native_results": res_nati, "manual_results": res_manu})

    _log_summary(total_native_search_time, total_manual_search_time, non_matching_results, len(example_queries))


async def build_reranking_search_index(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
    await recreate_reranking_collection(client, embedding_models, RERANKING_COLLECTION_NAME)
    await index_docs_to_collection(client, embedding_models, RERANKING_COLLECTION_NAME)
    await finalize_indexing(client, RERANKING_COLLECTION_NAME)
    logger.info("Setup and indexing complete")


async def build_hybrid_search_index(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
    hybrid_embedding_models = {DENSE_MODEL_NAME: embedding_models[DENSE_MODEL_NAME], SPARSE_MODEL_NAME: embedding_models[SPARSE_MODEL_NAME]}
    await recreate_hybrid_collection(client, hybrid_embedding_models, HYBRID_COLLECTION_NAME)
    await index_docs_to_collection(client, hybrid_embedding_models, HYBRID_COLLECTION_NAME)
    await finalize_indexing(client, HYBRID_COLLECTION_NAME)
    logger.info("Setup and indexing complete")


async def main():
    client = AsyncQdrantClient(url=f"http://{HOST}:{PORT}", timeout=600, prefer_grpc=True)
    embedding_models = initialize_embedding_models()

    await build_reranking_search_index(client, embedding_models)
    await build_hybrid_search_index(client, embedding_models)
    await search_examples(client, embedding_models)


if __name__ == "__main__":
    asyncio.run(main())
