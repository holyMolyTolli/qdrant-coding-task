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

from app.config import (BATCH_SIZE, COLLECTION_NAME,
                        COLLECTION_NAME_SIMPLIFIED, DATASET_NAME,
                        DENSE_MODEL_NAME, HNSW_M, HOST, MAX_DOCUMENTS, PORT,
                        PREFETCH_LIMIT, REPLICATION_FACTOR,
                        RERANKING_MODEL_NAME, SHARD_NUMBER, SPARSE_MODEL_NAME)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def initialize_embedding_models() -> Dict[str, Any]:
    logging.info("Initializing embedding models...")
    return {
        DENSE_MODEL_NAME: TextEmbedding(DENSE_MODEL_NAME),
        SPARSE_MODEL_NAME: SparseTextEmbedding(SPARSE_MODEL_NAME),
        RERANKING_MODEL_NAME: LateInteractionTextEmbedding(RERANKING_MODEL_NAME),
    }


async def create_or_recreate_collection(client: AsyncQdrantClient, embedding_models: Dict[str, Any], collection_name: str):
    logging.info(f"Checking for existing Qdrant collection '{collection_name}'...")

    if await client.collection_exists(collection_name=collection_name):
        logging.warning(f"Collection '{collection_name}' already exists. Deleting it.")
        await client.delete_collection(collection_name=collection_name)

    logging.info(f"Creating new Qdrant collection '{collection_name}'...")
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


async def create_or_recreate_simplified_collection(client: AsyncQdrantClient, embedding_models: Dict[str, Any], collection_name: str):
    logging.info(f"Checking for existing Qdrant collection '{collection_name}'...")

    if await client.collection_exists(collection_name=collection_name):
        logging.warning(f"Collection '{collection_name}' already exists. Deleting it.")
        await client.delete_collection(collection_name=collection_name)

    logging.info(f"Creating new Qdrant collection '{collection_name}'...")
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
    logging.info(f"Loading and streaming dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    for i, example in enumerate(islice(dataset, max_docs)):
        answer_text = " | ".join(example.get("answers", {}).get("text", [""]))
        text_content = f"Question: {example['question']} Answer: {answer_text}"

        yield {"id": i, "text": text_content, "metadata": {"question": example["question"], "answer": answer_text, "dataset": dataset_name}}


def batch_generator(generator: Generator, batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
    while batch := list(islice(generator, batch_size)):
        yield batch


async def index_documents_to_qdrant(client: AsyncQdrantClient, embedding_models: Dict[str, Any], collection_name: str):
    overall_start_time = time.time()
    document_stream = stream_and_prep_documents()
    batches = batch_generator(document_stream, BATCH_SIZE)
    num_batches = (MAX_DOCUMENTS + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_docs in tqdm(batches, total=int(num_batches), desc="Indexing Batches"):
        texts_to_embed = [doc["text"] for doc in batch_docs]

        start_time = time.time()
        dense_embeddings = list(embedding_models[DENSE_MODEL_NAME].embed(texts_to_embed, parallel=0))
        sparse_embeddings = list(embedding_models[SPARSE_MODEL_NAME].embed(texts_to_embed, parallel=0))
        reranking_embeddings = list(embedding_models[RERANKING_MODEL_NAME].embed(texts_to_embed, parallel=0))
        logging.info(f"Embedding time: {time.time() - start_time:.2f} seconds")

        points_to_upload = []
        for i, doc in enumerate(batch_docs):

            points_to_upload.append(
                PointStruct(
                    id=doc["id"],
                    payload={"text": doc["text"], **doc["metadata"]},
                    vector={
                        DENSE_MODEL_NAME: dense_embeddings[i],
                        SPARSE_MODEL_NAME: sparse_embeddings[i].as_object(),
                        RERANKING_MODEL_NAME: reranking_embeddings[i],
                    },
                )
            )

        start_time = time.time()
        client.upload_points(collection_name=collection_name, points=points_to_upload, wait=False, parallel=8, batch_size=BATCH_SIZE)
        logging.info(f"Upload time: {time.time() - start_time:.2f} seconds")

    logging.info(
        f"✅ Successfully uploaded {MAX_DOCUMENTS} large embeddings in {(time.time() - overall_start_time)/MAX_DOCUMENTS:.2f} seconds per document"
    )


async def index_simplified_documents_to_qdrant(client: AsyncQdrantClient, embedding_models: Dict[str, Any], collection_name: str):
    overall_start_time = time.time()
    document_stream = stream_and_prep_documents()
    batches = batch_generator(document_stream, BATCH_SIZE)
    num_batches = (MAX_DOCUMENTS + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_docs in tqdm(batches, total=int(num_batches), desc="Indexing Batches"):
        texts_to_embed = [doc["text"] for doc in batch_docs]

        start_time = time.time()
        dense_embeddings = list(embedding_models[DENSE_MODEL_NAME].embed(texts_to_embed, parallel=0))
        sparse_embeddings = list(embedding_models[SPARSE_MODEL_NAME].embed(texts_to_embed, parallel=0))
        logging.info(f"Embedding time: {time.time() - start_time:.2f} seconds")

        points_to_upload = []
        for i, doc in enumerate(batch_docs):

            points_to_upload.append(
                PointStruct(
                    id=doc["id"],
                    payload={"text": doc["text"], **doc["metadata"]},
                    vector={DENSE_MODEL_NAME: dense_embeddings[i], SPARSE_MODEL_NAME: sparse_embeddings[i].as_object()},
                )
            )

        start_time = time.time()
        client.upload_points(collection_name=collection_name, points=points_to_upload, wait=False, parallel=8, batch_size=BATCH_SIZE)
        logging.info(f"Upload time: {time.time() - start_time:.2f} seconds")

    logging.info(
        f"✅ Successfully uploaded {MAX_DOCUMENTS} SIMPLIFIED embeddings in {(time.time() - overall_start_time)/MAX_DOCUMENTS:.2f} seconds per document"
    )


async def finalize_indexing(client: AsyncQdrantClient, collection_name: str):
    logging.info("Resuming indexing for the collection...")
    await client.update_collection(collection_name=collection_name, hnsw_config=HnswConfigDiff(m=HNSW_M))
    logging.info("Indexing is now active and will build in the background.")


async def hybrid_search_with_reranking(
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


async def simplified_hybrid_search_with_reranking(
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


# async def search_examples(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
#     example_queries = [
#         "What is the capital of France?",
#         "Who wrote Romeo and Juliet?",
#         "What is the largest planet in our solar system?",
#         "When was the Declaration of Independence signed?",
#         "What is the chemical symbol for gold?",
#     ]

#     non_matching_results = []
#     average_search_time = 0
#     average_simplified_search_time = 0
#     for query_text in example_queries:
#         logging.info(f"{'='*60}")
#         logging.info(f"🔍 Query: {query_text}")
#         logging.info(f"{'='*60}")

#         start_time = time.time()
#         results = await hybrid_search_with_reranking(query_text, embedding_models, client, COLLECTION_NAME, limit=5)
#         search_time = time.time() - start_time

#         logging.info(f"{'-'*30}")
#         logging.info(f"⏱️  Search completed in {search_time:.3f} seconds")
#         logging.info(f"{'-'*30}")

#         for i, result in enumerate(results, 1):
#             logging.info(f"{i}. Score: {result['score']:.4f}")
#             logging.info(f"   Question: {result['payload'].get('question', 'N/A')}")
#             logging.info(f"   Answer: {result['payload'].get('answer', 'N/A')}")

#         average_search_time += search_time

#         start_time = time.time()
#         simplified_results = await simplified_hybrid_search_with_reranking(query_text, embedding_models, client, COLLECTION_NAME_SIMPLIFIED, limit=5)
#         search_time = time.time() - start_time

#         logging.info(f"{'-'*30}")
#         logging.info(f"⏱️  Simplified search completed in {search_time:.3f} seconds")
#         logging.info(f"{'-'*30}")

#         for i, result in enumerate(simplified_results, 1):
#             logging.info(f"{i}. Score: {result['score']:.4f}")
#             logging.info(f"   Question: {result['payload'].get('question', 'N/A')}")
#             logging.info(f"   Answer: {result['payload'].get('answer', 'N/A')}")

#         average_simplified_search_time += search_time

#         # compare results between normal and simplified search
#         for i, result in enumerate(results):
#             if result["id"] != simplified_results[i]["id"]:
#                 non_matching_results.append({"query": query_text, "rank": i, "normal_result": result, "simplified_result": simplified_results[i]})

#     logging.info(f"{'='*60}")
#     logging.info("Comparison of search times between normal and simplified search")
#     logging.info(f"{'='*60}")
#     logging.info(f"Average search time: {average_search_time / len(example_queries):.3f} seconds")
#     logging.info(f"Average simplified search time: {average_simplified_search_time / len(example_queries):.3f} seconds")

#     logging.info(f"{'='*60}")
#     logging.info("Comparison of results between normal and simplified search")
#     logging.info(f"{'='*60}")
#     if len(non_matching_results) > 0:
#         logging.info(f"{len(non_matching_results)} non-matching results")
#         current_query = None
#         for result in non_matching_results:
#             if current_query != result["query"]:
#                 current_query = result["query"]
#                 logging.info(f"{'-'*60}")
#                 logging.info(f"Query: {result['query']}")
#                 logging.info(f"{'-'*60}")
#             logging.info(f"   Rank:              {result['rank']}")
#             logging.info(f"   Normal result:     {result['normal_result']['payload']['question']}")
#             logging.info(f"   Simplified result: {result['simplified_result']['payload']['question']}")
#             logging.info(f"   {'-'*60}")
#     else:
#         logging.info("All results match")


# --- Helper Functions for Logging ---


def _log_search_results(title: str, results: List[Dict[str, Any]], search_time: float):
    """Logs the title, time, and formatted results of a search."""
    logging.info(f"\n--- {title} (took {search_time:.3f}s) ---")
    if not results:
        logging.info("  No results found.")
        return
    for i, result in enumerate(results, 1):
        score = result.get("score", 0.0)
        question = result.get("payload", {}).get("question", "N/A")
        result.get("payload", {}).get("answer", "N/A")
        logging.info(f"  {i}. Score: {score:<6.4f} | Q: {question}")


def _log_summary(total_time: float, total_simplified_time: float, non_matching_results: List[Dict], query_count: int):
    """Logs the final summary report comparing the two search methods."""
    logging.info("=" * 60)
    logging.info("📊 FINAL SUMMARY")
    logging.info("=" * 60)

    logging.info("Average Search Times:")
    logging.info(f"  - Normal:     {total_time / query_count:.3f} seconds")
    logging.info(f"  - Simplified: {total_simplified_time / query_count:.3f} seconds")
    logging.info("-" * 60)

    logging.info("Result Comparison:")
    if not non_matching_results:
        logging.info("  ✅ All results match.")
        return

    logging.warning(f"  ⚠️ Found {len(non_matching_results)} non-matching results:")
    current_query = None
    for result in non_matching_results:
        if current_query != result["query"]:
            current_query = result["query"]
            logging.info(f"Query: '{current_query}'")

        logging.info(f"    - Rank {result['rank']+1}:")
        logging.info(f"        Normal:     (ID: {result['normal_result']['id']}) {result['normal_result']['payload']['question']}")
        logging.info(f"        Simplified: (ID: {result['simplified_result']['id']}) {result['simplified_result']['payload']['question']}")


async def search_examples(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
    example_queries = ["What is the capital of France?", "Who wrote Romeo and Juliet?", "What is the largest planet in our solar system?"]

    non_matching_results = []
    total_search_time = 0
    total_simplified_search_time = 0

    for query_text in example_queries:
        logging.info(f"\n{'='*25} 🔍 Query: {query_text} {'='*25}")

        start_time = time.time()
        results = await hybrid_search_with_reranking(query_text, embedding_models, client, COLLECTION_NAME, limit=5)
        search_time = time.time() - start_time
        total_search_time += search_time
        _log_search_results("Normal Search Results", results, search_time)

        start_time = time.time()
        simplified_results = await simplified_hybrid_search_with_reranking(query_text, embedding_models, client, COLLECTION_NAME_SIMPLIFIED, limit=5)
        simplified_search_time = time.time() - start_time
        total_simplified_search_time += simplified_search_time
        _log_search_results("Simplified Search Results", simplified_results, simplified_search_time)

        for i, (res_norm, res_simp) in enumerate(zip(results, simplified_results)):
            if res_norm["id"] != res_simp["id"]:
                non_matching_results.append({"query": query_text, "rank": i, "normal_result": res_norm, "simplified_result": res_simp})

    _log_summary(total_search_time, total_simplified_search_time, non_matching_results, len(example_queries))


async def setup_and_index(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
    await create_or_recreate_collection(client, embedding_models, COLLECTION_NAME)
    await index_documents_to_qdrant(client, embedding_models, COLLECTION_NAME)
    await finalize_indexing(client, COLLECTION_NAME)
    logging.info("Setup and indexing complete")


async def setup_and_index_simplified(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
    await create_or_recreate_simplified_collection(client, embedding_models, COLLECTION_NAME_SIMPLIFIED)
    await index_simplified_documents_to_qdrant(client, embedding_models, COLLECTION_NAME_SIMPLIFIED)
    await finalize_indexing(client, COLLECTION_NAME_SIMPLIFIED)
    logging.info("Setup and indexing complete")


async def main():
    client = AsyncQdrantClient(url=f"http://{HOST}:{PORT}", timeout=600, prefer_grpc=True)
    embedding_models = initialize_embedding_models()

    await setup_and_index(client, embedding_models)
    await setup_and_index_simplified(client, embedding_models)
    await search_examples(client, embedding_models)


if __name__ == "__main__":
    asyncio.run(main())
