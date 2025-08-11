#!/usr/bin/env python3

import logging
import time
from itertools import islice
from typing import Any, Dict, Generator, List

import numpy as np
from datasets import load_dataset
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (BinaryQuantization, BinaryQuantizationConfig,
                                  Distance, HnswConfigDiff, Modifier,
                                  MultiVectorComparator, MultiVectorConfig,
                                  PointStruct, Prefetch, SparseIndexParams,
                                  SparseVector, SparseVectorParams,
                                  VectorParams)
from tqdm.asyncio import tqdm

from app.config import (DATASET_NAME, DENSE_MODEL_NAME, HNSW_M, N_CPU,
                        REPLICATION_FACTOR, RERANKING_MODEL_NAME, SHARD_NUMBER,
                        SPARSE_MODEL_NAME, SPLIT_NAME, SUBSET_NAME)

logger = logging.getLogger(__file__.split("/")[-1].split(".")[0])
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)


async def recreate_collection(client: AsyncQdrantClient, embedding_models: Dict[str, Any], collection_name: str):
    logger.info(f"Checking for existing Qdrant collection '{collection_name}'...")

    if await client.collection_exists(collection_name=collection_name):
        logger.warning(f"Collection '{collection_name}' already exists. Deleting it.")
        await client.delete_collection(collection_name=collection_name)

    logger.info(f"Creating new Qdrant collection '{collection_name}'...")

    vectors_config = {
        model_name: VectorParams(size=embedding_model.embedding_size, distance=Distance.COSINE, on_disk=True)
        for model_name, embedding_model in embedding_models.items()
        if model_name != SPARSE_MODEL_NAME
    }
    vectors_config[DENSE_MODEL_NAME].quantization_config = BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True))
    if RERANKING_MODEL_NAME in embedding_models:
        vectors_config[RERANKING_MODEL_NAME].multivector_config = MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM)
        vectors_config[RERANKING_MODEL_NAME].hnsw_config = HnswConfigDiff(m=0)

    await client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config={SPARSE_MODEL_NAME: SparseVectorParams(modifier=Modifier.IDF, index=SparseIndexParams(on_disk=True))},
        shard_number=SHARD_NUMBER,
        replication_factor=REPLICATION_FACTOR,
    )


def stream_and_prep_documents(max_docs: int) -> Generator[Dict[str, Any], None, None]:
    dataset = load_dataset(DATASET_NAME, name=SUBSET_NAME, split=SPLIT_NAME, streaming=True)

    for i, example in enumerate(islice(dataset, max_docs)):
        yield {
            "id": i,
            "text": example["text"],
            "metadata": {"dataset": DATASET_NAME, "subset": SUBSET_NAME, "split": SPLIT_NAME, "id": example["id"]},
        }


def batch_generator(generator: Generator, batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
    while batch := list(islice(generator, batch_size)):
        yield batch


async def index_docs_to_collection(client: AsyncQdrantClient, embedding_models: Dict[str, Any], collection_name: str, batch_size: int, max_docs: int):
    overall_start_time = time.time()
    logger.info(f"Streaming dataset '{DATASET_NAME}'...")
    document_stream = stream_and_prep_documents(max_docs)
    batches = batch_generator(document_stream, batch_size)
    num_batches = (max_docs + batch_size - 1) // batch_size

    document_count = 0
    embedding_duration = 0
    upload_duration = 0
    for i, batch_docs in enumerate(tqdm(batches, total=int(num_batches), desc="Indexing Batches")):
        texts_to_embed = [doc["text"] for doc in batch_docs]
        start_time = time.time()
        embeddings = {model_name: list(embedding_model.embed(texts_to_embed, parallel=0)) for model_name, embedding_model in embedding_models.items()}
        embedding_duration += time.time() - start_time

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
        client.upload_points(collection_name=collection_name, points=points_to_upload, wait=False, parallel=N_CPU, batch_size=batch_size)
        upload_duration += time.time() - start_time

        document_count += len(batch_docs)

    total_duration = time.time() - overall_start_time
    logger.info(
        f"✅ Final result: uploaded {document_count} documents to {collection_name} in {total_duration:.2f} seconds ({total_duration/document_count:.2f} seconds per document)"
    )
    logger.info(f"    - Embedding: {embedding_duration:.2f} seconds")
    logger.info(f"    - Upload: {upload_duration:.2f} seconds")


async def finalize_indexing(client: AsyncQdrantClient, collection_name: str):
    logger.info("Resuming indexing for the collection...")
    await client.update_collection(collection_name=collection_name, hnsw_config=HnswConfigDiff(m=HNSW_M))
    logger.info("Indexing is now active and will build in the background.")


async def native_hybrid_search_with_reranking(
    query_text: str, embedding_models: Dict[str, Any], client: AsyncQdrantClient, collection_name: str, limit: int, prefetch_limit: int
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


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    return vectors / norms


def _rerank_candidates(candidates: List[Dict[str, Any]], query_text: str, reranking_model: Any):
    texts_to_embed = [result.payload["text"] for result in candidates]
    reranking_embeddings = list(reranking_model.embed(texts_to_embed, parallel=0))
    normalized_query = _normalize_vectors(next(reranking_model.query_embed(query_text)))
    scored_results = []
    for i, (text, doc_embedding) in enumerate(zip(texts_to_embed, reranking_embeddings)):
        normalized_doc = _normalize_vectors(doc_embedding)
        similarity_matrix = normalized_query @ normalized_doc.T
        max_sim_scores = similarity_matrix.max(axis=1)
        final_score = float(max_sim_scores.sum())
        scored_results.append({"text": text, "score": final_score, "id": candidates[i].id, "payload": candidates[i].payload})
    return scored_results


async def manual_hybrid_search_with_reranking(
    query_text: str, embedding_models: Dict[str, Any], client: AsyncQdrantClient, collection_name: str, limit: int, prefetch_limit: int
):
    dense_query = next(embedding_models[DENSE_MODEL_NAME].query_embed(query_text))
    sparse_query = next(embedding_models[SPARSE_MODEL_NAME].query_embed(query_text))

    prefetch_results_dense = await client.query_points(
        collection_name=collection_name, query=dense_query, with_payload=True, limit=prefetch_limit, using=DENSE_MODEL_NAME
    )
    prefetch_results_sparse = await client.query_points(
        collection_name=collection_name, query=SparseVector(**sparse_query.as_object()), with_payload=True, limit=limit, using=SPARSE_MODEL_NAME
    )

    unique_candidates = prefetch_results_dense.points
    for point in prefetch_results_sparse.points:
        if point.id not in [p.id for p in unique_candidates]:
            unique_candidates.append(point)

    reranking_model = embedding_models[RERANKING_MODEL_NAME]
    scored_results = _rerank_candidates(unique_candidates, query_text, reranking_model)

    return sorted(scored_results, key=lambda x: x["score"], reverse=True)[:limit]
