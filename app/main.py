#!/usr/bin/env python3

import asyncio
import logging
import time
from typing import Any, Dict, List

from fastembed import (LateInteractionTextEmbedding, SparseTextEmbedding,
                       TextEmbedding)
from fastembed.late_interaction import LateInteractionTextEmbedding
from qdrant_client import AsyncQdrantClient

from app.config import (BATCH_SIZE, DENSE_MODEL_NAME, HOST,
                        HYBRID_COLLECTION_NAME, MAX_DOCUMENTS, PORT,
                        PREFETCH_LIMIT, RERANKING_COLLECTION_NAME,
                        RERANKING_MODEL_NAME, SPARSE_MODEL_NAME)
from app.hybrid_search_operations import (finalize_indexing,
                                          index_docs_to_collection,
                                          manual_hybrid_search_with_reranking,
                                          native_hybrid_search_with_reranking,
                                          recreate_collection)

logger = logging.getLogger(__file__.split("/")[-1].split(".")[0])
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)


# --- Helper Functions for logger ---


def initialize_embedding_models() -> Dict[str, Any]:
    logger.info("Initializing embedding models...")
    return {
        DENSE_MODEL_NAME: TextEmbedding(DENSE_MODEL_NAME),
        SPARSE_MODEL_NAME: SparseTextEmbedding(SPARSE_MODEL_NAME),
        RERANKING_MODEL_NAME: LateInteractionTextEmbedding(RERANKING_MODEL_NAME),
    }


async def build_reranking_search_index(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
    await recreate_collection(client, embedding_models, RERANKING_COLLECTION_NAME)
    await index_docs_to_collection(client, embedding_models, RERANKING_COLLECTION_NAME, batch_size=BATCH_SIZE, max_docs=MAX_DOCUMENTS)
    await finalize_indexing(client, RERANKING_COLLECTION_NAME)
    logger.info(f"Setup and indexing complete")


async def build_hybrid_search_index(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
    hybrid_embedding_models = {DENSE_MODEL_NAME: embedding_models[DENSE_MODEL_NAME], SPARSE_MODEL_NAME: embedding_models[SPARSE_MODEL_NAME]}
    await recreate_collection(client, hybrid_embedding_models, HYBRID_COLLECTION_NAME)
    await index_docs_to_collection(client, hybrid_embedding_models, HYBRID_COLLECTION_NAME, batch_size=BATCH_SIZE, max_docs=MAX_DOCUMENTS)
    await finalize_indexing(client, HYBRID_COLLECTION_NAME)
    logger.info("Setup and indexing complete")


def _log_search_results(title: str, results: List[Dict[str, Any]], search_time: float):
    """Logs the title, time, and formatted results of a search."""
    logger.info(f"\n--- {title} (took {search_time:.3f}s) ---")
    if not results:
        logger.info("  No results found.")
        return
    for i, result in enumerate(results, 1):
        score = result.get("score", 0.0)
        text = result.get("payload", {}).get("text", "N/A")
        logger.info(f"  {i}. Score: {score:<6.4f} | Text: {text}")


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
        logger.info(f"        Native: (ID: {result['native_results']['id']}) {result['native_results']['payload']['text']}")
        logger.info(f"        Manual: (ID: {result['manual_results']['id']}) {result['manual_results']['payload']['text']}")


async def search_and_compare(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
    example_queries = ["What is the capital of France?", "Who wrote Romeo and Juliet?", "What is the largest planet in our solar system?"]

    non_matching_results = []
    total_native_search_time = 0
    total_manual_search_time = 0

    for query_text in example_queries:
        logger.info(f"\n{'='*25} 🔍 Query: {query_text} {'='*25}")

        start_time = time.time()
        native_results = await native_hybrid_search_with_reranking(
            query_text, embedding_models, client, RERANKING_COLLECTION_NAME, limit=5, prefetch_limit=PREFETCH_LIMIT
        )
        native_search_time = time.time() - start_time
        total_native_search_time += native_search_time
        _log_search_results("Native Search Results", native_results, native_search_time)

        start_time = time.time()
        manual_results = await manual_hybrid_search_with_reranking(
            query_text, embedding_models, client, HYBRID_COLLECTION_NAME, limit=5, prefetch_limit=PREFETCH_LIMIT
        )
        manual_search_time = time.time() - start_time
        total_manual_search_time += manual_search_time
        _log_search_results("Manual Search Results", manual_results, manual_search_time)

        for i, (res_nati, res_manu) in enumerate(zip(native_results, manual_results)):
            if res_nati["id"] != res_manu["id"]:
                non_matching_results.append({"query": query_text, "rank": i, "native_results": res_nati, "manual_results": res_manu})

    _log_summary(total_native_search_time, total_manual_search_time, non_matching_results, len(example_queries))


async def main():
    client = AsyncQdrantClient(url=f"http://{HOST}:{PORT}", timeout=600, prefer_grpc=True)
    embedding_models = initialize_embedding_models()

    await build_reranking_search_index(client, embedding_models)
    await build_hybrid_search_index(client, embedding_models)
    await search_and_compare(client, embedding_models)


if __name__ == "__main__":
    asyncio.run(main())
