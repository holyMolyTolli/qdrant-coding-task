#!/usr/bin/env python3

import asyncio
import logging
import time
from itertools import islice
from typing import Any, Dict, Generator, List

from datasets import load_dataset
from fastembed import (LateInteractionTextEmbedding, SparseTextEmbedding,
                       TextEmbedding)
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (BinaryQuantization, BinaryQuantizationConfig,
                                  Distance, HnswConfigDiff, Modifier,
                                  MultiVectorComparator, MultiVectorConfig,
                                  PointStruct, Prefetch, SparseIndexParams,
                                  SparseVector, SparseVectorParams,
                                  VectorParams)
from tqdm.asyncio import tqdm

from app.config import (BATCH_SIZE, COLLECTION_NAME, DATASET_NAME,
                        DENSE_MODEL_NAME, HNSW_M, HOST, MAX_DOCUMENTS, PORT,
                        REPLICATION_FACTOR, RERANKING_MODEL_NAME, SHARD_NUMBER,
                        SPARSE_MODEL_NAME)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def initialize_embedding_models() -> Dict[str, Any]:
    logging.info("Initializing embedding models...")
    return {
        DENSE_MODEL_NAME: TextEmbedding(DENSE_MODEL_NAME),
        SPARSE_MODEL_NAME: SparseTextEmbedding(SPARSE_MODEL_NAME),
        RERANKING_MODEL_NAME: LateInteractionTextEmbedding(RERANKING_MODEL_NAME),
    }


async def create_or_recreate_collection(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
    logging.info(f"Checking for existing Qdrant collection '{COLLECTION_NAME}'...")

    if await client.collection_exists(collection_name=COLLECTION_NAME):
        logging.warning(f"Collection '{COLLECTION_NAME}' already exists. Deleting it.")
        await client.delete_collection(collection_name=COLLECTION_NAME)

    logging.info(f"Creating new Qdrant collection '{COLLECTION_NAME}'...")
    dense_model = embedding_models[DENSE_MODEL_NAME]
    reranking_model = embedding_models[RERANKING_MODEL_NAME]

    await client.create_collection(
        collection_name=COLLECTION_NAME,
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


async def index_documents_to_qdrant(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
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
        client.upload_points(collection_name=COLLECTION_NAME, points=points_to_upload, wait=False, parallel=8, batch_size=BATCH_SIZE)
        logging.info(f"Upload time: {time.time() - start_time:.2f} seconds")

    logging.info(f"✅ Successfully uploaded {MAX_DOCUMENTS} documents.")


async def finalize_indexing(client: AsyncQdrantClient):
    logging.info("Resuming indexing for the collection...")
    await client.update_collection(collection_name=COLLECTION_NAME, hnsw_config=HnswConfigDiff(m=HNSW_M))
    logging.info("Indexing is now active and will build in the background.")


async def hybrid_search_with_reranking(query: str, embedding_models: Dict[str, Any], client: AsyncQdrantClient, limit: int = 10):
    dense_query = next(embedding_models[DENSE_MODEL_NAME].query_embed(query))
    sparse_query = next(embedding_models[SPARSE_MODEL_NAME].query_embed(query))
    late_query = next(embedding_models[RERANKING_MODEL_NAME].query_embed(query))

    prefetch = [
        Prefetch(query=dense_query, using=DENSE_MODEL_NAME, limit=20),
        Prefetch(query=SparseVector(**sparse_query.as_object()), using=SPARSE_MODEL_NAME, limit=20),
    ]

    results = await client.query_points(
        collection_name=COLLECTION_NAME, prefetch=prefetch, query=late_query, using=RERANKING_MODEL_NAME, with_payload=True, limit=limit
    )

    return results


async def search_examples(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
    example_queries = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
        "When was the Declaration of Independence signed?",
        "What is the chemical symbol for gold?",
    ]

    for query in example_queries:
        logging.info(f"\n{'='*60}")
        logging.info(f"🔍 Query: {query}")
        logging.info(f"{'='*60}")

        start_time = time.time()
        results = await hybrid_search_with_reranking(query, embedding_models, client, limit=5)
        search_time = time.time() - start_time

        logging.info(f"⏱️  Search completed in {search_time:.3f} seconds")
        logging.info(f"📄 Found {len(results.points)} results:")

        for i, result in enumerate(results.points, 1):
            logging.info(f"\n{i}. Score: {result.score:.4f}")
            logging.info(f"   Question: {result.payload.get('question', 'N/A')}")
            logging.info(f"   Answer: {result.payload.get('answer', 'N/A')}")


async def setup_and_index(client: AsyncQdrantClient, embedding_models: Dict[str, Any]):
    await create_or_recreate_collection(client, embedding_models)
    await index_documents_to_qdrant(client, embedding_models)
    await finalize_indexing(client)
    logging.info("Setup and indexing complete")


async def main():
    client = AsyncQdrantClient(url=f"http://{HOST}:{PORT}", timeout=600, prefer_grpc=True)
    embedding_models = initialize_embedding_models()

    await setup_and_index(client, embedding_models)
    await search_examples(client, embedding_models)


if __name__ == "__main__":
    asyncio.run(main())
