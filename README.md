# Hybrid Search with Qdrant

A complete hybrid search implementation using Qdrant's Query API with sparse and dense vectors, binary quantization, and ColBERT reranking.

## Overview

This project implements a hybrid search system that combines:
- **Sparse vectors** (BM25) for keyword matching
- **Dense vectors** (sentence-transformers) for semantic similarity
- **Late interaction reranking** (ColBERT) for improved results
- **Binary quantization** for efficient storage

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Qdrant Node 1 │    │   Qdrant Node 2 │
│   Port: 6333    │◄──►│   Port: 6337    │
│   Shards: 1-2   │    │   Shards: 2-3   │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┘
                    │
         ┌─────────────────┐
         │  Python Client  │
         │  Hybrid Search  │
         │  + ColBERT      │
         └─────────────────┘
```

## Prerequisites

- Python 3.8+
- Docker & Docker Compose
- 2GB+ available space

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/holyMolyTolli/qdrant-coding-task.git
cd qdrant-coding-task

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Qdrant Cluster

```bash
docker-compose up -d
```

Wait for cluster to be ready at http://localhost:6333

### 3. Run Application

```bash
python -m app.hybrid_search_app
```

## Configuration

### Cluster Setup
- **Nodes**: 2 (peer-to-peer communication)
- **Shards**: 3 (distributed across nodes)
- **Replication**: 2x (high availability)
- **Quantization**: Binary (dense vectors)

### Models
- **Dense**: `sentence-transformers/all-MiniLM-L6-v2` (384d)
- **Sparse**: `Qdrant/bm25` (BM25 scoring)
- **Reranking**: `colbert-ir/colbertv2.0` (late interaction)

### Dataset
- **Source**: SQuAD (Stanford Question Answering Dataset)
- **Size**: 100K+ documents (configurable)