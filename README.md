---
title: LEANN RAG • Qwen3 0.6B
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: 97% smaller vector search running on CPU
---

# 🧠 LEANN RAG: The "Impossible" Vector Search

This Space demonstrates **LEANN** (Lightweight Embedding & Neural Network), a revolutionary vector search engine that runs highly efficient RAG (Retrieval Augmented Generation) on consumer hardware.

Unlike traditional vector databases that store heavy embedding vectors for every document (bloating storage), LEANN stores **only the graph structure** and re-computes embeddings on-the-fly during search. This reduces index size by **97%**, allowing us to run a powerful semantic search engine entirely on the **Hugging Face Free Tier (2 vCPU)**.

---

## 🏗️ Architecture: How it Works

This application is a complete RAG pipeline optimized for low-resource environments.

```mermaid
graph LR
    A[User PDF/Text] -->|Chunking| B(Text Chunks)
    B -->|LEANN Indexer| C{HNSW Graph Construction}
    C -->|Compress| D[Sparse Graph Index]
    
    D -->|User Query| E[Graph Traversal]
    E -->|On-Demand Compute| F[Embedding Re-calculation]
    F -->|Top-K Results| G[Qwen3-0.6B LLM]
    G -->|Response| H[User Chat]
    
    style D fill:#f96,stroke:#333,stroke-width:2px
    style F fill:#f9f,stroke:#333,stroke-width:2px
