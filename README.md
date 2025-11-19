---
title: Leann RAG Qwen3
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

# LEANN RAG Demo

This Space demonstrates [LEANN](https://github.com/yichuan-w/LEANN), a highly efficient vector search engine that runs on consumer hardware.

## Features
- **Vector Backend**: HNSW (via LEANN), running entirely on CPU.
- **LLM**: Qwen/Qwen3-0.6B (Free, Local) or OpenAI (Fast).
- **Privacy**: Documents are indexed locally within this Space's session.

## How to use
1. Upload PDF or Text files.
2. Click "Build Index".
3. Chat with your data!
