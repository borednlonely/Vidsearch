# vidsearch 🔍

VidSearch is a local video analysis that uses a Vision Language Model to extract structured data from videos, store it in a vector database and lets you ask questions about the video content, without re-running it every time.

## How it works

1. **Video Analysis** - Qwen2.5VL 3B - processes your video and outputs timestamped JSON segments describing actions, body language, emotions, visuals and more
2. **Vector Storage** ChromaDB - Each segment gets flattened into text and embedded into ChromaDB for semantic search
3. **QnA** (via Ollama) - Ask questions in plain english. ChromaDB finds the most relevant segments using cosine similarity, then passes them as context to a second LLM which answers based on the stored data

The vision model runs only once per video. After that, you can ask as many questions as you want against the stored data. This also helps reduce hallucinations since the Q&A model only has access to what was actually extracted from the video.

## Features

1. Frame by frame video analysis with configurable FPS
2. Structured extraction: descriptions, actions, body language, emotions, visuals, timestamps
3. Semantic search over video segments via ChromaD
4. RAG based Q&A using a separate local LLM through Ollama
5. Tunable resolution and FPS settings for VRAM-constrained hardware

## How I built it

This was built and tested locally on a single RTX 3070 (8GB VRAM). The full 7B parameter Qwen model wouldn't fit, so I went with the 3B version and spent time reading through Qwen's documentation to understand how it tokenizes video frames. Qwen2.5VL splits each frame into 28x28 pixel patches, where each patch becomes one visual token. So the formula for controlling VRAM usage is: `(width × height) / 784 = tokens per frame`. By tuning the min/max pixel settings and capping FPS, I was able to get it running on 8GB without sacrificing too much accuracy. ChromaDB was chosen for its simplicity, it handles embedding and similarity search out of the box. The Q&A model runs through Ollama locally, keeping the entire pipeline offline and free.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 3070 8GB)
- Ollama

## Usage

1. Update the video path and title in app.py and vidsearch.py
```python
result = analyze_vid(r"path/to/your/video.mp4")
store_data("your_video.mp4", result)
```
2. Run the script "app.py"

## Improvements 
1. **Temporal Search** - Currently only semeantic search is implemented. Timestamps are stored in metadata but not yet used for time-based queries.
2. **Persistent storage** - ChromaDB runs in memory right now, so data is lost when the script exits. Swapping to `PersistentClient` would fix this
3. **Longer Videos & better Hardware** - right now the hardware constraints limit reliable analysis for videos longer than 15-20 seconds
