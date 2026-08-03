# Semantic Search & Clustering (From Scratch)

A local tool that embeds text using `all-MiniLM-L6-v2` (via Hugging Face's `sentence-transformers`), and lets you:

1. Search text by meaning, not keywords
2. Cluster text into topic groups
3. Visualize those clusters in 2D

Built without any cloud APIs — runs fully offline.

## How to run

```bash
pip install -r requirements.txt
python embedding.py
```

## How it works

### Embeddings

Embeddings convert text (or audio/images) into numerical vectors that a model can reason over mathematically. Every input — regardless of length — gets mapped to a fixed-size vector (384 numbers, in this project), so a single word and a full paragraph can be compared using the same math.

### Cosine similarity

Cosine similarity measures how similar two vectors are by comparing the angle between them, not their distance or magnitude. This matters because a vector's length can be inflated by factors like sentence length, without that saying anything about meaning. Cosine similarity ignores length and focuses purely on direction, which correlates with semantic similarity.

### Clustering (K-Means)

K-Means groups similar vectors into `n_clusters` groups by finding cluster centers and assigning each point to its nearest center. I set `n_clusters=4` because my dataset was deliberately built with 4 topics (AI, food, space, finance) — in a real, unlabeled dataset, this number usually isn't known in advance and needs to be estimated or tuned.

### Visualization (PCA)

The embeddings are 384-dimensional, far beyond what can be visually plotted. PCA reduces this to the 2 dimensions along which the data varies most, so similar sentences end up positioned near each other on a 2D plot — a lossy but useful approximation for visual inspection.

## Example output

```
Enter your prompt: Top AI companies
Top 3 results
OpneAI, Antropic and Google are in an AI race  ->  0.5138077
Artificial intelligence models use vector embeddings to understand semantic meaning.  ->  0.32042196
...
```

*(No exact word overlap between the query and the top match — confirms this is matching on meaning, not keywords.)*

![Cluster visualization](cluster_plot.png)

## What I'd improve

- Add a similarity threshold so the search reports "no strong match found" instead of always returning a top-3, even when nothing is actually relevant
- Expand the dataset — 16 sentences is enough to sanity-check the pipeline, but too small for clustering to be meaningful at scale
- Try a stronger embedding model (e.g. `all-mpnet-base-v2` or a BGE model) and compare results
