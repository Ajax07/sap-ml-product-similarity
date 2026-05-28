# SAP Product Similarity Search Engine

A scalable product similarity search engine for fashion products built as part of the SAP Senior Machine Learning Engineer technical assessment.

The system retrieves semantically relevant products for a given product ID using a hybrid retrieval pipeline combining:

* Transformer-based text embeddings
* Structured product metadata
* Similarity ranking
* Optional multimodal image similarity prototype

The solution is exposed as a production-style FastAPI microservice and containerized using Docker for reproducible deployment.

---

# Problem Statement

Given a fashion product catalog, build a similarity search system that returns semantically relevant products for a given product ID.

The system should account for:

* Product title and textual metadata
* Brand
* Color
* Price
* Weight
* Ratings

Additionally, the system should be scalable, containerized, and extensible toward multimodal retrieval.

---

# Dataset

The provided dataset contains approximately **30,000 fashion products** in LDJSON format.

### Key attributes used

#### Text Features

* `product_name`
* `meta_keywords`
* `parent___child_category__all`

#### Structured Features

* `sales_price`
* `rating`
* `weight`
* `brand`
* `colour`
* `amazon_prime__y_or_n`
* `best_seller_tag__y_or_n`

#### Optional Image Features

* `image_urls__small`

---

# Solution Overview

The retrieval system follows a hybrid similarity approach.

The pipeline consists of:

1. Data Loading
2. Data Preprocessing
3. Text Embedding Generation
4. Structured Feature Engineering
5. Hybrid Feature Fusion
6. Similarity Search
7. API Serving
8. Dockerized Deployment

---

# System Architecture

```text
                    ┌──────────────────┐
                    │ Fashion Dataset  │
                    └─────────┬────────┘
                              │
                       Data Loading
                              │
                    ┌─────────▼─────────┐
                    │ Preprocessing     │
                    │ - missing values  │
                    │ - weight cleaning │
                    │ - metadata clean  │
                    └─────────┬─────────┘
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
┌─────────▼──────────┐                 ┌──────────▼──────────┐
│ Text Embeddings    │                 │ Structured Features │
│ SentenceTransformer│                 │ price/rating/brand │
└─────────┬──────────┘                 └──────────┬──────────┘
          └───────────────────┬───────────────────┘
                              │
                     Feature Fusion
                              │
                    ┌─────────▼────────┐
                    │ Similarity Engine│
                    │ Cosine Similarity│
                    └─────────┬────────┘
                              │
                    ┌─────────▼────────┐
                    │ FastAPI Service  │
                    └─────────┬────────┘
                              │
                    ┌─────────▼────────┐
                    │ Docker Container │
                    └──────────────────┘
```

---

# Preprocessing Pipeline

Several preprocessing steps were applied to improve data quality and retrieval performance.

### Missing Value Handling

Missing categorical values were standardized using fallback values such as:

```text
unknown
```

### Weight Cleaning

The raw `weight` field contained noisy values such as:

```text
999999999
```

Weight values were cleaned and normalized into numeric format.

### Text Construction

A combined text field was constructed using:

* product name
* metadata keywords
* category hierarchy
* brand

This improved semantic retrieval quality.

---

# Modeling Decisions

## 1. Text Similarity

Text embeddings were generated using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model was selected because it offers:

* strong semantic retrieval quality
* lightweight inference
* fast embedding generation
* production suitability

The model captures semantic similarity between fashion products beyond exact keyword matching.

---

## 2. Structured Features

The following structured signals were incorporated:

* sales price
* rating
* weight
* brand
* colour
* Prime availability
* bestseller tag

### Representation Strategy

Initially, high-cardinality one-hot encoding was explored for categorical variables such as brand and colour.

However, the dataset contains:

* ~6,400 brands
* ~4,700 color combinations

This significantly increased memory usage during containerized deployment.

To improve deployment stability and reduce memory footprint, lightweight categorical encodings were adopted.

Since semantic transformer embeddings already capture substantial product context, structured features primarily act as ranking refinements rather than primary retrieval drivers.

---

## 3. Similarity Retrieval

### Initial Approach

Approximate nearest neighbor retrieval using FAISS was initially implemented.

### Final Deployment Decision

During deployment, runtime instability was observed on:

```text
macOS + Python 3.13
```

particularly during containerized execution.

To prioritize:

* reproducibility
* deployment stability
* reliability

the submitted solution uses:

```text
Cosine Similarity
```

for retrieval.

This decision was made intentionally to ensure robust production behavior.

---

# Feature Fusion Strategy

Final retrieval embeddings are constructed using weighted feature fusion.

### Weights

```text
Text Embeddings      → 55%
Structured Features  → 45%
```

Text similarity acts as the dominant retrieval signal, while metadata improves ranking precision.

---

# Caching Strategy

To reduce cold-start latency, transformer embeddings are cached locally.

Cached artifact:

```text
cache/text_embeddings.npy
```

Benefits:

* avoids repeated transformer inference
* significantly improves startup performance
* reduces API initialization time

This optimization makes the system more suitable for deployment scenarios.

---

# API Service

The similarity engine is exposed through a FastAPI microservice.

## Start API locally

```bash
uvicorn app.main:app
```

API documentation:

```text
http://localhost:8000/docs
```

---

## Endpoint

### Find Similar Products

```http
GET /find_similar_products
```

### Query Parameters

| Parameter   | Type    | Description                |
| ----------- | ------- | -------------------------- |
| product_id  | string  | Unique product ID          |
| num_similar | integer | Number of similar products |

### Example Request

```http
/find_similar_products?product_id=26d41bdc1495de290bc8e6062d927729&num_similar=5
```

### Example Response

```json
{
  "product_id": "26d41bdc1495de290bc8e6062d927729",
  "similar_products": [
    "87cfe01c112f32fb0cb7e2b1354ce8a5",
    "8c1c7cfc44b7a5cbd38d47b506a58518",
    "1c93cbcd527aad73ae347dd6ac08d93c",
    "630cb8d7f5f66fc11dbd12631c269d36",
    "c8666212f128479ac11fa136778545a0"
  ]
}
```

---

# Docker Deployment

## Build Docker Image

```bash
docker build -t sap-product-similarity .
```

## Run Container

```bash
docker run -p 8000:8000 sap-product-similarity
```

Open:

```text
http://localhost:8000/docs
```

---

# Optional Multimodal Similarity Extension

As an optional enhancement, a multimodal similarity prototype was implemented.

## Text Similarity

Transformer embeddings generated using:

```text
SentenceTransformer
```

## Image Similarity

Visual embeddings extracted using:

```text
ResNet18
```

pretrained CNN features.

## Fusion Strategy

A weighted multimodal fusion strategy was explored:

```text
Text Similarity        → 70%
Structured Metadata   → 15%
Image Similarity      → 15%
```

### Fallback Mechanism

When image features are unavailable:

```text
Text + Metadata Retrieval
```

is used as fallback.

This ensures graceful degradation and stable retrieval quality.

---

# Assumptions and Limitations

* Product descriptions were sparse; product names and metadata were prioritized.
* High-cardinality categorical variables required lightweight encoding for deployment stability.
* Approximate nearest neighbor search was explored but replaced for deployment reliability.
* Multimodal retrieval was implemented as a prototype for demonstration purposes.
* Larger-scale production systems would benefit from dedicated vector indexes.

---

# Future Improvements

Potential future enhancements include:

* FAISS / HNSW approximate nearest neighbor retrieval
* Vector database integration
* Redis-based caching
* PCA-based dimensionality reduction
* Full multimodal production pipeline
* Online feedback ranking
* Real-time recommendation optimization

---

# Tech Stack

* Python
* FastAPI
* Sentence Transformers
* PyTorch
* Scikit-learn
* Docker
* Pandas
* NumPy

---

# Repository Structure

```text
.
├── app
│   ├── api
│   ├── core
│   ├── schemas
│   └── services
├── cache
├── data
├── notebooks
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Author

Ajay Singh

SAP Senior Machine Learning Engineer Assessment
