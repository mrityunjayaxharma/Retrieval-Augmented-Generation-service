Retrieval-Augmented Generation (RAG) Service
Overview
This project implements a small RAG service that, given any starting website URL, crawls the site (limited to the same domain), indexes the crawled content, and serves answers to questions grounded strictly in the indexed pages. Answers include explicit citations of source page URLs and are said to be refused with relevant snippets when insufficient supporting information is found.

The system covers core practical skills: polite web crawling, high-quality content ingestion, semantic retrieval, grounded prompt answer generation, and evaluation.

Objective
Crawl in-domain pages up to a configurable limit (default 30 pages), respecting robots.txt and domain scope.

Extract clean, boilerplate-reduced HTML text, chunk content with well-justified size and minimal overlap.

Compute semantic embeddings using open source models and build a FAISS vector index for similarity search.

Provide an API / CLI question answering interface that retrieves top-k chunks, generates answers strictly from retrieved context, and returns answers with source URLs and snippet highlights.

Return explicit refusal when an answer is not supported by retrieved content.

Log detailed timings for retrieval, generation, and end-to-end latency alongside retrieval metadata.

Architecture
Crawler
Uses a breadth-first crawl strategy restricted to the same registrable domain.

Respects crawl delay and robots.txt rules to avoid server overload.

Extracts main textual content and filters scripts, styles, and navigation components.

Normalizes text encoding and whitespace for cleaning.

Indexer
Splits documents into overlapping chunks of ~500-1000 characters based on sentence boundaries using NLTK.

Uses the SentenceTransformers all-MiniLM-L6-v2 embedding model (384-dimensional).

Embeddings are normalized and indexed using FAISS for efficient approximate nearest neighbor search.

Configurable chunk size and overlap documented with rationale in code comments.

QA System
Retrieves top-k relevant chunks based on vector similarity.

Aggregates candidate sentences with keyword overlap with the question.

Generates natural answers by concatenating top relevant sentences with source citations.

Implements strict refusal rules using similarity score thresholds to avoid hallucinations.

Logs confidence, retrieval, generation, and total latency metrics.

Safety and Guardrails
Enforces domain boundary by restricting crawls and retrieval to initial domain.

Refuses queries with no indexed content support, returning refusal reasons and closest snippets.

Prompt hardening includes ignoring any instructions embedded in crawled pages.

Installation
bash
pip install -r requirements.txt
python -m nltk.downloader punkt
Usage
Crawl and Index:
bash
python cli.py https://docs.python.org/3/ --max-pages 30 --max-depth 2
Collects and indexes content from the entire docs.python.org domain.

Interactive Q&A:
bash
python qa_cli.py
Enter questions interactively. Example queries:

What is Python?

How do you define a function in Python?

What is the weather today? (Expected refusal)

Evaluation
Answers are grounded strictly in retrieved chunks and include URL citations.

Refusals are explicit with closest available snippets.

Retrieval recall and top-k tuning are documented with reproduction steps in tests.

Errors and latencies are thoroughly logged and metrics available for analysis.

Tradeoffs and Limitations
Simple answer generation based on sentence extraction; no LLM fine-tuning due to scope/time limits.

Crawling excludes multimedia and heavy dynamic content.

Potential misses if useful info is outside collected pages or insufficiently chunked.

System tested primarily on static informative sites like docs.python.org.

Tooling and Prompts
Embedding model: SentenceTransformers all-MiniLM-L6-v2

Vector database: FAISS (Cosine similarity via inner product on normalized vectors)

Sentence tokenizer: NLTK punkt

QA generation: Keyword-based sentence extraction from retrieved chunks

Submission Contents
crawler.py: polite, domain-bound web crawler extracting clean text.

indexer.py: chunking, embedding, and FAISS indexing of crawled pages.

qa_system.py: retrieval, answer generation, refusal logic, confidence scoring.

cli.py: CLI tools for crawling, indexing, and querying.

Tests covering crawl limits, indexing correctness, QA answer accuracy, and refusal handling.

This README.md describing design, usage, and evaluation steps.
