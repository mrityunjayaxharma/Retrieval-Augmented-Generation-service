"""Text indexing module for RAG service using FAISS and sentence transformers"""
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import faiss
from sentence_transformers import SentenceTransformer
import nltk
from loguru import logger

from config import settings
from utils import PerformanceTimer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

@dataclass
class TextChunk:
    text: str
    source_url: str
    chunk_id: str
    start_pos: int
    end_pos: int
    embedding: Optional[np.ndarray] = None

class TextChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk_text(self, text: str, source_url: str) -> List[TextChunk]:
        if not text.strip():
            return []

        sentences = nltk.tokenize.sent_tokenize(text)

        chunks = []
        current_chunk = ""
        current_pos = 0
        chunk_counter = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunk_id = f"{source_url}#{chunk_counter}"
                chunk = TextChunk(
                    text=current_chunk.strip(),
                    source_url=source_url,
                    chunk_id=chunk_id,
                    start_pos=current_pos - len(current_chunk),
                    end_pos=current_pos
                )
                chunks.append(chunk)

                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence

                chunk_counter += 1
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

            current_pos += len(sentence) + 1

        if current_chunk.strip():
            chunk_id = f"{source_url}#{chunk_counter}"
            chunk = TextChunk(
                text=current_chunk.strip(),
                source_url=source_url,
                chunk_id=chunk_id,
                start_pos=current_pos - len(current_chunk),
                end_pos=current_pos
            )
            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} chunks from {source_url}")
        return chunks

class VectorIndexer:
    def __init__(self, embedding_model: str = None, index_path: str = None):
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.index_path = index_path or settings.vector_db_path

        self.embedding_model = None
        self.faiss_index = None
        self.chunks: List[TextChunk] = []
        self.embedding_dim = None

        self._load_embedding_model()

    def _load_embedding_model(self):
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            test_embedding = self.embedding_model.encode(["test"])
            self.embedding_dim = test_embedding.shape[1]
            logger.info(f"Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def create_embeddings(self, chunks: List[TextChunk]) -> List[TextChunk]:
        if not chunks:
            return chunks

        logger.info(f"Creating embeddings for {len(chunks)} chunks")

        with PerformanceTimer("Embedding creation") as timer:
            texts = [chunk.text for chunk in chunks]

            batch_size = 32
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                all_embeddings.append(batch_embeddings)

            embeddings = np.vstack(all_embeddings)

        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]

        logger.info(f"Created embeddings in {timer.elapsed_ms:.2f}ms")
        return chunks

    def build_index(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        if not chunks:
            raise ValueError("No chunks provided for indexing")

        logger.info(f"Building FAISS index for {len(chunks)} chunks")

        with PerformanceTimer("Index building") as timer:
            chunks_with_embeddings = [c for c in chunks if c.embedding is not None]
            if len(chunks_with_embeddings) < len(chunks):
                logger.info("Creating missing embeddings...")
                chunks = self.create_embeddings(chunks)

            embeddings = np.array([chunk.embedding for chunk in chunks])
            faiss.normalize_L2(embeddings)

            if settings.index_type == "IndexFlatL2":
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            elif settings.index_type == "IndexFlatIP":
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

            self.faiss_index.add(embeddings)
            self.chunks = chunks

        index_info = {
            'total_chunks': len(chunks),
            'embedding_dim': self.embedding_dim,
            'index_type': type(self.faiss_index).__name__,
            'build_time_ms': timer.elapsed_ms
        }

        logger.info(f"Index built successfully in {timer.elapsed_ms:.2f}ms")
        return index_info

    def search(self, query: str, top_k: int = None) -> List[Tuple[TextChunk, float]]:
        if not self.faiss_index or not self.chunks:
            raise ValueError("Index not built. Call build_index first.")

        top_k = top_k or settings.top_k

        logger.debug(f"Searching for: '{query[:50]}...' (top_k={top_k})")

        with PerformanceTimer("Vector search") as timer:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)

            scores, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                chunk = self.chunks[idx]
                similarity = 1.0 / (1.0 + score) if settings.index_type == "IndexFlatL2" else float(score)
                results.append((chunk, similarity))

        logger.debug(f"Search completed in {timer.elapsed_ms:.2f}ms, found {len(results)} results")
        return results

    def save_index(self, filepath: str = None):
        filepath = filepath or self.index_path
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        faiss_file = f"{filepath}.faiss"
        faiss.write_index(self.faiss_index, faiss_file)

        metadata_file = f"{filepath}.pkl"
        metadata = {
            'chunks': self.chunks,
            'embedding_model_name': self.embedding_model_name,
            'embedding_dim': self.embedding_dim,
            'index_type': type(self.faiss_index).__name__
        }

        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Index saved to {filepath}")

    def load_index(self, filepath: str = None):
        filepath = filepath or self.index_path

        faiss_file = f"{filepath}.faiss"
        if not os.path.exists(faiss_file):
            raise FileNotFoundError(f"Index file not found: {faiss_file}")

        self.faiss_index = faiss.read_index(faiss_file)

        metadata_file = f"{filepath}.pkl"
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)

        self.chunks = metadata['chunks']
        self.embedding_dim = metadata['embedding_dim']

        if metadata['embedding_model_name'] != self.embedding_model_name:
            logger.warning(f"Model mismatch: saved={metadata['embedding_model_name']}, current={self.embedding_model_name}")

        logger.info(f"Index loaded from {filepath} ({len(self.chunks)} chunks)")


class DocumentIndexer:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, embedding_model: str = None):
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.vector_indexer = VectorIndexer(embedding_model)

    def index_documents(self, url_content_list: List[Tuple[str, str]]) -> Dict[str, Any]:
        logger.info(f"Indexing {len(url_content_list)} documents")

        start_time = PerformanceTimer("Total indexing")
        start_time.__enter__()

        try:
            all_chunks = []
            for url, content in url_content_list:
                if content.strip():
                    chunks = self.chunker.chunk_text(content, url)
                    all_chunks.extend(chunks)

            if not all_chunks:
                raise ValueError("No valid chunks created from documents")

            logger.info(f"Created {len(all_chunks)} chunks from {len(url_content_list)} documents")

            index_info = self.vector_indexer.build_index(all_chunks)
            self.vector_indexer.save_index()

            start_time.__exit__(None, None, None)

            return {
                'total_documents': len(url_content_list),
                'total_chunks': len(all_chunks),
                'index_info': index_info,
                'total_time_ms': start_time.elapsed_ms,
                'chunks_per_doc': len(all_chunks) / len(url_content_list) if url_content_list else 0
            }

        except Exception as e:
            start_time.__exit__(None, None, None)
            logger.error(f"Indexing failed: {e}")
            raise

    def search(self, query: str, top_k: int = None) -> List[Tuple[TextChunk, float]]:
        return self.vector_indexer.search(query, top_k)

    def load_existing_index(self):
        self.vector_indexer.load_index()
