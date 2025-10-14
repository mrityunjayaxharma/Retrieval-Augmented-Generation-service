"""Question Answering system for RAG service"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

# Import NLTK sentence tokenizer
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

from config import settings
from indexer import TextChunk, DocumentIndexer
from utils import PerformanceTimer


@dataclass
class QAResult:
    question: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    is_answerable: bool
    refusal_reason: Optional[str] = None


class SimpleQAGenerator:
    def __init__(self, similarity_threshold: float = None):
        self.similarity_threshold = similarity_threshold or settings.similarity_threshold

    def generate_answer(self, question: str, context_chunks: List[Tuple[TextChunk, float]]) -> str:
        if not context_chunks:
            return "Not enough information available in the crawled content."

        relevant_chunks = [
            (chunk, score) for chunk, score in context_chunks
            if score >= self.similarity_threshold
        ]
        if not relevant_chunks:
            relevant_chunks = context_chunks[:3]

        answer_parts = []
        sources_used = []

        question_lower = question.lower()
        question_keywords = set(self._extract_keywords(question_lower))

        sentence_scores = []

        # Collect sentences with keyword overlap scores
        for chunk, score in relevant_chunks:
            sentences = sent_tokenize(chunk.text)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                overlap = len([kw for kw in question_keywords if kw in sentence_lower])
                if overlap > 0:
                    sentence_scores.append((overlap, sentence, chunk.source_url))

        # Sort sentences by keyword overlap descending
        sentence_scores.sort(key=lambda x: x[0], reverse=True)

        if not sentence_scores:
            # fallback to best chunk snippet
            best_chunk = relevant_chunks[0][0]
            snippet = best_chunk.text[:300]
            if len(best_chunk.text) > 300:
                snippet += "..."
            sources = "\n\nSources:\n- " + best_chunk.source_url
            return snippet + sources

        # Take top 5 scoring sentences
        top_sentences = sentence_scores[:5]
        answer = " ".join([s[1] for s in top_sentences])
        unique_sources = list(dict.fromkeys([s[2] for s in top_sentences]))
        answer += "\n\nSources:\n" + "\n".join(f"- {source}" for source in unique_sources)

        return answer

    def _extract_keywords(self, text: str) -> List[str]:
        stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were',
                      'do', 'does', 'did', 'will', 'would', 'could', 'should', 'the', 'a', 'an',
                      'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords


class QuestionAnsweringSystem:
    def __init__(self, indexer: DocumentIndexer = None):
        self.indexer = indexer or DocumentIndexer()
        self.qa_generator = SimpleQAGenerator()

        try:
            self.indexer.load_existing_index()
            logger.info("Loaded existing index")
        except (FileNotFoundError, Exception) as e:
            logger.info(f"No existing index found: {e}")

    def answer_question(self, question: str, top_k: int = None) -> QAResult:
        logger.info(f"Answering question: {question[:100]}...")

        total_timer = PerformanceTimer("Total QA")
        total_timer.__enter__()

        try:
            with PerformanceTimer("Retrieval") as retrieval_timer:
                retrieved_chunks = self.indexer.search(question, top_k)

            retrieval_time = retrieval_timer.elapsed_ms

            # Early refusal if lowest similarity threshold (adjust as needed)
            if retrieved_chunks and retrieved_chunks[0][1] < 0.4:
                answer = "Not enough information available in the crawled content."
                is_answerable = False
                refusal_reason = "Low similarity score"
                sources = []
                total_timer.__exit__(None, None, None)
                return QAResult(
                    question=question,
                    answer=answer,
                    confidence=0.0,
                    sources=sources,
                    retrieval_time_ms=retrieval_time,
                    generation_time_ms=0.0,
                    total_time_ms=total_timer.elapsed_ms,
                    is_answerable=is_answerable,
                    refusal_reason=refusal_reason,
                )

            with PerformanceTimer("Generation") as generation_timer:
                if not retrieved_chunks:
                    answer = "Not enough information available in the crawled content."
                    is_answerable = False
                    refusal_reason = "No relevant content found"
                    sources = []
                else:
                    answer = self.qa_generator.generate_answer(question, retrieved_chunks)
                    is_answerable = not answer.startswith("Not enough information")
                    refusal_reason = None if is_answerable else "Insufficient relevant content"
                    sources = [
                        {
                            "url": chunk.source_url,
                            "snippet": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                            "similarity_score": float(score),
                            "chunk_id": chunk.chunk_id,
                        }
                        for chunk, score in retrieved_chunks[:5]
                    ]

            generation_time = generation_timer.elapsed_ms
            total_timer.__exit__(None, None, None)
            total_time = total_timer.elapsed_ms

            confidence = self._calculate_confidence(retrieved_chunks, answer, is_answerable)

            return QAResult(
                question=question,
                answer=answer,
                confidence=confidence,
                sources=sources,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=total_time,
                is_answerable=is_answerable,
                refusal_reason=refusal_reason,
            )
        except Exception as e:
            total_timer.__exit__(None, None, None)
            logger.error(f"QA failed: {e}")
            return QAResult(
                question=question,
                answer=f"Error processing question: {str(e)}",
                confidence=0.0,
                sources=[],
                retrieval_time_ms=0.0,
                generation_time_ms=0.0,
                total_time_ms=total_timer.elapsed_ms,
                is_answerable=False,
                refusal_reason="Processing error",
            )

    def _calculate_confidence(self, chunks: List[Tuple[TextChunk, float]], answer: str, is_answerable: bool) -> float:
        if not is_answerable or not chunks:
            return 0.0
        avg_similarity = sum(score for _, score in chunks[:3]) / min(3, len(chunks))
        confidence = max(0.0, min(1.0, avg_similarity))
        if len(answer) > 100:
            confidence *= 1.1
        if len(answer) < 50:
            confidence *= 0.8
        if len(set(chunk.source_url for chunk, _ in chunks[:3])) > 1:
            confidence *= 1.05
        return min(1.0, confidence)

    def get_system_status(self) -> Dict[str, Any]:
        try:
            index_info = {
                "has_index": self.indexer.vector_indexer.faiss_index is not None,
                "total_chunks": len(self.indexer.vector_indexer.chunks)
                if self.indexer.vector_indexer.chunks
                else 0,
                "embedding_model": self.indexer.vector_indexer.embedding_model_name,
                "embedding_dim": self.indexer.vector_indexer.embedding_dim,
            }
        except Exception as e:
            index_info = {"error": str(e), "has_index": False}
        return {
            "status": "ready" if index_info.get("has_index") else "no_index",
            "index_info": index_info,
            "config": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k": settings.top_k,
                "similarity_threshold": settings.similarity_threshold,
            },
        }
