import argparse
import sys
import json
import time
from crawler import WebCrawler
from indexer import DocumentIndexer
from qa_system import QuestionAnsweringSystem

def crawl_site(url, max_pages, max_depth, crawl_delay_ms):
    print(f"Starting crawl on {url} (max pages={max_pages}, max depth={max_depth})")
    crawler = WebCrawler(max_pages=max_pages, max_depth=max_depth, crawl_delay_ms=crawl_delay_ms)
    result = crawler.crawl(url)
    print(f"Crawled {result['stats']['pages_crawled']} pages, skipped {result['stats']['pages_skipped']}, failed {result['stats']['pages_failed']}")
    return crawler

def index_content(crawler, chunk_size, chunk_overlap, embedding_model):
    print("Starting indexing...")
    indexer = DocumentIndexer(chunk_size=chunk_size, chunk_overlap=chunk_overlap, embedding_model=embedding_model)
    # Correctly extract (url, content) pairs for indexing
    url_to_content = [(page.url, page.content) for page in crawler.crawled_pages]
    index_result = indexer.index_documents(url_to_content)
    print(f"Indexed {index_result['total_chunks']} chunks from {index_result['total_documents']} documents in {index_result['total_time_ms']:.2f} ms")
    return indexer

def interactive_qa(qa_system):
    print("\nEntering interactive Q&A session. Type 'exit' to quit.")
    while True:
        question = input("\nEnter your question: ").strip()
        if question.lower() in ['exit', 'quit']:
            print("Exiting Q&A session.")
            break
        result = qa_system.answer_question(question)
        print("\nAnswer:")
        print(result.answer)
        print("\nSources:")
        for src in result.sources:
            print(f" - {src['url']}")
        print(f"\nRetrieval time: {result.retrieval_time_ms:.2f} ms, Generation time: {result.generation_time_ms:.2f} ms, Total time: {result.total_time_ms:.2f} ms")

def main():
    parser = argparse.ArgumentParser(description="RAG CLI for crawling, indexing, and Q&A")
    
    parser.add_argument('url', help='Starting website URL to crawl')
    parser.add_argument('--max-pages', type=int, default=50, help='Maximum number of pages to crawl (default 50)')
    parser.add_argument('--max-depth', type=int, default=3, help='Maximum crawl depth (default 3)')
    parser.add_argument('--crawl-delay', type=int, default=1000, help='Delay between requests in ms (default 1000)')
    parser.add_argument('--chunk-size', type=int, default=800, help='Text chunk size for indexing (default 800 chars)')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap size (default 100 chars)')
    parser.add_argument('--embedding-model', type=str, default='all-MiniLM-L6-v2', help='Embedding model name (default all-MiniLM-L6-v2)')
    
    args = parser.parse_args()

    try:
        crawler = crawl_site(args.url, args.max_pages, args.max_depth, args.crawl_delay)
        indexer = index_content(crawler, args.chunk_size, args.chunk_overlap, args.embedding_model)
        qa_system = QuestionAnsweringSystem(indexer)

        interactive_qa(qa_system)

    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
