"""Basic tests for RAG service components"""
import pytest
from utils import DomainValidator
from crawler import WebCrawler

class TestDomainValidator:
    def test_get_domain(self):
        assert DomainValidator.get_domain("https://example.com/path") == "example.com"
        assert DomainValidator.get_domain("http://sub.example.com") == "sub.example.com"

    def test_same_domain_check(self):
        assert DomainValidator.is_same_domain(
            "https://example.com/page1",
            "https://example.com/page2"
        )
        assert not DomainValidator.is_same_domain(
            "https://example.com",
            "https://other.com"
        )

    def test_url_normalization(self):
        normalized = DomainValidator.normalize_url("https://Example.Com/Path/")
        assert normalized == "https://example.com/Path"

class TestWebCrawler:
    def test_initialization(self):
        crawler = WebCrawler(max_pages=5, max_depth=2, crawl_delay_ms=100)
        assert crawler.max_pages == 5
        assert crawler.max_depth == 2
        assert crawler.crawl_delay_ms == 100

if __name__ == "__main__":
    pytest.main([__file__])
