"""Web crawler module for RAG service"""
import time
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin
from dataclasses import dataclass
from collections import deque
import requests
from bs4 import BeautifulSoup
from loguru import logger

from config import settings
from utils import DomainValidator, RobotsChecker, TextCleaner, RateLimiter, PerformanceTimer

@dataclass
class CrawledPage:
    url: str
    title: str
    content: str
    links: List[str]
    crawl_time: float
    status_code: int
    content_type: str

class WebCrawler:
    def __init__(self, max_pages: int = None, max_depth: int = None, crawl_delay_ms: int = None, user_agent: str = None):
        self.max_pages = max_pages or settings.max_pages
        self.max_depth = max_depth or settings.max_depth
        self.crawl_delay_ms = crawl_delay_ms or settings.crawl_delay_ms
        # Set a realistic browser user-agent to avoid blocking
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/116.0.0.0 Safari/537.36"
        )

        self.robots_checker = RobotsChecker(self.user_agent)
        self.rate_limiter = RateLimiter(self.crawl_delay_ms)
        self.text_cleaner = TextCleaner()

        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.crawled_pages: List[CrawledPage] = []
        self.url_queue = deque()

        self.stats = {
            'pages_crawled': 0,
            'pages_skipped': 0,
            'pages_failed': 0,
            'total_time': 0,
            'avg_page_time': 0
        }

    def crawl(self, start_url: str) -> Dict:
        logger.info(f"Starting crawl from: {start_url}")
        start_time = time.time()

        try:
            normalized_start_url = DomainValidator.normalize_url(start_url)
            self.url_queue.append((normalized_start_url, 0))
            start_domain = DomainValidator.get_domain(start_url)

            while self.url_queue and len(self.crawled_pages) < self.max_pages:
                current_url, depth = self.url_queue.popleft()

                if depth > self.max_depth:
                    logger.debug(f"Skipping {current_url} - depth limit exceeded")
                    self.stats['pages_skipped'] += 1
                    continue

                if current_url in self.visited_urls:
                    continue

                if not DomainValidator.is_same_domain(current_url, start_url):
                    logger.debug(f"Skipping {current_url} - different domain")
                    self.stats['pages_skipped'] += 1
                    continue

                if not self.robots_checker.can_fetch(current_url):
                    logger.debug(f"Skipping {current_url} - robots.txt disallows")
                    self.stats['pages_skipped'] += 1
                    self.visited_urls.add(current_url)
                    continue

                page = self._crawl_page(current_url)
                if page:
                    self.crawled_pages.append(page)
                    self.stats['pages_crawled'] += 1

                    if depth < self.max_depth:
                        for link in page.links:
                            normalized_link = DomainValidator.normalize_url(link)
                            if normalized_link not in self.visited_urls and DomainValidator.is_same_domain(normalized_link, start_url):
                                self.url_queue.append((normalized_link, depth + 1))

                self.visited_urls.add(current_url)
                self.rate_limiter.wait_if_needed()

        except Exception as e:
            logger.error(f"Crawl failed: {e}")
            raise

        finally:
            total_time = time.time() - start_time
            self.stats['total_time'] = total_time
            if self.stats['pages_crawled']:
                self.stats['avg_page_time'] = total_time / self.stats['pages_crawled']

        logger.info(f"Crawl completed. Pages: {len(self.crawled_pages)}, Time: {total_time:.2f}s")
        return {
            'pages': self.crawled_pages,
            'stats': self.stats,
            'start_url': start_url,
            'domain': start_domain
        }

    def _crawl_page(self, url: str) -> Optional[CrawledPage]:
        with PerformanceTimer(f"Crawling {url}") as timer:
            try:
                logger.debug(f"Crawling page: {url}")

                headers = {
                    'User-Agent': self.user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive'
                }

                response = requests.get(url, headers=headers, timeout=settings.request_timeout, allow_redirects=True)

                if response.status_code != 200:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                    self.stats['pages_failed'] += 1
                    self.failed_urls.add(url)
                    return None

                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    logger.debug(f"Skipping {url} - not HTML content")
                    self.stats['pages_skipped'] += 1
                    return None

                soup = BeautifulSoup(response.content, 'html.parser')

                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else ''

                clean_text = self.text_cleaner.clean_html_text(response.text)
                links = self._extract_links(soup, url)

                return CrawledPage(
                    url=url,
                    title=title,
                    content=clean_text,
                    links=links,
                    crawl_time=timer.elapsed_ms,
                    status_code=response.status_code,
                    content_type=content_type
                )

            except requests.RequestException as e:
                logger.error(f"Request failed for {url}: {e}")
                self.stats['pages_failed'] += 1
                self.failed_urls.add(url)
                return None
            except Exception as e:
                logger.error(f"Unexpected error crawling {url}: {e}")
                self.stats['pages_failed'] += 1
                self.failed_urls.add(url)
                return None

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        links = []
        for link_tag in soup.find_all('a', href=True):
            href = link_tag.get('href')
            if not href:
                continue
            if href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                continue
            absolute_url = urljoin(base_url, href)
            if absolute_url.startswith(('http://', 'https://')):
                links.append(absolute_url)
        return list(set(links))

    def get_url_to_document_mapping(self) -> Dict[str, str]:
        return {page.url: page.content for page in self.crawled_pages}

    def save_crawl_results(self, filepath: str):
        import json
        data = {
            'pages': [
                {
                    'url': page.url,
                    'title': page.title,
                    'content': page.content,
                    'links': page.links,
                    'crawl_time': page.crawl_time,
                    'status_code': page.status_code,
                    'content_type': page.content_type
                }
                for page in self.crawled_pages
            ],
            'stats': self.stats,
            'visited_urls': list(self.visited_urls),
            'failed_urls': list(self.failed_urls)
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Crawl results saved to {filepath}")
