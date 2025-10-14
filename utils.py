"""Utility functions for RAG service"""
import re
import time
import hashlib
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
from loguru import logger

class DomainValidator:
    """Validate URLs are within the same domain"""

    @staticmethod
    def get_domain(url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.netloc.lower()}"

    @staticmethod
    def is_same_domain(url1: str, url2: str) -> bool:
        return DomainValidator.get_domain(url1) == DomainValidator.get_domain(url2)

    @staticmethod
    def normalize_url(url: str) -> str:
        parsed = urlparse(url)
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip('/') or '/',
            parsed.params,
            parsed.query,
            ''  # remove fragment
        ))
        return normalized

class RobotsChecker:
    """Check robots.txt compliance"""

    def __init__(self, user_agent: str = "*"):
        self.user_agent = user_agent
        self._robots_cache = {}

    def can_fetch(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            robots_url = urljoin(base_url, "/robots.txt")
            if robots_url not in self._robots_cache:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                    self._robots_cache[robots_url] = rp
                except Exception as e:
                    logger.warning(f"Could not read robots.txt for {base_url}: {e}")
                    return True  # allow if cannot fetch robots.txt
            rp = self._robots_cache[robots_url]
            return rp.can_fetch(self.user_agent, url)
        except Exception as e:
            logger.error(f"Error checking robots.txt for {url}: {e}")
            return True

class TextCleaner:
    """Clean and process text content"""

    @staticmethod
    def clean_html_text(html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button", "input"]):
            element.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        text = re.sub(r'[.,!?;:]{3,}', '...', text)
        return text.strip()

class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

    @property
    def elapsed_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

class RateLimiter:
    """Simple rate limiter"""

    def __init__(self, delay_ms: int = 1000):
        self.delay_seconds = delay_ms / 1000.0
        self.last_request_time = 0

    def wait_if_needed(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay_seconds:
            time.sleep(self.delay_seconds - time_since_last)
        self.last_request_time = time.time()

# Add the setup_logging function here, as it was missing
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up structured logging with Loguru"""
    logger.remove()  # Remove default handlers
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>"
    )
    if log_file:
        logger.add(
            sink=log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB"
        )
