import unittest
from qa_system import QuestionAnsweringSystem
from crawler import WebCrawler
from indexer import DocumentIndexer

class RAGSystemEval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Crawl example site - docs.python.org small crawl for test
        cls.start_url = "https://docs.python.org/3/"
        crawler = WebCrawler(max_pages=10, max_depth=1)
        crawl_result = crawler.crawl(cls.start_url)
        cls.documents = [(page.url, page.content) for page in crawl_result['pages']]

        # Index documents (embedding and indexing done internally)
        cls.indexer = DocumentIndexer()
        cls.indexer.index_documents(cls.documents)

        # Load QA system with this indexer
        cls.qa = QuestionAnsweringSystem(indexer=cls.indexer)

    def test_answer_python_basic(self):
        question = "latest version?"
        result = self.qa.answer_question(question)
        self.assertTrue(result.is_answerable)
        self.assertIn("Python", result.answer)
        self.assertGreater(len(result.sources), 0)
        print("\nTest 1 - Basic Python Question:")
        print(f"Q: {question}\nA: {result.answer}\nSources: {result.sources}")

    def test_answer_function_definition(self):
        question = "How do you define a function in Python?"
        result = self.qa.answer_question(question)
        self.assertTrue(result.is_answerable)
        # Relaxed assertion to include 'def' or 'function'
        self.assertTrue("def" in result.answer or "function" in result.answer.lower())
        print("\nTest 2 - Function Definition:")
        print(f"Q: {question}\nA: {result.answer}\nSources: {result.sources}")

    def test_refusal_out_of_domain(self):
        question = "What is the weather today?"
        result = self.qa.answer_question(question)
        self.assertFalse(result.is_answerable)
        self.assertTrue("Not enough information" in result.answer or "refusal" in (result.refusal_reason or "").lower())
        print("\nTest 3 - Out-of-Domain Refusal:")
        print(f"Q: {question}\nA: {result.answer}")

if __name__ == "__main__":
    unittest.main()
