import unittest
from vectorizators import TFIDFVectorizator

class MyTestCase(unittest.TestCase):
	def test_something(self):
		tfidf = TFIDFVectorizator()
		print(tfidf.vectorize("C++"))
		print(tfidf.vectorize("puzzle"))
		print(tfidf.vectorize("love"))
		print(tfidf.vectorize("do"))
		print(tfidf.vectorize("hello"))
		print(tfidf.vectorize("help"))

if __name__ == '__main__':
	unittest.main()
