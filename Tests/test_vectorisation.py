import unittest
from vectorizators import TFIDFVectorizator

class MyTestCase(unittest.TestCase):
	def test_something(self):
		tfidf = TFIDFVectorizator()
		print(tfidf.vectorize("C++"))
		print(tfidf.vectorize("puzzle"))

if __name__ == '__main__':
	unittest.main()
