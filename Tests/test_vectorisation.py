import unittest
from vectorizators import TFIDFVectorizator

class MyTestCase(unittest.TestCase):
	def test_something(self):
		tfidf = TFIDFVectorizator()
		print(tfidf.feature_names)
		print(tfidf.matrix)
		print(tfidf.vectorizer.vocabulary_)
		print(tfidf.word_tfidf)

if __name__ == '__main__':
	unittest.main()
