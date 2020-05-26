import nltk.corpus
from nltk.text import TextCollection


class TFIDFVectorizator:
	def __init__(self):
		self.collection = TextCollection(nltk.corpus.gutenberg) # TODO: add a set of cvs here as well

	def vectorize(self, word, text):
		return self.collection.tf_idf(word, text)
