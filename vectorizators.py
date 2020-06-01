import nltk.corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import Pickler
import pickle as pic
from nltk.text import TextCollection
import numpy as np

class TFIDFVectorizator:

	COLLECTION_FILE_NAME = "text_collection.obj"

	def __init__(self):
		corpus = [nltk.corpus.gutenberg.raw(f) for f in
				  nltk.corpus.gutenberg.fileids()]  # TODO: add a set of cvs here as well
		# corpus = ["Hello. What's up? I'm fine.", "I'm in love with a beautiful girl.",
		#		  "What is your problem, my love?"]
		try:
			with open(self.COLLECTION_FILE_NAME, 'rb') as collection_file:
				self.vectorizer = pic.load(collection_file)

				print("Loaded collection from file.")
		except FileNotFoundError:
			print("Collection file not found, creating new...")
			self.vectorizer = TfidfVectorizer(stop_words=None)
			self.vectorizer.fit(corpus)
			with open(self.COLLECTION_FILE_NAME, 'wb') as collection_file:
				pic.dump(self.vectorizer, collection_file)
				print("Collection file created and collection loaded")
		self.matrix = self.vectorizer.transform(corpus).todense()
		self.feature_names = self.vectorizer.get_feature_names()

		self.word_tfidf = dict()
		for word in self.vectorizer.vocabulary_.items():
			docs = np.squeeze(np.asarray(self.matrix[:, int(word[1])]))
			avg = sum(docs) / len(docs)
			self.word_tfidf[word[0]] = sum(docs) / len(docs)  # average TF-IDF score

	def vectorize(self, word):
		"""Returns TF-IDF score for given word, or 0 if the word does not exist in vocabulary."""
		return self.word_tfidf.get(word, 0.0)

