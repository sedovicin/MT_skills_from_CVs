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
		#corpus = ["Hello. What's up? I'm fine.", "I'm in love with a beautiful girl.",
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
		for word in self.vectorizer.vocabulary_.items(): # FIXME: is the score inverse maybe?
			docs = np.squeeze(np.asarray(self.matrix[:, int(word[1])]))
			scores = [score for score in docs if not score < 0.0000000001]  # remove scores if 0 or very close to 0
			self.word_tfidf[word[0]] = sum(scores) / len(scores)  # average TF-IDF score

	def vectorize(self, word):
		"""Returns TF-IDF score for given word, or 0 if the word does not exist in vocabulary."""
		return self.word_tfidf.get(word, 0.0)


def get_word2vec(word2vec_path=None, corpus_path=None):
	"""Loads pickled Word2Vec model from path. If the file doesn't exist, creates a new trained one using given corpus.
	If the corpus is None, uses default corpus (Gutenberg).

	:type word2vec_path: str
	:type corpus_path: str
	:return: Word2Vec trained object
	:rtype: gensim.models.Word2Vec
	"""
	from gensim.models import Word2Vec
	if word2vec_path is not None:
		try:
			with open(word2vec_path, 'rb') as collection_file:
				word2vec = pic.load(collection_file)
				print("Loaded Word2Vec from file.")
		except FileNotFoundError:
			print("Word2Vec file not found, creating new...")
			if corpus_path is None:
				corpus = get_gutenberg_corpus()
				print("Got default corpus.")
			else:
				import json
				try:
					with open(corpus_path, 'r', encoding='utf8') as fp_corpus:
						corpus = json.load(fp_corpus)
						print("Loaded corpus from file.")
				except FileNotFoundError:
					corpus = get_gutenberg_corpus()
					print("Corpus file not found, loading default corpus.")
			word2vec = Word2Vec(sentences=corpus)
			with open(word2vec_path, 'wb') as collection_file:
				pic.dump(word2vec, collection_file)
				print("Word2Vec file created and loaded")
	else:
		raise ValueError("Path to file must be given so the method knows where to find or to save model.")
	return word2vec


def get_gutenberg_corpus():
	"""
	:returns: corpus made from NLTK Gutenberg files, in shape of sentences.
	:rtype: list
	"""
	import nltk
	corpus = []
	for f in nltk.corpus.gutenberg.fileids():
		corpus.extend(nltk.corpus.gutenberg.sents(f))
	return corpus
